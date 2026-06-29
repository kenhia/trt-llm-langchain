"""Model lifecycle manager over the Triton KServe v2 control plane.

A ``trt-llm-explore``-style backend serves each model as a 5-component *ensemble* whose
components are named ``{pipeline}_{model_key}`` (e.g. ``ensemble_qwen2_5-coder-7b-fp16``).
This manager drives load/unload and reads state via the KServe repository API, exposing a
small, exception-raising library surface:

* :meth:`TrtLlmManager.models` / :meth:`available_keys` — what the backend knows about.
* :meth:`is_responsive` / :meth:`loaded_keys` — what is resident right now.
* :meth:`load` / :meth:`unload` — explicit lifecycle control.
* :meth:`ensure_loaded` — make a model resident, unloading others first (single-GPU swap).

The component naming and pipeline sets below mirror ``trt-llm-explore``'s
``src/common.py``; they are the backend *contract* this client depends on (to be formalized in
Phase 2, see ``docs/02-plan.md``).
"""

from __future__ import annotations

import shlex
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx

from .config import TrtLlmSettings
from .errors import (
    BackendRestartRequiredError,
    InsufficientVramError,
    ModelLoadError,
    ModelNotFoundError,
    ModelUnloadError,
    ResidentModelError,
    ServerUnavailableError,
)


def free_vram_gb() -> float | None:
    """Best-effort free VRAM (GB) of GPU 0 via nvidia-smi; None if unavailable.

    Only meaningful when this client runs on the same host as the GPU.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None
        return float(result.stdout.strip().splitlines()[0]) / 1024.0
    except (FileNotFoundError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def _looks_like_oom(text: str) -> bool:
    low = text.lower()
    return "out of memory" in low or "cudamalloc" in low or "outofmemory" in low

# Component pipelines per model, in dependency (load) order. Unload walks these in reverse.
PIPELINE_NAMES: list[str] = [
    "preprocessing",
    "tensorrt_llm",
    "postprocessing",
    "tensorrt_llm_bls",
    "ensemble",
]
MULTIMODAL_PIPELINE_NAMES: list[str] = [
    "preprocessing",
    "tensorrt_llm",
    "postprocessing",
    "multimodal_encoders",
    "ensemble",
]

# Longest-first so prefix stripping is unambiguous (e.g. ``tensorrt_llm_bls`` before
# ``tensorrt_llm``).
_ALL_PIPELINES: list[str] = sorted(
    set(PIPELINE_NAMES) | set(MULTIMODAL_PIPELINE_NAMES), key=len, reverse=True
)

# A component unique to each pipeline kind, used to classify a model from the index.
_VISION_MARKER = "multimodal_encoders"
_STANDARD_MARKER = "tensorrt_llm_bls"


@dataclass
class ModelInfo:
    """Derived state for one model key, parsed from the KServe repository index."""

    key: str
    is_vision: bool
    components: dict[str, str] = field(default_factory=dict)  # pipeline -> Triton state

    @property
    def pipelines(self) -> list[str]:
        return MULTIMODAL_PIPELINE_NAMES if self.is_vision else PIPELINE_NAMES

    @property
    def ensemble_name(self) -> str:
        """The component used as the model id for inference / readiness checks."""
        return f"ensemble_{self.key}"

    @property
    def loaded(self) -> bool:
        """True when the ensemble component is READY (i.e. servable)."""
        return self.components.get("ensemble") == "READY"


def _split_component(name: str) -> tuple[str, str] | None:
    """Split ``{pipeline}_{key}`` into ``(pipeline, key)``; None if no pipeline prefix."""
    for pipeline in _ALL_PIPELINES:
        prefix = f"{pipeline}_"
        if name.startswith(prefix):
            return pipeline, name[len(prefix) :]
    return None


class TrtLlmManager:
    """Drives model load/unload against the Triton KServe v2 control plane."""

    def __init__(
        self,
        settings: TrtLlmSettings | None = None,
        *,
        client: httpx.Client | None = None,
        restart_backend: Callable[[], None] | None = None,
    ) -> None:
        self.settings = settings or TrtLlmSettings.from_env()
        # ``client`` is an injection seam for tests (e.g. httpx.MockTransport).
        self._client = client or httpx.Client(
            base_url=self.settings.control_url,
            timeout=self.settings.request_timeout_s,
        )
        # Optional restart strategy: explicit callable wins; else built from settings.restart_command.
        self._restart_backend = restart_backend

    # -- low-level HTTP ----------------------------------------------------------------

    def _get(self, path: str) -> httpx.Response:
        try:
            return self._client.get(path)
        except httpx.HTTPError as exc:  # connection refused, timeout, etc.
            raise ServerUnavailableError(self.settings.control_url, str(exc)) from exc

    def _post(self, path: str, *, timeout: float | None = None) -> httpx.Response:
        try:
            return self._client.post(path, json={}, timeout=timeout)
        except httpx.HTTPError as exc:
            raise ServerUnavailableError(self.settings.control_url, str(exc)) from exc

    # -- health & discovery ------------------------------------------------------------

    def is_healthy(self) -> bool:
        """True if the Triton server answers its readiness probe."""
        try:
            return self._get("/v2/health/ready").status_code == 200
        except ServerUnavailableError:
            return False

    def _index(self) -> list[dict]:
        """Return the raw KServe repository index (all models, any state)."""
        resp = self._post("/v2/repository/index")
        if resp.status_code != 200:
            raise ServerUnavailableError(
                self.settings.control_url,
                f"repository index returned HTTP {resp.status_code}",
            )
        return resp.json()

    def models(self) -> dict[str, ModelInfo]:
        """Parse the index into ``{model_key: ModelInfo}``.

        A model's kind (vision vs standard) is inferred from which marker component is present:
        ``multimodal_encoders_*`` => vision, ``tensorrt_llm_bls_*`` => standard.
        """
        by_key: dict[str, dict[str, str]] = {}
        for entry in self._index():
            split = _split_component(entry.get("name", ""))
            if split is None:
                continue
            pipeline, key = split
            by_key.setdefault(key, {})[pipeline] = entry.get("state", "")

        out: dict[str, ModelInfo] = {}
        for key, components in by_key.items():
            is_vision = _VISION_MARKER in components and _STANDARD_MARKER not in components
            out[key] = ModelInfo(key=key, is_vision=is_vision, components=components)
        return out

    def available_keys(self) -> list[str]:
        """Sorted list of model keys the backend knows about (loaded or not)."""
        return sorted(self.models())

    def loaded_keys(self) -> list[str]:
        """Model keys whose ensemble is currently READY."""
        return sorted(k for k, info in self.models().items() if info.loaded)

    def is_responsive(self, key: str) -> bool:
        """True if ``ensemble_{key}`` answers its readiness probe right now."""
        try:
            return self._get(f"/v2/models/ensemble_{key}/ready").status_code == 200
        except ServerUnavailableError:
            return False

    def _require(self, key: str) -> ModelInfo:
        models = self.models()
        if key not in models:
            raise ModelNotFoundError(key, available=models.keys())
        return models[key]

    def validate(self, key: str) -> None:
        """Raise :class:`ModelNotFoundError` if ``key`` is not in the backend registry."""
        self._require(key)

    def resident_model(self) -> str:
        """The single currently-loaded model key (for ``ChatTrtLlm()`` with no explicit model).

        Raises :class:`ResidentModelError` if zero or more than one model is loaded.
        """
        loaded = self.loaded_keys()
        if len(loaded) == 1:
            return loaded[0]
        if not loaded:
            raise ResidentModelError(
                "No model is currently loaded. Pass model=... to ChatTrtLlm, or load one first "
                "(e.g. `trtllm-lc load <key>`)."
            )
        raise ResidentModelError(
            f"Multiple models are loaded ({', '.join(loaded)}). Pass model=... to choose one."
        )

    # -- lifecycle ---------------------------------------------------------------------

    def load(self, key: str) -> None:
        """Load all components for ``key`` in dependency order, polling each to READY."""
        info = self._require(key)
        for pipeline in info.pipelines:
            component = f"{pipeline}_{key}"
            resp = self._post(
                f"/v2/repository/models/{component}/load",
                timeout=self.settings.load_timeout_s,
            )
            if resp.status_code != 200:
                self._raise_load_error(component, resp)
            self._wait_ready(component)

    def _raise_load_error(self, component: str, resp: httpx.Response) -> None:
        text = resp.text
        if _looks_like_oom(text):
            free = free_vram_gb()
            free_str = f"{free:.1f} GB free" if free is not None else "free VRAM unknown"
            raise InsufficientVramError(
                f"Out of VRAM loading {component} ({free_str}). On a single GPU the previous "
                "model's memory is often not reclaimed on unload (TensorRT-LLM pool retention), "
                "so a backend restart is usually required before a different model will fit. "
                "See trt-llm-explore WI #91."
            )
        raise ModelLoadError(
            f"Failed to load {component}: HTTP {resp.status_code} {text!r}. "
            "A missing model repo for this key is the usual non-VRAM cause."
        )

    def _wait_ready(self, component: str) -> None:
        deadline = self.settings.ready_timeout_s
        waited = 0.0
        while waited < deadline:
            if self._get(f"/v2/models/{component}/ready").status_code == 200:
                return
            time.sleep(1.0)
            waited += 1.0
        raise ModelLoadError(
            f"Timed out after {deadline:.0f}s waiting for {component} to become READY"
        )

    def unload(self, key: str) -> None:
        """Unload all components for ``key`` in reverse dependency order (best effort)."""
        info = self._require(key)
        failures: list[str] = []
        for pipeline in reversed(info.pipelines):
            component = f"{pipeline}_{key}"
            resp = self._post(
                f"/v2/repository/models/{component}/unload",
                timeout=self.settings.unload_timeout_s,
            )
            if resp.status_code != 200:
                failures.append(f"{component} (HTTP {resp.status_code})")
        if failures:
            raise ModelUnloadError(f"Unload incomplete for {key}: {', '.join(failures)}")

    def ensure_loaded(self, key: str) -> None:
        """Make ``key`` the resident model, using a **restart-based** swap on a single GPU.

        No-op if ``key`` is already responsive. If a *different* model is loaded, a clean unload
        would not free its VRAM (TensorRT-LLM pool retention; see trt-llm-explore sprint 006 /
        WI #91), so we restart the backend to reclaim VRAM, then load ``key``. If no restart
        strategy is configured, raise :class:`BackendRestartRequiredError` with guidance.
        """
        self.validate(key)
        if self.is_responsive(key):
            return

        others = [k for k in self.loaded_keys() if k != key]
        if others:
            if not self.can_restart():
                raise BackendRestartRequiredError(target=key, current=others)
            self.restart_backend()  # reclaims VRAM; backend comes back with nothing loaded
            if self.is_responsive(key):
                # A restart hook that also loads (e.g. `just swap <key>`) may already be done.
                return

        self.load(key)

    # -- restart strategy --------------------------------------------------------------

    def can_restart(self) -> bool:
        """Whether an automatic backend restart is available (callable or configured command)."""
        return self._restart_backend is not None or bool(self.settings.restart_command)

    def restart_backend(self) -> None:
        """Restart the backend to reclaim VRAM, then wait for it to become healthy.

        Uses the injected ``restart_backend`` callable if present, else runs
        ``settings.restart_command``. Raises :class:`BackendRestartRequiredError`-family /
        :class:`ModelLoadError` on failure.
        """
        if self._restart_backend is not None:
            self._restart_backend()
        elif self.settings.restart_command:
            self._run_restart_command(self.settings.restart_command)
        else:  # pragma: no cover - guarded by can_restart() at call sites
            raise ModelLoadError("No restart strategy configured")
        self._wait_healthy(self.settings.restart_timeout_s)

    def _run_restart_command(self, command: str) -> None:
        try:
            proc = subprocess.run(
                shlex.split(command),
                capture_output=True,
                text=True,
                timeout=self.settings.restart_timeout_s,
            )
        except (FileNotFoundError, subprocess.SubprocessError) as exc:
            raise ModelLoadError(f"Restart command failed to run ({command!r}): {exc}") from exc
        if proc.returncode != 0:
            raise ModelLoadError(
                f"Restart command exited {proc.returncode} ({command!r}): "
                f"{proc.stderr.strip()[:300]}"
            )

    def _wait_healthy(self, timeout_s: float) -> None:
        waited = 0.0
        while waited < timeout_s:
            if self.is_healthy():
                return
            time.sleep(1.0)
            waited += 1.0
        raise ServerUnavailableError(
            self.settings.control_url, f"did not become healthy within {timeout_s:.0f}s of restart"
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> TrtLlmManager:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
