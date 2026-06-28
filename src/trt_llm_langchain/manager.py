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

import time
from dataclasses import dataclass, field

import httpx

from .config import TrtLlmSettings
from .errors import (
    ModelLoadError,
    ModelNotFoundError,
    ModelUnloadError,
    ServerUnavailableError,
)

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
    ) -> None:
        self.settings = settings or TrtLlmSettings.from_env()
        # ``client`` is an injection seam for tests (e.g. httpx.MockTransport).
        self._client = client or httpx.Client(
            base_url=self.settings.control_url,
            timeout=self.settings.request_timeout_s,
        )

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
                raise ModelLoadError(
                    f"Failed to load {component}: HTTP {resp.status_code} {resp.text!r}. "
                    "Insufficient VRAM (unload another model first) or the model repo is "
                    "missing for this key are the usual causes."
                )
            self._wait_ready(component)

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
        """Make ``key`` the resident model.

        No-op if it is already responsive. Otherwise unload every other loaded model first
        (a single 5090 holds one model at a time), then load ``key``.
        """
        self.validate(key)
        if self.is_responsive(key):
            return
        for other in self.loaded_keys():
            if other != key:
                try:
                    self.unload(other)
                except ModelUnloadError:
                    # Best effort — proceed; the load below will fail loudly on real OOM.
                    pass
        self.load(key)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> TrtLlmManager:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
