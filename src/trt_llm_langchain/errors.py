"""Exception hierarchy for trt-llm-langchain.

All errors derive from :class:`TrtLlmError` so callers can catch the whole family.
"""

from __future__ import annotations

from collections.abc import Iterable


class TrtLlmError(Exception):
    """Base class for all trt-llm-langchain errors."""


class ServerUnavailableError(TrtLlmError):
    """The backend (KServe control plane or OpenAI proxy) could not be reached."""

    def __init__(self, url: str, detail: str | None = None) -> None:
        msg = f"TRT-LLM backend unreachable at {url}"
        if detail:
            msg += f" ({detail})"
        msg += ". Is the serving stack running?"
        super().__init__(msg)
        self.url = url


class ModelNotFoundError(TrtLlmError):
    """Requested model key is not present in the backend's registry."""

    def __init__(self, model: str, available: Iterable[str] | None = None) -> None:
        msg = f"Model {model!r} is not available on the backend"
        avail = sorted(available) if available is not None else None
        if avail:
            msg += f". Available models: {', '.join(avail)}"
        elif avail == []:
            msg += " (the backend reports no models — check that engines are built/set up)"
        super().__init__(msg)
        self.model = model
        self.available = avail


class ModelLoadError(TrtLlmError):
    """A model failed to load (HTTP error or readiness timeout)."""


class InsufficientVramError(ModelLoadError):
    """A load failed with a CUDA out-of-memory error.

    On a single GPU this usually means the previously-loaded model's VRAM was not reclaimed on
    unload (TensorRT-LLM's cudaMallocAsync pool retains pages), so a backend restart is typically
    required before a different model will fit. See trt-llm-explore WI #91.
    """


class BackendRestartRequiredError(ModelLoadError):
    """A swap to a different model needs a backend restart that isn't configured.

    On a single GPU, unload does not free VRAM, so loading a different model requires restarting
    the backend first. Configure ``restart_command`` (or ``TRTLLM_RESTART_CMD``) / pass a
    ``restart_backend`` callable to let the client do this automatically; otherwise restart the
    backend manually (``just swap <key>`` / ``just restart`` in trt-llm-explore) and retry.
    """

    def __init__(self, target: str, current: list[str]) -> None:
        self.target = target
        self.current = current
        super().__init__(
            f"Switching to {target!r} requires reclaiming VRAM held by currently-loaded "
            f"model(s) {current}. On a single GPU, unload does not free VRAM (TensorRT-LLM pool "
            "retention), so the backend must be restarted before a different model will load. "
            f"Restart it (e.g. `just swap {target}` or `just restart` in trt-llm-explore, or "
            "restart the Triton container) and retry — or set TRTLLM_RESTART_CMD / pass "
            "restart_backend so the client does it automatically. See trt-llm-explore sprint 006."
        )


class ModelUnloadError(TrtLlmError):
    """A model failed to unload cleanly."""
