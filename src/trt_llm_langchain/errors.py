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
    """A model failed to load (HTTP error, readiness timeout, or insufficient VRAM)."""


class ModelUnloadError(TrtLlmError):
    """A model failed to unload cleanly."""
