"""Connection settings for the TRT-LLM backend.

Two HTTP surfaces are involved, both provided by a `trt-llm-explore`-style backend:

* ``chat_url``    — OpenAI-compatible proxy (``/v1/chat/completions``), used by ``ChatTrtLlm``.
* ``control_url`` — Triton KServe v2 API (``/v2/repository/...``), used by ``TrtLlmManager``
                    for model load/unload/index.

All fields are overridable via environment variables so the package carries no hard-coded,
host-specific assumptions (see :meth:`TrtLlmSettings.from_env`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_CHAT_URL = "http://localhost:8003"
DEFAULT_CONTROL_URL = "http://localhost:8000"
DEFAULT_API_KEY = "trt-llm"  # dummy; the OpenAI client requires *some* key


@dataclass(frozen=True)
class TrtLlmSettings:
    """Where the backend lives and how patient to be with it."""

    chat_url: str = DEFAULT_CHAT_URL
    control_url: str = DEFAULT_CONTROL_URL
    api_key: str = DEFAULT_API_KEY

    # Optional shell command that restarts the backend and reclaims VRAM (nothing loaded
    # afterward), enabling automatic restart-based model swaps. e.g.
    # "just -C /home/ken/src/ai/trt-llm-explore restart" or
    # "docker restart trt-llm-explore-triton-1". If None, a swap that needs a restart raises
    # BackendRestartRequiredError instead of acting. See trt-llm-explore sprint 006 / WI #91.
    restart_command: str | None = None

    # Timeouts (seconds)
    request_timeout_s: float = 10.0  # quick control-plane calls (health, index, ready)
    load_timeout_s: float = 60.0  # per-component load POST
    unload_timeout_s: float = 30.0  # per-component unload POST
    ready_timeout_s: float = 30.0  # budget to poll a component to READY after load
    restart_timeout_s: float = 180.0  # budget for the backend to come back healthy after restart

    @classmethod
    def from_env(cls) -> TrtLlmSettings:
        """Build settings from ``TRTLLM_*`` env vars, falling back to localhost defaults."""
        return cls(
            chat_url=os.environ.get("TRTLLM_CHAT_URL", DEFAULT_CHAT_URL).rstrip("/"),
            control_url=os.environ.get("TRTLLM_CONTROL_URL", DEFAULT_CONTROL_URL).rstrip("/"),
            api_key=os.environ.get("TRTLLM_API_KEY", DEFAULT_API_KEY),
            restart_command=os.environ.get("TRTLLM_RESTART_CMD") or None,
        )
