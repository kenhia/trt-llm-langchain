"""trt-llm-langchain — use TensorRT-LLM models from LangChain.

``ChatTrtLlm`` is a call-site drop-in for ``ChatAnthropic``/``ChatOpenAI`` backed by a local
TensorRT-LLM server, with lazy model load/unload via :class:`TrtLlmManager`.
"""

from .chat import ChatTrtLlm
from .config import TrtLlmSettings
from .errors import (
    BackendRestartRequiredError,
    InsufficientVramError,
    ModelLoadError,
    ModelNotFoundError,
    ModelUnloadError,
    ResidentModelError,
    ServerUnavailableError,
    TrtLlmError,
)
from .manager import ModelInfo, TrtLlmManager, free_vram_gb

__all__ = [
    "ChatTrtLlm",
    "TrtLlmSettings",
    "TrtLlmManager",
    "ModelInfo",
    "free_vram_gb",
    "TrtLlmError",
    "ServerUnavailableError",
    "ModelNotFoundError",
    "ResidentModelError",
    "ModelLoadError",
    "InsufficientVramError",
    "BackendRestartRequiredError",
    "ModelUnloadError",
]
