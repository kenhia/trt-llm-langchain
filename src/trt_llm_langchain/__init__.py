"""trt-llm-langchain — use TensorRT-LLM models from LangChain.

``ChatTrtLlm`` is a call-site drop-in for ``ChatAnthropic``/``ChatOpenAI`` backed by a local
TensorRT-LLM server, with lazy model load/unload via :class:`TrtLlmManager`.
"""

from .chat import ChatTrtLlm
from .config import TrtLlmSettings
from .errors import (
    ModelLoadError,
    ModelNotFoundError,
    ModelUnloadError,
    ServerUnavailableError,
    TrtLlmError,
)
from .manager import ModelInfo, TrtLlmManager

__all__ = [
    "ChatTrtLlm",
    "TrtLlmSettings",
    "TrtLlmManager",
    "ModelInfo",
    "TrtLlmError",
    "ServerUnavailableError",
    "ModelNotFoundError",
    "ModelLoadError",
    "ModelUnloadError",
]
