"""``ChatTrtLlm`` — a LangChain chat model backed by a local TensorRT-LLM server.

Designed as a call-site drop-in for ``ChatAnthropic`` / ``ChatOpenAI``::

    chat = ChatTrtLlm(model="qwen2_5-coder-7b-fp16")
    chat.invoke("Write a haiku about CUDA")

It subclasses :class:`langchain_openai.ChatOpenAI` (pointed at the backend's OpenAI-compatible
proxy) so all of invoke/stream/batch/async/``bind_tools``/``with_structured_output`` come for
free. The only thing it adds is **lazy model management**: before the first generation it calls
:meth:`TrtLlmManager.ensure_loaded`, which makes the requested model resident — unloading any
other model first, since a single GPU holds one at a time.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_openai import ChatOpenAI
from pydantic import PrivateAttr

from .config import TrtLlmSettings
from .manager import TrtLlmManager


class ChatTrtLlm(ChatOpenAI):
    """Chat model for a local TensorRT-LLM backend, with lazy load/unload.

    Args:
        model: Backend model key (the OpenAI ``id``), e.g. ``"qwen2_5-coder-7b-fp16"``.
        settings: Connection settings. Defaults to :meth:`TrtLlmSettings.from_env`.
        eager_load: If True, load the model during construction instead of lazily on first call.
        manager: Pre-built manager (injection seam for tests); defaults to one from ``settings``.
        **kwargs: Forwarded to ``ChatOpenAI`` (``temperature``, ``max_tokens``, ``http_client``…).
    """

    _settings: TrtLlmSettings = PrivateAttr()
    _manager: TrtLlmManager = PrivateAttr()
    _ensured: bool = PrivateAttr(default=False)

    def __init__(
        self,
        model: str,
        *,
        settings: TrtLlmSettings | None = None,
        eager_load: bool = False,
        manager: TrtLlmManager | None = None,
        **kwargs: Any,
    ) -> None:
        resolved = settings or TrtLlmSettings.from_env()
        kwargs.setdefault("stream_usage", True)
        super().__init__(
            model=model,
            base_url=f"{resolved.chat_url}/v1",
            api_key=resolved.api_key,
            **kwargs,
        )
        self._settings = resolved
        self._manager = manager or TrtLlmManager(resolved)
        self._manager.validate(model)  # raises ModelNotFoundError if unknown
        self._ensured = False
        if eager_load:
            self._manager.ensure_loaded(model)
            self._ensured = True

    @property
    def _llm_type(self) -> str:
        return "trt-llm"

    # -- model management hooks --------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if not self._ensured:
            self._manager.ensure_loaded(self.model_name)
            self._ensured = True

    async def _aensure_loaded(self) -> None:
        # ensure_loaded is sync (and a load can take seconds), so run it off the event loop.
        if not self._ensured:
            await asyncio.to_thread(self._manager.ensure_loaded, self.model_name)
            self._ensured = True

    # -- generation overrides: make the model resident, then delegate ------------------

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self._ensure_loaded()
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def _stream(self, *args: Any, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        self._ensure_loaded()
        yield from super()._stream(*args, **kwargs)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        await self._aensure_loaded()
        return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _astream(self, *args: Any, **kwargs: Any) -> AsyncIterator[ChatGenerationChunk]:
        await self._aensure_loaded()
        async for chunk in super()._astream(*args, **kwargs):
            yield chunk
