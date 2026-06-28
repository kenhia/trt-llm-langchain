"""Offline tests that ChatTrtLlm composes in LCEL and exposes inherited tool/runnable APIs.

These assert wiring (no backend calls): pipe construction, bind_tools returning a runnable, and
that the model is a proper LangChain Runnable. Live LCEL/tool-calling is verified via examples.
"""

from __future__ import annotations

import httpx
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from trt_llm_langchain import ChatTrtLlm, TrtLlmSettings


class _FakeManager:
    def validate(self, key: str) -> None: ...
    def ensure_loaded(self, key: str) -> None: ...


def _chat() -> ChatTrtLlm:
    http_client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(404)))
    return ChatTrtLlm(
        model="qwen2_5-coder-7b-fp16",
        settings=TrtLlmSettings(chat_url="http://backend"),
        manager=_FakeManager(),
        http_client=http_client,
    )


def test_is_runnable() -> None:
    assert isinstance(_chat(), Runnable)


def test_lcel_pipe_constructs() -> None:
    prompt = ChatPromptTemplate.from_messages([("human", "{q}")])
    chain = prompt | _chat() | StrOutputParser()
    assert isinstance(chain, Runnable)  # no network touched by composition


def test_bind_tools_returns_runnable() -> None:
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    bound = _chat().bind_tools([add])
    assert isinstance(bound, Runnable)
