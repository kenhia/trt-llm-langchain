"""Offline unit tests for ChatTrtLlm.

The OpenAI HTTP surface is faked with httpx.MockTransport (injected via ``http_client``); the
model manager is faked to record load calls. No backend required. Live end-to-end coverage
lands in Sprint 4.
"""

from __future__ import annotations

import httpx
import pytest

from trt_llm_langchain import ChatTrtLlm, TrtLlmSettings


class FakeManager:
    """Duck-typed stand-in for TrtLlmManager recording its calls."""

    def __init__(self, known: set[str]) -> None:
        self.known = known
        self.validated: list[str] = []
        self.ensured: list[str] = []

    def validate(self, key: str) -> None:
        from trt_llm_langchain import ModelNotFoundError

        self.validated.append(key)
        if key not in self.known:
            raise ModelNotFoundError(key, available=self.known)

    def ensure_loaded(self, key: str) -> None:
        self.ensured.append(key)


def _completion(content: str, model: str) -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }


def _chat(manager: FakeManager, reply: str = "hello from trt", **kwargs) -> ChatTrtLlm:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        return httpx.Response(200, json=_completion(reply, "qwen2_5-coder-7b-fp16"))

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    return ChatTrtLlm(
        model="qwen2_5-coder-7b-fp16",
        settings=TrtLlmSettings(chat_url="http://backend"),
        manager=manager,
        http_client=http_client,
        **kwargs,
    )


def test_validates_on_construction() -> None:
    mgr = FakeManager(known={"qwen2_5-coder-7b-fp16"})
    _chat(mgr)
    assert mgr.validated == ["qwen2_5-coder-7b-fp16"]


def test_unknown_model_raises() -> None:
    from trt_llm_langchain import ModelNotFoundError

    with pytest.raises(ModelNotFoundError):
        _chat(FakeManager(known=set()))


def test_lazy_load_then_invoke() -> None:
    mgr = FakeManager(known={"qwen2_5-coder-7b-fp16"})
    chat = _chat(mgr)
    # construction must NOT load the model
    assert mgr.ensured == []

    resp = chat.invoke("hi")
    assert resp.content == "hello from trt"
    # first generation loaded exactly once, with the right key
    assert mgr.ensured == ["qwen2_5-coder-7b-fp16"]

    chat.invoke("again")
    assert mgr.ensured == ["qwen2_5-coder-7b-fp16"]  # idempotent, not reloaded


def test_eager_load_loads_in_constructor() -> None:
    mgr = FakeManager(known={"qwen2_5-coder-7b-fp16"})
    _chat(mgr, eager_load=True)
    assert mgr.ensured == ["qwen2_5-coder-7b-fp16"]


def test_llm_type() -> None:
    assert _chat(FakeManager(known={"qwen2_5-coder-7b-fp16"}))._llm_type == "trt-llm"
