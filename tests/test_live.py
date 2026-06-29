"""Opt-in live integration tests against a real backend.

These are skipped unless TRTLLM_LIVE=1 and the backend is reachable. They use the currently
resident model where possible to avoid forcing a heavy load/restart. The swap test is gated
further (TRTLLM_LIVE_SWAP=1 + TRTLLM_RESTART_CMD) because it restarts the backend.

Run:  TRTLLM_LIVE=1 uv run pytest -q -m live
"""

from __future__ import annotations

import os

import pytest

from trt_llm_langchain import ChatTrtLlm, TrtLlmManager, TrtLlmSettings

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.environ.get("TRTLLM_LIVE"), reason="set TRTLLM_LIVE=1 to run"),
]


@pytest.fixture(scope="module")
def mgr() -> TrtLlmManager:
    m = TrtLlmManager(TrtLlmSettings.from_env())
    if not m.is_healthy():
        pytest.skip("backend control plane not reachable")
    return m


def _a_chat_model(mgr: TrtLlmManager) -> str:
    """A resident chat model if one is loaded, else the first available chat model."""
    models = mgr.models()
    chat = [k for k, i in models.items() if not i.is_vision]
    if not chat:
        pytest.skip("no chat models available on backend")
    for k in chat:
        if models[k].loaded:
            return k
    return sorted(chat)[0]


def test_list_models(mgr: TrtLlmManager) -> None:
    assert mgr.available_keys(), "backend reports no models"


def test_invoke_and_stream(mgr: TrtLlmManager) -> None:
    model = _a_chat_model(mgr)
    chat = ChatTrtLlm(model=model, max_tokens=32, settings=mgr.settings)
    assert chat.invoke("Say the word OK.").content.strip()
    chunks = [c.content for c in chat.stream("Count to 3.")]
    assert "".join(chunks).strip()


def test_bind_tools_live(mgr: TrtLlmManager) -> None:
    # Tool-calling works non-streaming on tool-capable models (llama/qwen/mistral families);
    # some models (e.g. phi-3.5) have no native tool support, so skip if none are emitted.
    model = _a_chat_model(mgr)

    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    chat = ChatTrtLlm(model=model, max_tokens=64, temperature=0, settings=mgr.settings).bind_tools(
        [add]
    )
    msg = chat.invoke("What is 2 + 3? Use the add tool.")
    if not msg.tool_calls:
        pytest.skip(f"{model} emitted no tool_calls (model may lack native tool support)")
    call = msg.tool_calls[0]
    assert call["name"] == "add"
    assert call["args"] == {"a": 2, "b": 3}


def test_structured_output_json_mode(mgr: TrtLlmManager) -> None:
    from langchain_core.exceptions import OutputParserException
    from pydantic import BaseModel

    class Person(BaseModel):
        name: str
        age: int

    model = _a_chat_model(mgr)
    chat = ChatTrtLlm(model=model, max_tokens=128, temperature=0, settings=mgr.settings)
    structured = chat.with_structured_output(Person, method="json_mode")
    try:
        # json_mode relies on prompt adherence (the proxy doesn't enforce response_format), so
        # be directive. Adherence is model-dependent — skip rather than fail if a model adds prose.
        out = structured.invoke(
            'Output only a JSON object with keys "name" and "age". Person: Ada Lovelace was 36.'
        )
    except OutputParserException:
        pytest.skip(f"{model} did not produce clean JSON for json_mode (model-dependent)")
    assert out.name and isinstance(out.age, int)


@pytest.mark.skipif(
    not (os.environ.get("TRTLLM_LIVE_SWAP") and os.environ.get("TRTLLM_RESTART_CMD")),
    reason="set TRTLLM_LIVE_SWAP=1 and TRTLLM_RESTART_CMD to run the restart-based swap test",
)
def test_restart_based_swap(mgr: TrtLlmManager) -> None:
    chat_models = [k for k, i in mgr.models().items() if not i.is_vision]
    if len(chat_models) < 2:
        pytest.skip("need >=2 chat models for a swap test")
    loaded = mgr.loaded_keys()
    target = next(k for k in sorted(chat_models) if k not in loaded)
    ChatTrtLlm(model=target, max_tokens=16, settings=mgr.settings).invoke("hi")
    assert mgr.is_responsive(target)
