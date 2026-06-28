"""Offline tests for VRAM/OOM handling on load (the client-side half of WI #91)."""

from __future__ import annotations

import httpx
import pytest

from trt_llm_langchain import InsufficientVramError, ModelLoadError, TrtLlmManager, TrtLlmSettings
from trt_llm_langchain.manager import _looks_like_oom

_OOM_BODY = (
    '{"error":"load failed for model \'tensorrt_llm_qwen2_5-coder-7b-fp16\': version 1 is at '
    "UNAVAILABLE state: Internal: unexpected error when creating modelInstanceState: "
    '[TensorRT-LLM][ERROR] CUDA runtime error in ::cudaMallocAsync(...): out of memory"}'
)

_INDEX = [
    {"name": f"{p}_qwen2_5-coder-7b-fp16", "version": "1", "state": "UNAVAILABLE"}
    for p in ("preprocessing", "tensorrt_llm", "postprocessing", "tensorrt_llm_bls", "ensemble")
]


def test_looks_like_oom() -> None:
    assert _looks_like_oom(_OOM_BODY)
    assert _looks_like_oom("CUDA error: OutOfMemory")
    assert not _looks_like_oom("model repository does not contain model")


def _manager(load_status: int, load_body: str) -> TrtLlmManager:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v2/repository/index":
            return httpx.Response(200, json=_INDEX)
        if path.endswith("/ready"):
            return httpx.Response(400)  # never becomes ready in this test
        if path.endswith("/load"):
            return httpx.Response(load_status, text=load_body)
        return httpx.Response(404)

    client = httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler))
    return TrtLlmManager(TrtLlmSettings(), client=client)


def test_oom_load_raises_insufficient_vram() -> None:
    mgr = _manager(400, _OOM_BODY)
    with pytest.raises(InsufficientVramError) as ei:
        mgr.load("qwen2_5-coder-7b-fp16")
    msg = str(ei.value)
    assert "restart" in msg.lower()
    assert "WI #91" in msg


def test_non_oom_load_failure_raises_plain_model_load_error() -> None:
    mgr = _manager(400, '{"error":"model repository does not contain model"}')
    with pytest.raises(ModelLoadError) as ei:
        mgr.load("qwen2_5-coder-7b-fp16")
    assert not isinstance(ei.value, InsufficientVramError)


def test_insufficient_vram_is_a_model_load_error() -> None:
    # callers catching the broad type still work
    assert issubclass(InsufficientVramError, ModelLoadError)
