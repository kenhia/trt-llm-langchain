"""Offline unit tests for TrtLlmManager using an httpx mock transport.

These exercise index parsing, vision classification, and the ensure_loaded
unload-before-load logic without a running backend. Live integration tests come in Sprint 4.
"""

from __future__ import annotations

import httpx
import pytest

from trt_llm_langchain import (
    BackendRestartRequiredError,
    ModelNotFoundError,
    ResidentModelError,
    TrtLlmManager,
    TrtLlmSettings,
)


def _index_payload(loaded: set[str]) -> list[dict]:
    """Build a KServe-style index for one chat model and one vision model.

    Components belonging to a key in ``loaded`` report state READY; others UNAVAILABLE.
    """
    chat_pipelines = ["preprocessing", "tensorrt_llm", "postprocessing", "tensorrt_llm_bls", "ensemble"]
    vision_pipelines = ["preprocessing", "tensorrt_llm", "postprocessing", "multimodal_encoders", "ensemble"]
    entries = []
    for key, pipelines in [
        ("qwen2_5-coder-7b-fp16", chat_pipelines),
        ("qwen2-vl-7b-fp16", vision_pipelines),
    ]:
        for p in pipelines:
            entries.append(
                {
                    "name": f"{p}_{key}",
                    "version": "1",
                    "state": "READY" if key in loaded else "UNAVAILABLE",
                }
            )
    return entries


_PIPELINES = (
    "tensorrt_llm_bls",
    "multimodal_encoders",
    "preprocessing",
    "postprocessing",
    "tensorrt_llm",
    "ensemble",
)


def _key_of(component: str) -> str:
    """Strip the longest matching pipeline prefix to recover the model key."""
    for p in _PIPELINES:
        if component.startswith(f"{p}_"):
            return component[len(p) + 1 :]
    return component


class Backend:
    """Stateful fake KServe control plane for MockTransport."""

    def __init__(self, loaded: set[str] | None = None) -> None:
        self.loaded = set(loaded or set())
        self.calls: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.calls.append(f"{request.method} {path}")

        if path == "/v2/health/ready":
            return httpx.Response(200)
        if path == "/v2/repository/index":
            return httpx.Response(200, json=_index_payload(self.loaded))

        # readiness: /v2/models/{component}/ready
        if path.startswith("/v2/models/") and path.endswith("/ready"):
            comp = path[len("/v2/models/") : -len("/ready")]
            return httpx.Response(200 if _key_of(comp) in self.loaded else 400)

        # load / unload: /v2/repository/models/{component}/{load|unload}
        if path.startswith("/v2/repository/models/"):
            rest = path[len("/v2/repository/models/") :]
            comp, action = rest.rsplit("/", 1)
            key = _key_of(comp)
            if action == "load":
                self.loaded.add(key)
            elif action == "unload":
                self.loaded.discard(key)
            return httpx.Response(200)

        return httpx.Response(404)


def _manager(backend: Backend, **kw) -> TrtLlmManager:
    settings = TrtLlmSettings(ready_timeout_s=2.0)
    client = httpx.Client(base_url="http://test", transport=httpx.MockTransport(backend.handler))
    return TrtLlmManager(settings, client=client, **kw)


def test_models_parsing_and_vision_classification() -> None:
    mgr = _manager(Backend(loaded=set()))
    models = mgr.models()
    assert set(models) == {"qwen2_5-coder-7b-fp16", "qwen2-vl-7b-fp16"}
    assert models["qwen2_5-coder-7b-fp16"].is_vision is False
    assert models["qwen2-vl-7b-fp16"].is_vision is True
    assert mgr.available_keys() == ["qwen2-vl-7b-fp16", "qwen2_5-coder-7b-fp16"]


def test_loaded_and_responsive() -> None:
    mgr = _manager(Backend(loaded={"qwen2_5-coder-7b-fp16"}))
    assert mgr.loaded_keys() == ["qwen2_5-coder-7b-fp16"]
    assert mgr.is_responsive("qwen2_5-coder-7b-fp16") is True
    assert mgr.is_responsive("qwen2-vl-7b-fp16") is False


def test_validate_unknown_model_raises() -> None:
    mgr = _manager(Backend())
    with pytest.raises(ModelNotFoundError) as ei:
        mgr.validate("does-not-exist")
    assert "qwen2_5-coder-7b-fp16" in str(ei.value)  # lists available


def test_resident_model_single() -> None:
    mgr = _manager(Backend(loaded={"qwen2_5-coder-7b-fp16"}))
    assert mgr.resident_model() == "qwen2_5-coder-7b-fp16"


def test_resident_model_none_raises() -> None:
    mgr = _manager(Backend(loaded=set()))
    with pytest.raises(ResidentModelError) as ei:
        mgr.resident_model()
    assert "No model" in str(ei.value)


def test_resident_model_multiple_raises() -> None:
    mgr = _manager(Backend(loaded={"qwen2_5-coder-7b-fp16", "qwen2-vl-7b-fp16"}))
    with pytest.raises(ResidentModelError) as ei:
        mgr.resident_model()
    assert "Multiple" in str(ei.value)


def test_ensure_loaded_noop_when_already_responsive() -> None:
    backend = Backend(loaded={"qwen2_5-coder-7b-fp16"})
    mgr = _manager(backend)
    mgr.ensure_loaded("qwen2_5-coder-7b-fp16")
    assert not any("/load" in c or "/unload" in c for c in backend.calls)


def test_ensure_loaded_fresh_load_needs_no_restart() -> None:
    backend = Backend(loaded=set())
    mgr = _manager(backend)
    mgr.ensure_loaded("qwen2_5-coder-7b-fp16")
    assert backend.loaded == {"qwen2_5-coder-7b-fp16"}


def test_ensure_loaded_swap_without_restart_raises() -> None:
    # A different model is resident and no restart strategy is configured.
    backend = Backend(loaded={"qwen2-vl-7b-fp16"})
    mgr = _manager(backend)
    with pytest.raises(BackendRestartRequiredError) as ei:
        mgr.ensure_loaded("qwen2_5-coder-7b-fp16")
    assert ei.value.target == "qwen2_5-coder-7b-fp16"
    assert "qwen2-vl-7b-fp16" in ei.value.current
    # crucially: did NOT do an in-place unload/load (which would OOM on real hardware)
    assert not any("/load" in c or "/unload" in c for c in backend.calls)


def test_ensure_loaded_swap_with_restart_callable() -> None:
    # restart hook simulates VRAM reclaim: backend comes back with nothing loaded.
    backend = Backend(loaded={"qwen2-vl-7b-fp16"})
    restarts: list[int] = []

    def restart() -> None:
        restarts.append(1)
        backend.loaded.clear()

    mgr = _manager(backend, restart_backend=restart)
    mgr.ensure_loaded("qwen2_5-coder-7b-fp16")
    assert restarts == [1]  # restarted exactly once
    assert backend.loaded == {"qwen2_5-coder-7b-fp16"}  # target loaded after restart
    # no in-place unload of the previous model
    assert not any("unload" in c for c in backend.calls)
