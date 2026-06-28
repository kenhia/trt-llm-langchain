"""Offline error-path coverage: unreachable backend surfaces ServerUnavailableError."""

from __future__ import annotations

import httpx
import pytest

from trt_llm_langchain import ServerUnavailableError, TrtLlmManager, TrtLlmSettings


def _refusing_manager() -> TrtLlmManager:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    client = httpx.Client(base_url="http://down", transport=httpx.MockTransport(handler))
    return TrtLlmManager(TrtLlmSettings(control_url="http://down:8000"), client=client)


def test_is_healthy_false_when_down() -> None:
    assert _refusing_manager().is_healthy() is False


def test_models_raises_server_unavailable() -> None:
    mgr = _refusing_manager()
    with pytest.raises(ServerUnavailableError) as ei:
        mgr.models()
    assert "http://down:8000" in str(ei.value)


def test_is_responsive_false_when_down() -> None:
    assert _refusing_manager().is_responsive("anything") is False


def test_wait_healthy_times_out_when_down() -> None:
    mgr = _refusing_manager()
    with pytest.raises(ServerUnavailableError) as ei:
        mgr._wait_healthy(timeout_s=0)  # immediate budget exhaustion
    assert "restart" in str(ei.value).lower() or "healthy" in str(ei.value).lower()
