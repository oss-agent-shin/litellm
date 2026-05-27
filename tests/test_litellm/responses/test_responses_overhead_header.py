"""Regression tests for LIT-1756.

`/v1/responses` did not emit the `x-litellm-overhead-duration-ms` response
header because `litellm_overhead_time_ms` never landed in `_hidden_params`:

- `BaseLLMHTTPHandler.{response_api_handler, async_response_api_handler}` did
  not pass `logging_obj=` to the underlying `(a)sync_httpx_client.post(...)`,
  so `@track_llm_api_timing()` could not record duration via the logging object.
- `HTTPHandler.post` (sync) was undecorated, so even when callers did pass
  `logging_obj`, the sync path never set `llm_api_duration_ms`.

These tests assert that after the fix,
`_hidden_params["litellm_overhead_time_ms"]` is populated for both sync and
async responses calls -- the same field that
`ProxyBaseLLMRequestProcessing.get_custom_headers` reads to emit the header.
"""
import asyncio
import json
from unittest.mock import patch

import httpx

import litellm


_RESPONSES_PAYLOAD = {
    "id": "resp_test",
    "object": "response",
    "status": "completed",
    "model": "gpt-4o",
    "created_at": 1700000000,
    "output": [
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": "Paris.", "annotations": []}
            ],
        }
    ],
    "usage": {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
}


def _make_response():
    return httpx.Response(
        200,
        content=json.dumps(_RESPONSES_PAYLOAD).encode(),
        headers={"content-type": "application/json"},
        request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
    )


def test_responses_populates_litellm_overhead_time_ms_sync(monkeypatch):
    """Sync ``litellm.responses(...)`` populates the hidden-param that backs
    ``x-litellm-overhead-duration-ms``."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def _send(self, request, **kwargs):
        return _make_response()

    with patch.object(httpx.Client, "send", _send):
        response = litellm.responses(
            model="gpt-4o",
            input="What is the capital of France?",
            max_output_tokens=8,
        )

    hidden_params = getattr(response, "_hidden_params", {}) or {}
    assert "litellm_overhead_time_ms" in hidden_params, (
        "_hidden_params is missing 'litellm_overhead_time_ms'; the proxy will "
        "emit `x-litellm-overhead-duration-ms: None`. Keys: "
        f"{sorted(hidden_params.keys())}"
    )
    assert hidden_params["litellm_overhead_time_ms"] is not None
    assert isinstance(hidden_params["litellm_overhead_time_ms"], (int, float))


def test_responses_populates_litellm_overhead_time_ms_async(monkeypatch):
    """Async ``litellm.aresponses(...)`` populates the hidden-param that backs
    ``x-litellm-overhead-duration-ms``."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    async def _send(self, request, **kwargs):
        return _make_response()

    async def _run():
        with patch.object(httpx.AsyncClient, "send", _send):
            return await litellm.aresponses(
                model="gpt-4o",
                input="What is the capital of France?",
                max_output_tokens=8,
            )

    response = asyncio.run(_run())
    hidden_params = getattr(response, "_hidden_params", {}) or {}
    assert "litellm_overhead_time_ms" in hidden_params, (
        "_hidden_params is missing 'litellm_overhead_time_ms'; the proxy will "
        "emit `x-litellm-overhead-duration-ms: None`. Keys: "
        f"{sorted(hidden_params.keys())}"
    )
    assert hidden_params["litellm_overhead_time_ms"] is not None
    assert isinstance(hidden_params["litellm_overhead_time_ms"], (int, float))
