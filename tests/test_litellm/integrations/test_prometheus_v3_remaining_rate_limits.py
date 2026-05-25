"""Tests for the v3-rate-limiter path of
PrometheusLogger._set_virtual_key_rate_limit_metrics (LIT-2577).

The v3 parallel-request rate limiter writes per-model-per-key remaining
values into response._hidden_params["additional_headers"] under keys like
x-ratelimit-model_per_key-remaining-{tokens,requests}. These flow into
standard_logging_object.hidden_params.additional_headers.

Before the fix, _set_virtual_key_rate_limit_metrics only consulted the v1
legacy keys (litellm-key-remaining-{requests,tokens}-{model_group}) in
metadata. When the v3 limiter was in use, the lookup missed and the gauge
was set to sys.maxsize (~9e18), so DataDog reported a constant ~9e18 for
the remaining-tokens / remaining-requests gauges.
"""
import sys
import pytest
from prometheus_client import REGISTRY
import litellm
from litellm.integrations.prometheus import PrometheusLogger


# sys.maxsize is an int that can't be represented exactly as a double; the
# prometheus_client stores gauge values as floats, so the readback rounds to
# ~9.223372036854776e+18. Use the rounded float for assertions and a large
# threshold for "is this an unbounded/maxsize-style value?" checks.
MAXSIZE_FLOAT = float(sys.maxsize)


def _is_maxsize_like(value: float) -> bool:
    return value >= 1e18


def _clear_registry():
    for c in list(REGISTRY._collector_to_names.keys()):
        REGISTRY.unregister(c)


def _logger(monkeypatch):
    monkeypatch.setattr(litellm, "custom_prometheus_metadata_labels", [])
    _clear_registry()
    return PrometheusLogger()


def _samples(name):
    return [s for m in REGISTRY.collect() for s in m.samples if s.name == name]


def _slp(headers):
    return {
        "model_id": "model-123",
        "model_group": "gpt-4o-mini",
        "api_base": "https://api.openai.com",
        "custom_llm_provider": "openai",
        "metadata": {
            "user_api_key_hash": "test-hash",
            "user_api_key_alias": "test-alias",
            "user_api_key_team_id": None,
            "user_api_key_team_alias": None,
            "user_api_key_user_id": None,
            "user_api_key_user_email": None,
            "user_api_key_org_id": None,
            "requester_metadata": {},
            "user_api_key_auth_metadata": None,
            "spend_logs_metadata": None,
        },
        "hidden_params": {"additional_headers": headers},
        "request_tags": [],
        "completion_tokens": 0,
        "total_tokens": 0,
        "response_cost": 0,
    }


def _call(logger, *, metadata, headers=None, response_hidden=None):
    kw = {
        "litellm_params": {"metadata": metadata},
        "standard_logging_object": _slp(headers or {}),
    }
    if response_hidden is not None:
        class _R: pass
        r = _R()
        r._hidden_params = response_hidden
        kw["response_obj"] = r
    logger._set_virtual_key_rate_limit_metrics(
        user_api_key="test-hash",
        user_api_key_alias="test-alias",
        kwargs=kw,
        metadata=metadata,
        model_id="model-123",
    )


def test_v3_additional_headers_populate_remaining_rate_limit_gauges(monkeypatch):
    """Headers written by the v3 limiter must drive the gauges (this is the
    main regression LIT-2577 was about: before the fix, both values fell
    through to sys.maxsize when the v3 limiter was in use)."""
    logger = _logger(monkeypatch)
    _call(
        logger,
        metadata={"model_group": "gpt-4o-mini"},
        headers={
            "x-ratelimit-model_per_key-remaining-requests": 7,
            "x-ratelimit-model_per_key-remaining-tokens": 1234,
        },
    )
    req = _samples("litellm_remaining_api_key_requests_for_model")
    tok = _samples("litellm_remaining_api_key_tokens_for_model")
    assert any(s.value == 7 for s in req)
    assert any(s.value == 1234 for s in tok)
    assert not any(_is_maxsize_like(s.value) for s in req)
    assert not any(_is_maxsize_like(s.value) for s in tok)


def test_v3_additional_headers_preserve_zero_remaining(monkeypatch):
    """A legitimate 0 (key/model is exactly at its quota) must not be
    swallowed by a falsy check and replaced with sys.maxsize."""
    logger = _logger(monkeypatch)
    _call(
        logger,
        metadata={"model_group": "gpt-4o-mini"},
        headers={
            "x-ratelimit-model_per_key-remaining-requests": 0,
            "x-ratelimit-model_per_key-remaining-tokens": 0,
        },
    )
    req = _samples("litellm_remaining_api_key_requests_for_model")
    tok = _samples("litellm_remaining_api_key_tokens_for_model")
    assert any(s.value == 0 for s in req)
    assert any(s.value == 0 for s in tok)
    assert not any(_is_maxsize_like(s.value) for s in req)
    assert not any(_is_maxsize_like(s.value) for s in tok)


def test_v3_additional_headers_take_precedence_over_legacy_metadata(monkeypatch):
    """When both v3 additional_headers and v1 legacy metadata keys are
    present, the v3 source wins."""
    logger = _logger(monkeypatch)
    md = {
        "model_group": "gpt-4o-mini",
        "litellm-key-remaining-requests-gpt-4o-mini": 999,
        "litellm-key-remaining-tokens-gpt-4o-mini": 8888,
    }
    _call(
        logger,
        metadata=md,
        headers={
            "x-ratelimit-model_per_key-remaining-requests": 3,
            "x-ratelimit-model_per_key-remaining-tokens": 4,
        },
    )
    req = _samples("litellm_remaining_api_key_requests_for_model")
    tok = _samples("litellm_remaining_api_key_tokens_for_model")
    assert any(s.value == 3 for s in req)
    assert any(s.value == 4 for s in tok)
    assert not any(s.value == 999 for s in req)
    assert not any(s.value == 8888 for s in tok)


def test_legacy_metadata_path_still_works_without_v3_headers(monkeypatch):
    """Backward compatibility: when only the legacy v1 metadata keys are
    populated (v3 limiter disabled), gauges still set from metadata."""
    logger = _logger(monkeypatch)
    md = {
        "model_group": "gpt-4o-mini",
        "litellm-key-remaining-requests-gpt-4o-mini": 11,
        "litellm-key-remaining-tokens-gpt-4o-mini": 22,
    }
    _call(logger, metadata=md, headers={})
    req = _samples("litellm_remaining_api_key_requests_for_model")
    tok = _samples("litellm_remaining_api_key_tokens_for_model")
    assert any(s.value == 11 for s in req)
    assert any(s.value == 22 for s in tok)
    assert not any(_is_maxsize_like(s.value) for s in req)
    assert not any(_is_maxsize_like(s.value) for s in tok)


def test_no_sources_falls_back_to_sys_maxsize(monkeypatch):
    """When neither v3 headers nor v1 metadata expose a remaining value, the
    gauge falls back to sys.maxsize (previous behavior; this is the only
    case where ~9e18 is correct)."""
    logger = _logger(monkeypatch)
    _call(logger, metadata={"model_group": "gpt-4o-mini"}, headers={})
    req = _samples("litellm_remaining_api_key_requests_for_model")
    tok = _samples("litellm_remaining_api_key_tokens_for_model")
    assert any(s.value == MAXSIZE_FLOAT for s in req)
    assert any(s.value == MAXSIZE_FLOAT for s in tok)


def test_v3_headers_read_from_response_obj_fallback(monkeypatch):
    """If the v3 headers are present on the response object's _hidden_params
    but not on the standard logging payload (defensive fallback path), they
    should still be picked up."""
    logger = _logger(monkeypatch)
    _call(
        logger,
        metadata={"model_group": "gpt-4o-mini"},
        headers={},
        response_hidden={
            "additional_headers": {
                "x-ratelimit-model_per_key-remaining-requests": 42,
                "x-ratelimit-model_per_key-remaining-tokens": 100,
            }
        },
    )
    req = _samples("litellm_remaining_api_key_requests_for_model")
    tok = _samples("litellm_remaining_api_key_tokens_for_model")
    assert any(s.value == 42 for s in req)
    assert any(s.value == 100 for s in tok)


def test_v3_partial_headers_fall_back_to_legacy_for_other_dimension(monkeypatch):
    """Each dimension (requests/tokens) resolves independently: if only the
    v3 tokens header is present, requests can still come from legacy
    metadata. Either side missing must not poison the other."""
    logger = _logger(monkeypatch)
    md = {
        "model_group": "gpt-4o-mini",
        "litellm-key-remaining-requests-gpt-4o-mini": 50,
    }
    _call(
        logger,
        metadata=md,
        headers={"x-ratelimit-model_per_key-remaining-tokens": 9000},
    )
    req = _samples("litellm_remaining_api_key_requests_for_model")
    tok = _samples("litellm_remaining_api_key_tokens_for_model")
    assert any(s.value == 50 for s in req)
    assert any(s.value == 9000 for s in tok)
    assert not any(_is_maxsize_like(s.value) for s in req)
    assert not any(_is_maxsize_like(s.value) for s in tok)
