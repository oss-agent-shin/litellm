import logging
import sys

import pytest
from prometheus_client import REGISTRY

import litellm
from litellm.integrations.prometheus import PrometheusLogger


def _clear_prometheus_registry() -> None:
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        REGISTRY.unregister(collector)


def _create_prometheus_logger_with_custom_labels(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        litellm,
        "custom_prometheus_metadata_labels",
        ["metadata.department", "metadata.environment"],
    )
    _clear_prometheus_registry()
    return PrometheusLogger()


def _standard_logging_payload_with_requester_metadata() -> dict:
    return {
        "model_id": "model-123",
        "model_group": "gpt-4o-mini",
        "api_base": "https://api.openai.com",
        "custom_llm_provider": "openai",
        "metadata": {
            "user_api_key_hash": "test-hash",
            "user_api_key_alias": "test-alias",
            "user_api_key_team_id": "test-team",
            "user_api_key_team_alias": "test-team-alias",
            "user_api_key_user_id": "test-user",
            "user_api_key_user_email": "test@example.com",
            "user_api_key_org_id": None,
            "requester_metadata": {
                "department": "engineering",
                "environment": "production",
            },
            "user_api_key_auth_metadata": None,
            "spend_logs_metadata": None,
        },
        "request_tags": [],
        "completion_tokens": 0,
        "total_tokens": 0,
        "response_cost": 0,
    }


def _metric_samples(metric_name: str):
    return [
        sample
        for metric in REGISTRY.collect()
        for sample in metric.samples
        if sample.name == metric_name
    ]


@pytest.mark.asyncio
async def test_async_log_failure_event_accepts_custom_metadata_labels(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    prometheus_logger = _create_prometheus_logger_with_custom_labels(monkeypatch)
    kwargs = {
        "model": "gpt-4o-mini",
        "litellm_params": {
            "metadata": {
                "user_api_key_end_user_id": "test-end-user",
            }
        },
        "standard_logging_object": _standard_logging_payload_with_requester_metadata(),
    }

    with caplog.at_level(logging.ERROR):
        await prometheus_logger.async_log_failure_event(
            kwargs=kwargs,
            response_obj=None,
            start_time=None,
            end_time=None,
        )

    assert "Incorrect label count" not in caplog.text
    samples = _metric_samples("litellm_llm_api_failed_requests_metric_total")
    assert any(
        sample.labels.get("metadata_department") == "engineering"
        and sample.labels.get("metadata_environment") == "production"
        for sample in samples
    )


def test_virtual_key_rate_limit_metrics_accept_custom_metadata_labels(
    monkeypatch: pytest.MonkeyPatch,
):
    prometheus_logger = _create_prometheus_logger_with_custom_labels(monkeypatch)
    metadata = {
        "model_group": "gpt-4o-mini",
        "litellm-key-remaining-requests-gpt-4o-mini": 3,
        "litellm-key-remaining-tokens-gpt-4o-mini": 200,
    }
    kwargs = {
        "litellm_params": {
            "metadata": metadata,
        },
        "standard_logging_object": _standard_logging_payload_with_requester_metadata(),
    }

    prometheus_logger._set_virtual_key_rate_limit_metrics(
        user_api_key="test-hash",
        user_api_key_alias="test-alias",
        kwargs=kwargs,
        metadata=metadata,
        model_id="model-123",
    )

    samples = _metric_samples("litellm_remaining_api_key_requests_for_model")
    assert any(
        sample.labels.get("metadata_department") == "engineering"
        and sample.labels.get("metadata_environment") == "production"
        and sample.value == 3
        for sample in samples
    )


def test_virtual_key_rate_limit_metrics_preserve_zero_remaining_values(
    monkeypatch: pytest.MonkeyPatch,
):
    prometheus_logger = _create_prometheus_logger_with_custom_labels(monkeypatch)
    metadata = {
        "model_group": "gpt-4o-mini",
        "litellm-key-remaining-requests-gpt-4o-mini": 0,
        "litellm-key-remaining-tokens-gpt-4o-mini": 0,
    }
    kwargs = {
        "litellm_params": {
            "metadata": metadata,
        },
        "standard_logging_object": _standard_logging_payload_with_requester_metadata(),
    }

    prometheus_logger._set_virtual_key_rate_limit_metrics(
        user_api_key="test-hash",
        user_api_key_alias="test-alias",
        kwargs=kwargs,
        metadata=metadata,
        model_id="model-123",
    )

    request_samples = _metric_samples("litellm_remaining_api_key_requests_for_model")
    token_samples = _metric_samples("litellm_remaining_api_key_tokens_for_model")

    assert any(sample.value == 0 for sample in request_samples)
    assert any(sample.value == 0 for sample in token_samples)
    assert not any(sample.value == sys.maxsize for sample in request_samples)
    assert not any(sample.value == sys.maxsize for sample in token_samples)



def test_virtual_key_rate_limit_metrics_fallback_to_additional_headers(
    monkeypatch: pytest.MonkeyPatch,
):
    """LIT-2577: when the legacy ``litellm-key-remaining-{requests,tokens}-{model_group}``
    keys are missing from ``metadata`` (the v3 ``parallel_request_limiter_v3``
    path), the gauges must read from
    ``standard_logging_payload.hidden_params.additional_headers`` instead of
    falling back to ``sys.maxsize`` (~9.22e18).
    """
    prometheus_logger = _create_prometheus_logger_with_custom_labels(monkeypatch)
    # Legacy metadata location: empty. Only model_group present.
    metadata = {"model_group": "gpt-4o-mini"}

    standard_logging_payload = _standard_logging_payload_with_requester_metadata()
    standard_logging_payload["hidden_params"] = {
        "additional_headers": {
            "x-ratelimit-model_per_key-remaining-requests": 7,
            "x-ratelimit-model_per_key-remaining-tokens": 1234,
            "x-ratelimit-model_per_key-limit-requests": 10,
            "x-ratelimit-model_per_key-limit-tokens": 2000,
        }
    }

    kwargs = {
        "litellm_params": {"metadata": metadata},
        "standard_logging_object": standard_logging_payload,
    }

    prometheus_logger._set_virtual_key_rate_limit_metrics(
        user_api_key="test-hash",
        user_api_key_alias="test-alias",
        kwargs=kwargs,
        metadata=metadata,
        model_id="model-123",
    )

    request_samples = _metric_samples("litellm_remaining_api_key_requests_for_model")
    token_samples = _metric_samples("litellm_remaining_api_key_tokens_for_model")

    assert any(sample.value == 7 for sample in request_samples), (
        "remaining-requests should come from additional_headers fallback, "
        f"got samples={[s.value for s in request_samples]}"
    )
    assert any(sample.value == 1234 for sample in token_samples), (
        "remaining-tokens should come from additional_headers fallback, "
        f"got samples={[s.value for s in token_samples]}"
    )
    # And critically, no sys.maxsize leakage.
    assert not any(sample.value == sys.maxsize for sample in request_samples)
    assert not any(sample.value == sys.maxsize for sample in token_samples)


def test_virtual_key_rate_limit_metrics_metadata_takes_precedence_over_headers(
    monkeypatch: pytest.MonkeyPatch,
):
    """When both legacy metadata keys and additional_headers are populated,
    metadata wins. Guards against regressing the v1 limiter behaviour.
    """
    prometheus_logger = _create_prometheus_logger_with_custom_labels(monkeypatch)
    metadata = {
        "model_group": "gpt-4o-mini",
        "litellm-key-remaining-requests-gpt-4o-mini": 3,
        "litellm-key-remaining-tokens-gpt-4o-mini": 200,
    }

    standard_logging_payload = _standard_logging_payload_with_requester_metadata()
    standard_logging_payload["hidden_params"] = {
        "additional_headers": {
            "x-ratelimit-model_per_key-remaining-requests": 99,
            "x-ratelimit-model_per_key-remaining-tokens": 9999,
        }
    }

    kwargs = {
        "litellm_params": {"metadata": metadata},
        "standard_logging_object": standard_logging_payload,
    }

    prometheus_logger._set_virtual_key_rate_limit_metrics(
        user_api_key="test-hash",
        user_api_key_alias="test-alias",
        kwargs=kwargs,
        metadata=metadata,
        model_id="model-123",
    )

    request_samples = _metric_samples("litellm_remaining_api_key_requests_for_model")
    token_samples = _metric_samples("litellm_remaining_api_key_tokens_for_model")

    assert any(sample.value == 3 for sample in request_samples)
    assert any(sample.value == 200 for sample in token_samples)
    assert not any(sample.value == 99 for sample in request_samples)
    assert not any(sample.value == 9999 for sample in token_samples)


def test_virtual_key_rate_limit_metrics_partial_metadata_fills_from_headers(
    monkeypatch: pytest.MonkeyPatch,
):
    """If only one of remaining-requests/tokens is in metadata, the other
    should still be picked up from additional_headers."""
    prometheus_logger = _create_prometheus_logger_with_custom_labels(monkeypatch)
    metadata = {
        "model_group": "gpt-4o-mini",
        "litellm-key-remaining-requests-gpt-4o-mini": 5,
        # tokens deliberately absent
    }

    standard_logging_payload = _standard_logging_payload_with_requester_metadata()
    standard_logging_payload["hidden_params"] = {
        "additional_headers": {
            "x-ratelimit-model_per_key-remaining-tokens": 42,
        }
    }

    kwargs = {
        "litellm_params": {"metadata": metadata},
        "standard_logging_object": standard_logging_payload,
    }

    prometheus_logger._set_virtual_key_rate_limit_metrics(
        user_api_key="test-hash",
        user_api_key_alias="test-alias",
        kwargs=kwargs,
        metadata=metadata,
        model_id="model-123",
    )

    request_samples = _metric_samples("litellm_remaining_api_key_requests_for_model")
    token_samples = _metric_samples("litellm_remaining_api_key_tokens_for_model")

    assert any(sample.value == 5 for sample in request_samples)
    assert any(sample.value == 42 for sample in token_samples)
    assert not any(sample.value == sys.maxsize for sample in token_samples)


def test_virtual_key_rate_limit_metrics_no_data_anywhere_falls_back_to_maxsize(
    monkeypatch: pytest.MonkeyPatch,
):
    """If neither metadata nor additional_headers have values, the original
    ``sys.maxsize`` sentinel is preserved (we only want to remove the false
    sentinel when real data is available)."""
    prometheus_logger = _create_prometheus_logger_with_custom_labels(monkeypatch)
    metadata = {"model_group": "gpt-4o-mini"}

    standard_logging_payload = _standard_logging_payload_with_requester_metadata()
    standard_logging_payload["hidden_params"] = {"additional_headers": {}}

    kwargs = {
        "litellm_params": {"metadata": metadata},
        "standard_logging_object": standard_logging_payload,
    }

    prometheus_logger._set_virtual_key_rate_limit_metrics(
        user_api_key="test-hash",
        user_api_key_alias="test-alias",
        kwargs=kwargs,
        metadata=metadata,
        model_id="model-123",
    )

    request_samples = _metric_samples("litellm_remaining_api_key_requests_for_model")
    token_samples = _metric_samples("litellm_remaining_api_key_tokens_for_model")

    assert any(sample.value == float(sys.maxsize) for sample in request_samples)
    assert any(sample.value == float(sys.maxsize) for sample in token_samples)


def test_virtual_key_rate_limit_metrics_zero_remaining_in_headers_preserved(
    monkeypatch: pytest.MonkeyPatch,
):
    """Zero is a meaningful value (key/model is exhausted) and must not be
    treated as ``None`` by ``or``-coalescing logic."""
    prometheus_logger = _create_prometheus_logger_with_custom_labels(monkeypatch)
    metadata = {"model_group": "gpt-4o-mini"}

    standard_logging_payload = _standard_logging_payload_with_requester_metadata()
    standard_logging_payload["hidden_params"] = {
        "additional_headers": {
            "x-ratelimit-model_per_key-remaining-requests": 0,
            "x-ratelimit-model_per_key-remaining-tokens": 0,
        }
    }

    kwargs = {
        "litellm_params": {"metadata": metadata},
        "standard_logging_object": standard_logging_payload,
    }

    prometheus_logger._set_virtual_key_rate_limit_metrics(
        user_api_key="test-hash",
        user_api_key_alias="test-alias",
        kwargs=kwargs,
        metadata=metadata,
        model_id="model-123",
    )

    request_samples = _metric_samples("litellm_remaining_api_key_requests_for_model")
    token_samples = _metric_samples("litellm_remaining_api_key_tokens_for_model")

    assert any(sample.value == 0 for sample in request_samples)
    assert any(sample.value == 0 for sample in token_samples)
    assert not any(sample.value == sys.maxsize for sample in request_samples)
    assert not any(sample.value == sys.maxsize for sample in token_samples)
