# used for /metrics endpoint on LiteLLM Proxy
#### What this does ####
#    On success, log events to Prometheus
from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timedelta
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import litellm
from litellm._logging import print_verbose, verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.prometheus_helpers.bounded_prometheus_series_tracker import (
    BoundedPrometheusSeriesTracker,
)
from litellm.integrations.prometheus_helpers import (
    PrometheusLabelFactoryContext,
    _get_cached_end_user_id_for_cost_tracking,
)
from litellm.litellm_core_utils.core_helpers import (
    get_litellm_metadata_from_kwargs,
    get_metadata_variable_name_from_kwargs,
)
from litellm.proxy._types import (
    LiteLLM_DeletedVerificationToken,
    LiteLLM_TeamTable,
    LiteLLM_UserTable,
    UserAPIKeyAuth,
)
from litellm.types.integrations.prometheus import *
from litellm.types.integrations.prometheus import (
    _sanitize_prometheus_label_name,
    _sanitize_prometheus_label_value,
)
from litellm.types.utils import StandardLoggingPayload

if TYPE_CHECKING:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
else:
    AsyncIOScheduler = Any


class PrometheusLogger(CustomLogger):
    # Class variables or attributes

    @staticmethod
    def get_instance() -> Optional["PrometheusLogger"]:
        """Find the PrometheusLogger instance from litellm.callbacks, if registered."""
        import litellm

        for cb in litellm.callbacks:
            if isinstance(cb, PrometheusLogger):
                return cb
        return None

    def __init__(  # noqa: PLR0915
        self,
        **kwargs,
    ):
        try:
            from prometheus_client import Counter, Gauge, Histogram

            # Always initialize label_filters, even for non-premium users
            self.label_filters = self._parse_prometheus_config()

            _custom_buckets = litellm.prometheus_latency_buckets
            self.latency_buckets = (
                tuple(_custom_buckets)
                if _custom_buckets is not None
                else LATENCY_BUCKETS
            )
            self._bounded_prometheus_series_tracker = BoundedPrometheusSeriesTracker()

            # Create metric factory functions
            self._counter_factory = self._create_metric_factory(Counter)
            self._gauge_factory = self._create_metric_factory(Gauge)
            self._histogram_factory = self._create_metric_factory(Histogram)

            self.litellm_proxy_failed_requests_metric = self._counter_factory(
                name="litellm_proxy_failed_requests_metric",
                documentation="Total number of failed responses from proxy - the client did not get a success response from litellm proxy",
                labelnames=self.get_labels_for_metric(
                    "litellm_proxy_failed_requests_metric"
                ),
            )

            self.litellm_proxy_total_requests_metric = self._counter_factory(
                name="litellm_proxy_total_requests_metric",
                documentation="Total number of requests made to the proxy server - track number of client side requests",
                labelnames=self.get_labels_for_metric(
                    "litellm_proxy_total_requests_metric"
                ),
            )

            # request latency metrics
            self.litellm_request_total_latency_metric = self._histogram_factory(
                "litellm_request_total_latency_metric",
                "Total latency (seconds) for a request to LiteLLM",
                labelnames=self.get_labels_for_metric(
                    "litellm_request_total_latency_metric"
                ),
                buckets=self.latency_buckets,
            )

            self.litellm_llm_api_latency_metric = self._histogram_factory(
                "litellm_llm_api_latency_metric",
                "Total latency (seconds) for a models LLM API call",
                labelnames=self.get_labels_for_metric("litellm_llm_api_latency_metric"),
                buckets=self.latency_buckets,
            )

            self.litellm_llm_api_time_to_first_token_metric = self._histogram_factory(
                "litellm_llm_api_time_to_first_token_metric",
                "Time to first token for a models LLM API call",
                # labelnames=[
                #     "model",
                #     "hashed_api_key",
                #     "api_key_alias",
                #     "team",
                #     "team_alias",
                # ],
                labelnames=self.get_labels_for_metric(
                    "litellm_llm_api_time_to_first_token_metric"
                ),
                buckets=self.latency_buckets,
            )

            # Counter for spend
            self.litellm_spend_metric = self._counter_factory(
                "litellm_spend_metric",
                "Total spend on LLM requests",
                labelnames=self.get_labels_for_metric("litellm_spend_metric"),
            )

            # Counter for total_output_tokens
            self.litellm_tokens_metric = self._counter_factory(
                "litellm_total_tokens_metric",
                "Total number of input + output tokens from LLM requests",
                labelnames=self.get_labels_for_metric("litellm_total_tokens_metric"),
            )

PLACEHOLDER_TRUNCATED_FOR_DEMO