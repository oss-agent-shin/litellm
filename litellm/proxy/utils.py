import asyncio
import copy
import hashlib
import inspect
import json
import os
import smtplib
import sys
import threading
import time
import traceback
from datetime import date, datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Awaitable,
    ClassVar,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    cast,
    overload,
)

from litellm import _custom_logger_compatible_callbacks_literal
from litellm.constants import DEFAULT_MODEL_CREATED_AT_TIME, MAX_TEAM_LIST_LIMIT
from litellm.proxy._types import (
    DB_CONNECTION_ERROR_TYPES,
    CommonProxyErrors,
    ProxyErrorTypes,
    ProxyException,
    SpendLogsMetadata,
    SpendLogsPayload,
)
from litellm.proxy.spend_tracking.spend_log_error_logger import spend_log_error
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import CallTypes, CallTypesLiteral

try:
    from litellm_enterprise.enterprise_callbacks.send_emails.base_email import (
        BaseEmailLogger,
    )
    from litellm_enterprise.enterprise_callbacks.send_emails.resend_email import (
        ResendEmailLogger,
    )
    from litellm_enterprise.enterprise_callbacks.send_emails.sendgrid_email import (
        SendGridEmailLogger,
    )
    from litellm_enterprise.enterprise_callbacks.send_emails.smtp_email import (
        SMTPEmailLogger,
    )
except ImportError:
    BaseEmailLogger = None  # type: ignore
    SendGridEmailLogger = None  # type: ignore
    SMTPEmailLogger = None  # type: ignore
    ResendEmailLogger = None  # type: ignore

try:
    import backoff
except ImportError:
    raise ImportError(
        "backoff is not installed. Please install it via 'pip install backoff'"
    )

from fastapi import HTTPException, status

import litellm
import litellm.litellm_core_utils
import litellm.litellm_core_utils.litellm_logging
from litellm import (
    EmbeddingResponse,
    ImageResponse,
    ModelResponse,
    ModelResponseStream,
    Router,
)
from litellm._logging import _redact_string, verbose_proxy_logger
from litellm._service_logger import ServiceLogging, ServiceTypes
from litellm.caching.caching import DualCache, RedisCache
from litellm.caching.dual_cache import LimitedSizeOrderedDict
from litellm.exceptions import RejectedRequestError
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    ModifyResponseException,
)
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.SlackAlerting.slack_alerting import SlackAlerting
from litellm.integrations.SlackAlerting.utils import _add_langfuse_trace_id_to_alert
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.litellm_core_utils.safe_json_loads import safe_json_loads
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.proxy._types import (
    AlertType,
    CallInfo,
    LiteLLM_VerificationTokenView,
    Member,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.db.create_views import (
    create_missing_views,
    should_create_missing_views,
)
from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter
from litellm.proxy.db.exception_handler import (
    PrismaDBExceptionHandler,
    call_with_db_reconnect_retry,
)
from litellm.proxy.db.log_db_metrics import log_db_metrics
from litellm.proxy.db.prisma_client import (
    PrismaWrapper,
    parse_iam_endpoint_from_url,
)
from litellm.proxy.db.routing_prisma_wrapper import RoutingPrismaWrapper
from litellm.proxy.guardrails.guardrail_hooks.unified_guardrail.unified_guardrail import (
    UnifiedLLMGuardrails,
)
from litellm.proxy.hooks import PROXY_HOOKS, get_proxy_hook
from litellm.proxy.hooks.cache_control_check import _PROXY_CacheControlCheck
from litellm.proxy.hooks.max_budget_limiter import _PROXY_MaxBudgetLimiter
from litellm.proxy.hooks.parallel_request_limiter import (
    _PROXY_MaxParallelRequestsHandler,
)
from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup
from litellm.proxy.policy_engine.pipeline_executor import PipelineExecutor
from litellm.secret_managers.main import str_to_bool
from litellm.types.integrations.slack_alerting import DEFAULT_ALERT_TYPES
from litellm.types.mcp import (
    MCPDuringCallResponseObject,
    MCPPreCallRequestObject,
    MCPPreCallResponseObject,
)
from litellm.types.proxy.policy_engine.pipeline_types import PipelineExecutionResult
from litellm.types.utils import LLMResponseTypes, LoggedLiteLLMParams

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

    Span = Union[_Span, Any]
else:
    Span = Any


unified_guardrail = UnifiedLLMGuardrails()


# REMAINDER OMITTED FOR LENGTH - see issue report
