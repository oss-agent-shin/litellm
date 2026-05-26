# What is this?
## Common auth checks between jwt + key based auth
"""
Got Valid Token from Cache, DB
Run checks for:

1. If user can call model
2. If user is in budget
3. If end_user ('user' passed to /chat/completions, /embeddings endpoint) is in budget
"""

import asyncio
import math
import re
import time
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Type, Union, cast

from fastapi import HTTPException, Request, status
from pydantic import BaseModel

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.caching.dual_cache import LimitedSizeOrderedDict
from litellm.constants import (
    CLI_JWT_EXPIRATION_HOURS,
    CLI_JWT_TOKEN_NAME,
    DEFAULT_ACCESS_GROUP_CACHE_TTL,
    DEFAULT_IN_MEMORY_TTL,
    DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL,
    DEFAULT_MAX_RECURSE_DEPTH,
    EMAIL_BUDGET_ALERT_MAX_SPEND_ALERT_PERCENTAGE,
)
from litellm.litellm_core_utils.dd_tracing import tracer
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.litellm_core_utils.safe_json_loads import safe_json_loads
from litellm.proxy._types import (
    RBAC_ROLES,
    CallInfo,
    LiteLLM_AccessGroupTable,
    LiteLLM_BudgetTable,
    LiteLLM_EndUserTable,
    Litellm_EntityType,
    LiteLLM_JWTAuth,
    LiteLLM_ManagedVectorStoresTable,
    LiteLLM_ObjectPermissionTable,
    LiteLLM_OrganizationMembershipTable,
    LiteLLM_OrganizationTable,
    LiteLLM_ProjectTableCachedObj,
    LiteLLM_TagTable,
    LiteLLM_TeamMembership,
    LiteLLM_TeamTable,
    LiteLLM_TeamTableCachedObj,
    LiteLLM_UserTable,
    LiteLLMRoutes,
    LitellmUserRoles,
    NewTeamRequest,
    ProxyErrorTypes,
    ProxyException,
    RoleBasedPermissions,
    SpecialModelNames,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy.common_utils.http_parsing_utils import (
    _safe_get_request_headers,
    _safe_get_request_query_params,
)
from litellm.proxy.db.exception_handler import PrismaDBExceptionHandler
from litellm.proxy.guardrails.tool_name_extraction import (
    TOOL_CAPABLE_CALL_TYPES,
    extract_request_tool_names,
)
from litellm.proxy.common_utils.cache_pydantic_utils import CacheCodec
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.route_llm_request import route_request
from litellm.proxy.utils import PrismaClient, ProxyLogging, log_db_metrics
from litellm.router import Router
from litellm.utils import get_utc_datetime

from .auth_checks_organization import organization_role_based_access_check
from .auth_utils import get_model_from_request

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    Span = Union[_Span, Any]
else:
    Span = Any


last_db_access_time = LimitedSizeOrderedDict(max_size=100)
db_cache_expiry = DEFAULT_IN_MEMORY_TTL  # refresh every 5s

all_routes = LiteLLMRoutes.openai_routes.value + LiteLLMRoutes.management_routes.value


def _log_budget_lookup_failure(entity: str, error: Exception) -> None:
    """
    Log a warning when budget lookup fails; cache will not be populated.

    Skips logging for expected "user not found" cases (bare Exception from
    get_user_object when user_id_upsert=False). Adds a schema migration hint
    when the error appears schema-related.
    """
    # Skip logging for expected "user not found" - not caching is correct
    if str(error) == "" and type(error).__name__ == "Exception":
        return
    err_str = str(error).lower()
    hint = ""
    if any(
        x in err_str
        for x in ("column", "schema", "does not exist", "prisma", "migrate")
    ):
        hint = (
            " Run `prisma db push` or `prisma migrate deploy` to fix schema mismatches."
        )
    verbose_proxy_logger.error(
        f"Budget lookup failed for {entity}; cache will not be populated. "
        f"Each request will hit the database. Error: {error}.{hint}"
    )
