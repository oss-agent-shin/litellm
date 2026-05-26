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


def _get_router_zero_cost_cache(llm_router: Router) -> Optional[Dict[str, bool]]:
    """
    Return the router's per-instance zero-cost cache, or ``None`` for objects
    that don't expose one (e.g. ``MagicMock`` stand-ins in unit tests).

    The cache lives on the ``Router`` instance so it:
        * is invalidated by ``Router._invalidate_model_group_info_cache`` on
          any model add/remove/upsert (including in-place pricing changes via
          ``/model/update``, which go through ``upsert_deployment``);
        * dies with the router itself — no risk of CPython reusing the
          previous router's ``id()`` and serving its cached entries.
    """
    cache = getattr(llm_router, "_zero_cost_cache", None)
    return cache if isinstance(cache, dict) else None


def _is_model_cost_zero(
    model: Optional[Union[str, List[str]]], llm_router: Optional[Router]
) -> bool:
    """
    Check if a model has zero cost (no configured pricing).

    Uses the router's get_model_group_info method to get pricing information.

    Args:
        model: The model name or list of model names
        llm_router: The LiteLLM router instance

    Returns:
        bool: True if all costs for the model are zero, False otherwise
    """
    if model is None or llm_router is None:
        return False

    # Handle list of models
    model_list = [model] if isinstance(model, str) else model

    zero_cost_cache = _get_router_zero_cost_cache(llm_router)

    for model_name in model_list:
        if zero_cost_cache is not None:
            cached = zero_cost_cache.get(model_name)
            if cached is not None:
                if cached is False:
                    return False
                continue
        try:
            # Use router's get_model_group_info method directly for better reliability
            model_group_info = llm_router.get_model_group_info(model_group=model_name)

            if model_group_info is None:
                # Model not found or no pricing info available
                # Conservative approach: assume it has cost
                verbose_proxy_logger.debug(
                    f"No model group info found for {model_name}, assuming it has cost"
                )
                if zero_cost_cache is not None:
                    zero_cost_cache[model_name] = False
                return False

            # Check costs for this model
            # Only allow bypass if BOTH costs are explicitly set to 0 (not None)
            input_cost = model_group_info.input_cost_per_token
            output_cost = model_group_info.output_cost_per_token

            # If costs are not explicitly configured (None), assume it has cost
            if input_cost is None or output_cost is None:
                verbose_proxy_logger.debug(
                    f"Model {model_name} has undefined cost (input: {input_cost}, output: {output_cost}), assuming it has cost"
                )
                if zero_cost_cache is not None:
                    zero_cost_cache[model_name] = False
                return False

            # If either cost is non-zero, return False
            if input_cost > 0 or output_cost > 0:
                verbose_proxy_logger.debug(
                    f"Model {model_name} has non-zero cost (input: {input_cost}, output: {output_cost})"
                )
                if zero_cost_cache is not None:
                    zero_cost_cache[model_name] = False
                return False

            # Costs are 0 — verify this is from explicit configuration,
            # not from defaulted sparse auto-registration entries.
            # See: https://github.com/BerriAI/litellm/issues/24770
            safe_name = str(model_name).replace("\n", "").replace("\r", "")
            if not _is_cost_explicitly_configured(model_name, llm_router):
                verbose_proxy_logger.debug(
                    "Model %s has zero cost but no explicit cost "
                    "configuration in model_cost entry — treating as unknown "
                    "cost (enforce budget)",
                    safe_name,
                )
                if zero_cost_cache is not None:
                    zero_cost_cache[model_name] = False
                return False

            verbose_proxy_logger.debug(
                "Model %s has zero cost explicitly configured (input: %s, output: %s)",
                safe_name,
                input_cost,
                output_cost,
            )
            if zero_cost_cache is not None:
                zero_cost_cache[model_name] = True

        except Exception as e:
            # If we can't determine the cost, assume it has cost (conservative approach)
            verbose_proxy_logger.debug(
                f"Error checking cost for model {model_name}: {str(e)}, assuming it has cost"
            )
            return False

    # All models checked have zero cost
    return True


def _is_cost_explicitly_configured(model: str, llm_router: "Router") -> bool:
    """
    Check if any deployment in the model group has cost fields explicitly
    set in its litellm.model_cost entry.

    When Router._create_deployment() registers a model not in the global
    cost map, it creates a sparse entry like {"id": "<hash>"} with no cost
    fields. _get_model_info_helper() then defaults missing costs to 0.
    This function detects that scenario by checking the raw model_cost entry.
    """
    for deployment in llm_router.model_list:
        if deployment.get("model_name") != model:
            continue
        model_id = deployment.get("model_info", {}).get("id")
        if model_id is None:
            continue
        raw_entry = litellm.model_cost.get(model_id, {})
        if "input_cost_per_token" in raw_entry or "output_cost_per_token" in raw_entry:
            return True
    return False


async def _run_project_checks(
    project_object: Optional[LiteLLM_ProjectTableCachedObj],
    _model: Optional[Union[str, List[str]]],
    llm_router: Optional[Router],
    skip_budget_checks: bool,
    valid_token: Optional[UserAPIKeyAuth],
    proxy_logging_obj: ProxyLogging,
) -> None:
    """
    Run all project-level checks: blocked, model access, budget, soft budget.
    Extracted from common_checks() to keep statement count manageable.
    """
    if project_object is None:
        return

    # 1.1. If project is blocked
    if project_object.blocked is True:
        raise Exception(
            f"Project={project_object.project_id} is blocked. Update via `/project/update` if you're an admin."
        )

    # 2.2 If project can call model
    if _model and len(project_object.models) > 0:
        can_project_access_model(
            model=_model,
            project_object=project_object,
            llm_router=llm_router,
        )

    if not skip_budget_checks:
        # 3.0.2. If project is in budget
        await _project_max_budget_check(
            project_object=project_object,
            valid_token=valid_token,
            proxy_logging_obj=proxy_logging_obj,
        )

        # 3.0.3. If project is over soft budget (alert only, doesn't block)
        await _project_soft_budget_check(
            project_object=project_object,
            valid_token=valid_token,
            proxy_logging_obj=proxy_logging_obj,
        )


def _enforce_user_param_check(
    general_settings: dict, request: Request, request_body: dict, route: str
) -> None:
    if not general_settings.get("enforce_user_param", False):
        return

    http_method = request.method if hasattr(request, "method") else None
    is_post_method = http_method and http_method.upper() == "POST"
    is_openai_route = RouteChecks.is_llm_api_route(route=route)
    is_mcp_route = (
        route in LiteLLMRoutes.mcp_routes.value
        or RouteChecks.check_route_access(
            route=route, allowed_routes=LiteLLMRoutes.mcp_routes.value
        )
    )

    if (
        is_post_method
        and is_openai_route
        and not is_mcp_route
        and "user" not in request_body
    ):
        raise Exception(
            f"'user' param not passed in. 'enforce_user_param'={general_settings['enforce_user_param']}"
        )
