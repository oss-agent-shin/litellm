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


def _reject_clientside_metadata_tags_check(
    general_settings: dict, request_body: dict, route: str
) -> None:
    if not general_settings.get("reject_clientside_metadata_tags", False):
        return

    if (
        RouteChecks.is_llm_api_route(route=route)
        and "metadata" in request_body
        and isinstance(request_body["metadata"], dict)
        and "tags" in request_body["metadata"]
    ):
        raise ProxyException(
            message=f"Client-side 'metadata.tags' not allowed in request. 'reject_clientside_metadata_tags'={general_settings['reject_clientside_metadata_tags']}. Tags can only be set via API key metadata.",
            type=ProxyErrorTypes.bad_request_error,
            param="metadata.tags",
            code=status.HTTP_400_BAD_REQUEST,
        )


def _global_proxy_budget_check(
    global_proxy_spend: Optional[float], skip_budget_checks: bool, route: str
) -> None:
    if (
        litellm.max_budget > 0
        and not skip_budget_checks
        and global_proxy_spend is not None
        and RouteChecks.is_llm_api_route(route=route)
        and route != "/v1/models"
        and route != "/models"
    ):
        if (
            math.isfinite(litellm.max_budget)
            and global_proxy_spend > litellm.max_budget
        ):
            raise litellm.BudgetExceededError(
                current_cost=global_proxy_spend, max_budget=litellm.max_budget
            )


_GUARDRAIL_MODIFICATION_KEYS: tuple = (
    "guardrails",
    "disable_global_guardrails",
    "disable_global_guardrail",
    "opted_out_global_guardrails",
)


def _guardrail_modification_check(
    request_body: dict, team_object: Optional[LiteLLM_TeamTable]
) -> None:
    """
    Reject user-supplied metadata flags that would modify guardrail behavior
    unless the team has explicit permission. Checked keys include the plural
    ``guardrails`` list plus the per-request toggles that influence whether
    default-on guardrails run (``disable_global_guardrails``,
    ``disable_global_guardrail`` singular, and ``opted_out_global_guardrails``).

    User-supplied values for the bypass toggles are also silently ignored by
    ``_get_admin_metadata`` at read time; this check adds defense in depth by
    failing loudly at the auth layer so operators see an explicit 403 instead
    of a confusing silent-ignore.
    """
    from litellm.proxy.guardrails.guardrail_helpers import can_modify_guardrails

    def _coerce_to_dict(container: Any) -> Optional[dict]:
        """Accept dict or JSON-string (from multipart/form-data or extra_body).

        Without this, an attacker can smuggle guardrail keys past the check by
        sending ``{"metadata": "{\\"disable_global_guardrails\\": true}"}`` —
        ``isinstance(dict)`` on the string returns False, the check returns
        no-modification, and ``add_litellm_data_to_request`` parses the string
        to a dict downstream.
        """
        if isinstance(container, dict):
            return container
        if isinstance(container, str):
            parsed = safe_json_loads(container)
            return parsed if isinstance(parsed, dict) else None
        return None

    def _user_requested_modification(container: Any) -> bool:
        coerced = _coerce_to_dict(container)
        if coerced is None:
            return False
        return any(key in coerced for key in _GUARDRAIL_MODIFICATION_KEYS)

    # Check both metadata keys — callers can populate either depending on the
    # endpoint. Cover the top-level too so root-level injection is rejected.
    modifies = (
        _user_requested_modification(request_body.get("metadata"))
        or _user_requested_modification(request_body.get("litellm_metadata"))
        or _user_requested_modification(request_body)
    )
    if not modifies:
        return

    if not can_modify_guardrails(team_object):
        raise HTTPException(
            status_code=403,
            detail={
                "error": "Your team does not have permission to modify guardrails."
            },
        )


async def check_tools_allowlist(
    request_body: dict,
    valid_token: Optional[UserAPIKeyAuth],
    team_object: Optional[LiteLLM_TeamTable],
    route: str,
) -> None:
    """
    Enforce key/team tool allowlist (metadata.allowed_tools). No DB in hot path —
    effective allowlist is read from valid_token.metadata and valid_token.team_metadata.
    Raises ProxyException with tool_access_denied if a tool is not allowed.
    """
    from litellm.litellm_core_utils.api_route_to_call_types import (
        get_call_types_for_route,
    )

    if valid_token is None:
        return
    call_types = get_call_types_for_route(route)
    if not call_types or not any(
        ct.value in TOOL_CAPABLE_CALL_TYPES for ct in call_types
    ):
        return
    tool_names = extract_request_tool_names(route, request_body)
    if not tool_names:
        return
    key_meta = (
        (valid_token.metadata or {}) if isinstance(valid_token.metadata, dict) else {}
    )
    team_meta = (
        (valid_token.team_metadata or {})
        if isinstance(valid_token.team_metadata, dict)
        else {}
    )
    key_allowed = key_meta.get("allowed_tools")
    team_allowed = team_meta.get("allowed_tools")
    effective = (
        key_allowed
        if (isinstance(key_allowed, list) and len(key_allowed) > 0)
        else team_allowed
    )
    if not isinstance(effective, list) or len(effective) == 0:
        return
    allowed_set = {str(t) for t in effective}
    disallowed = [n for n in tool_names if n not in allowed_set]
    if disallowed:
        raise ProxyException(
            message=f"Tool(s) {disallowed} are not in the allowed tools list for this key/team.",
            type=ProxyErrorTypes.tool_access_denied,
            param="tools",
            code=status.HTTP_403_FORBIDDEN,
        )


async def common_checks(  # noqa: PLR0915
    request_body: dict,
    team_object: Optional[LiteLLM_TeamTable],
    user_object: Optional[LiteLLM_UserTable],
    end_user_object: Optional[LiteLLM_EndUserTable],
    global_proxy_spend: Optional[float],
    general_settings: dict,
    route: str,
    llm_router: Optional[Router],
    proxy_logging_obj: ProxyLogging,
    valid_token: Optional[UserAPIKeyAuth],
    request: Request,
    skip_budget_checks: bool = False,
    project_object: Optional[LiteLLM_ProjectTableCachedObj] = None,
) -> bool:
    """
    Common checks across jwt + key-based auth.

    1. If team is blocked
    1.1. If project is blocked
    2. If team can call model
    2.2 If project can call model
    3. If team is in budget
    3.0.2. If project is in budget
    3.0.3. If project is over soft budget (alert only)
    4. If user passed in (JWT or key.user_id) - is in budget
    5. If end_user (either via JWT or 'user' passed to /chat/completions, /embeddings endpoint) is in budget
    6. [OPTIONAL] If 'enforce_end_user' enabled - did developer pass in 'user' param for openai endpoints
    7. [OPTIONAL] If 'litellm.max_budget' is set (>0), is proxy under budget
    8. [OPTIONAL] If guardrails modified - is request allowed to change this
    9. Check if request body is safe
    10. [OPTIONAL] Organization checks - is user_object.organization_id is set, run these checks
    11. [OPTIONAL] Vector store checks - is the object allowed to access the vector store
    """
    from litellm.proxy.proxy_server import prisma_client, user_api_key_cache

    _model: Optional[Union[str, List[str]]] = get_model_from_request(
        request_data=request_body,
        route=route,
        request_headers=_safe_get_request_headers(request=request),
        request_query_params=_safe_get_request_query_params(request=request),
    )

    # 1. If team is blocked
    if team_object is not None and team_object.blocked is True:
        raise Exception(
            f"Team={team_object.team_id} is blocked. Update via `/team/unblock` if you're an admin."
        )

    # 2. If team can call model (or key's access_group_ids grant it)
    if _model and team_object:
        with tracer.trace("litellm.proxy.auth.common_checks.can_team_access_model"):
            try:
                await can_team_access_model(
                    model=_model,
                    team_object=team_object,
                    llm_router=llm_router,
                    team_model_aliases=(
                        valid_token.team_model_aliases if valid_token else None
                    ),
                )
            except ProxyException as team_denial:
                if team_denial.type != ProxyErrorTypes.team_model_access_denied:
                    raise
                if not await _key_access_group_grants_model(
                    model=_model,
                    valid_token=valid_token,
                    team_object=team_object,
                    llm_router=llm_router,
                ):
                    raise

    # 2.2. If team member has per-member model scope, enforce it
    if _model and team_object and valid_token and valid_token.user_id:
        with tracer.trace(
            "litellm.proxy.auth.common_checks.check_team_member_model_access"
        ):
            await _check_team_member_model_access(
                model=_model,
                team_object=team_object,
                valid_token=valid_token,
                llm_router=llm_router,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                proxy_logging_obj=proxy_logging_obj,
            )

    # Require trace id for agent keys when agent has require_trace_id_on_calls_by_agent
    if valid_token is not None and valid_token.agent_id:
        from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry
        from litellm.proxy.litellm_pre_call_utils import get_chain_id_from_headers

        agent = global_agent_registry.get_agent_by_id(agent_id=valid_token.agent_id)
        if agent is not None:
            require_trace_id = (agent.litellm_params or {}).get(
                "require_trace_id_on_calls_by_agent"
            )
            if require_trace_id:
                headers_dict = dict(request.headers)
                trace_id = get_chain_id_from_headers(headers_dict)
                if not trace_id:
                    raise ProxyException(
                        message="Requests made with this agent's key must include the x-litellm-trace-id header.",
                        type=ProxyErrorTypes.bad_request_error,
                        param=None,
                        code=status.HTTP_400_BAD_REQUEST,
                    )

    ## 2.1 If user can call model (if personal key)
    if _model and team_object is None and user_object is not None:
        with tracer.trace("litellm.proxy.auth.common_checks.can_user_call_model"):
            await can_user_call_model(
                model=_model,
                llm_router=llm_router,
                user_object=user_object,
            )

    # 1.1 - 2.2 - 3.0.2 - 3.0.3: Project checks (blocked, model access, budget)
    with tracer.trace("litellm.proxy.auth.common_checks.run_project_checks"):
        await _run_project_checks(
            project_object=project_object,
            _model=_model,
            llm_router=llm_router,
            skip_budget_checks=skip_budget_checks,
            valid_token=valid_token,
            proxy_logging_obj=proxy_logging_obj,
        )

    # If this is a free model, skip all budget checks
    if not skip_budget_checks:
        # 3. If team is in budget
        with tracer.trace("litellm.proxy.auth.common_checks.team_max_budget_check"):
            await _team_max_budget_check(
                team_object=team_object,
                proxy_logging_obj=proxy_logging_obj,
                valid_token=valid_token,
            )

        # 3.1. Multi-window budget check for team
        with tracer.trace("litellm.proxy.auth.common_checks.team_multi_budget_check"):
            await _team_multi_budget_check(team_object=team_object)

        # 3.2. Multi-window budget check for key
        with tracer.trace(
            "litellm.proxy.auth.common_checks.virtual_key_multi_budget_check"
        ):
            if valid_token is not None:
                await _virtual_key_multi_budget_check(valid_token=valid_token)

        # 3.0.5. If team is over soft budget (alert only, doesn't block)
        with tracer.trace("litellm.proxy.auth.common_checks.team_soft_budget_check"):
            await _team_soft_budget_check(
                team_object=team_object,
                proxy_logging_obj=proxy_logging_obj,
                valid_token=valid_token,
            )

        # 3.1. If organization is in budget
        with tracer.trace(
            "litellm.proxy.auth.common_checks.organization_max_budget_check"
        ):
            await _organization_max_budget_check(
                valid_token=valid_token,
                team_object=team_object,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                proxy_logging_obj=proxy_logging_obj,
            )

        with tracer.trace("litellm.proxy.auth.common_checks.tag_max_budget_check"):
            await _tag_max_budget_check(
                request_body=request_body,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                proxy_logging_obj=proxy_logging_obj,
                valid_token=valid_token,
            )

        # 4. If user is in budget
        ## 4.1 check personal budget, if personal key
        if (
            (team_object is None or team_object.team_id is None)
            and user_object is not None
            and user_object.max_budget is not None
        ):
            user_budget = user_object.max_budget
            from litellm.proxy.proxy_server import get_current_spend

            user_spend = await get_current_spend(
                counter_key=f"spend:user:{user_object.user_id}",
                fallback_spend=user_object.spend or 0.0,
            )
            if math.isfinite(user_budget) and user_spend >= user_budget:
                raise litellm.BudgetExceededError(
                    current_cost=user_spend,
                    max_budget=user_budget,
                    message=f"ExceededBudget: User={user_object.user_id} over budget. Spend={user_spend}, Budget={user_budget}",
                )

        ## 4.2 check team member budget, if team key
        with tracer.trace("litellm.proxy.auth.common_checks.check_team_member_budget"):
            await _check_team_member_budget(
                team_object=team_object,
                user_object=user_object,
                valid_token=valid_token,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                proxy_logging_obj=proxy_logging_obj,
            )

        # 5. If end_user ('user' passed to /chat/completions, /embeddings endpoint) is in budget
        if (
            end_user_object is not None
            and end_user_object.litellm_budget_table is not None
        ):
            await _check_end_user_budget(end_user_obj=end_user_object, route=route)

    _enforce_user_param_check(general_settings, request, request_body, route)
    _reject_clientside_metadata_tags_check(general_settings, request_body, route)
    _global_proxy_budget_check(global_proxy_spend, skip_budget_checks, route)
    _guardrail_modification_check(request_body, team_object)

    # 10 [OPTIONAL] Organization RBAC checks
    organization_role_based_access_check(
        user_object=user_object, route=route, request_body=request_body
    )

    _is_route_allowed = _is_api_route_allowed(
        route=route,
        request=request,
        request_data=request_body,
        valid_token=valid_token,
        user_obj=user_object,
    )

    # 11. [OPTIONAL] Vector store checks - is the object allowed to access the vector store
    with tracer.trace("litellm.proxy.auth.common_checks.vector_store_access_check"):
        await vector_store_access_check(
            request_body=request_body,
            team_object=team_object,
            valid_token=valid_token,
        )

    # 12. [OPTIONAL] Tool allowlist - key/team allowed_tools (no DB in hot path)
    with tracer.trace("litellm.proxy.auth.common_checks.check_tools_allowlist"):
        await check_tools_allowlist(
            request_body=request_body,
            valid_token=valid_token,
            team_object=team_object,
            route=route,
        )

    return True


def _get_user_role(
    user_obj: Optional[LiteLLM_UserTable],
) -> Optional[LitellmUserRoles]:
    if user_obj is None:
        return None

    _user = user_obj

    _user_role = _user.user_role
    try:
        role = LitellmUserRoles(_user_role)
    except ValueError:
        return LitellmUserRoles.INTERNAL_USER

    return role


def _is_api_route_allowed(
    route: str,
    request: Request,
    request_data: dict,
    valid_token: Optional[UserAPIKeyAuth],
    user_obj: Optional[LiteLLM_UserTable] = None,
) -> bool:
    """
    - Route b/w api token check and normal token check
    """
    _user_role = _get_user_role(user_obj=user_obj)

    if valid_token is None:
        raise Exception("Invalid proxy server token passed. valid_token=None.")

    if not _is_user_proxy_admin(user_obj=user_obj):  # if non-admin
        RouteChecks.non_proxy_admin_allowed_routes_check(
            user_obj=user_obj,
            _user_role=_user_role,
            route=route,
            request=request,
            request_data=request_data,
            valid_token=valid_token,
        )
    return True


def _is_user_proxy_admin(user_obj: Optional[LiteLLM_UserTable]):
    if user_obj is None:
        return False

    if (
        user_obj.user_role is not None
        and user_obj.user_role == LitellmUserRoles.PROXY_ADMIN.value
    ):
        return True

    if (
        user_obj.user_role is not None
        and user_obj.user_role == LitellmUserRoles.PROXY_ADMIN.value
    ):
        return True

    return False


def _allowed_routes_check(user_route: str, allowed_routes: list) -> bool:
    """
    Return if a user is allowed to access route. Helper function for `allowed_routes_check`.

    Parameters:
    - user_route: str - the route the user is trying to call
    - allowed_routes: List[str|LiteLLMRoutes] - the list of allowed routes for the user.
    """
    from starlette.routing import compile_path

    for allowed_route in allowed_routes:
        if allowed_route in LiteLLMRoutes.__members__:
            for template in LiteLLMRoutes[allowed_route].value:
                regex, _, _ = compile_path(template)
                if regex.match(user_route):
                    return True
        elif allowed_route == user_route:
            return True
    return False


def allowed_routes_check(
    user_role: LitellmUserRoles,
    user_route: str,
    litellm_proxy_roles: LiteLLM_JWTAuth,
) -> bool:
    """
    Check if user -> not admin - allowed to access these routes
    """

    if user_role == LitellmUserRoles.PROXY_ADMIN:
        is_allowed = _allowed_routes_check(
            user_route=user_route,
            allowed_routes=litellm_proxy_roles.admin_allowed_routes,
        )
        return is_allowed

    elif user_role == LitellmUserRoles.TEAM:
        if litellm_proxy_roles.team_allowed_routes is None:
            """
            By default allow a team to call openai + info routes
            """
            is_allowed = _allowed_routes_check(
                user_route=user_route, allowed_routes=["openai_routes", "info_routes"]
            )
            return is_allowed
        elif litellm_proxy_roles.team_allowed_routes is not None:
            is_allowed = _allowed_routes_check(
                user_route=user_route,
                allowed_routes=litellm_proxy_roles.team_allowed_routes,
            )
            return is_allowed
    return False


def allowed_route_check_inside_route(
    user_api_key_dict: UserAPIKeyAuth,
    requested_user_id: Optional[str],
) -> bool:
    ret_val = True
    if (
        user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN
        and user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY
    ):
        ret_val = False
    if requested_user_id is not None and user_api_key_dict.user_id is not None:
        if user_api_key_dict.user_id == requested_user_id:
            ret_val = True
    return ret_val


def get_actual_routes(allowed_routes: list) -> list:
    actual_routes: list = []
    for route_name in allowed_routes:
        try:
            route_value = LiteLLMRoutes[route_name].value
            if isinstance(route_value, set):
                actual_routes.extend(list(route_value))
            else:
                actual_routes.extend(route_value)

        except KeyError:
            actual_routes.append(route_name)
    return actual_routes


async def get_default_end_user_budget(
    prisma_client: Optional[PrismaClient],
    user_api_key_cache: UserApiKeyCache,
    parent_otel_span: Optional[Span] = None,
) -> Optional[LiteLLM_BudgetTable]:
    """
    Fetches the default end user budget from the database if litellm.max_end_user_budget_id is configured.

    This budget is applied to end users who don't have an explicit budget_id set.
    Results are cached for performance.

    Args:
        prisma_client: Database client instance
        user_api_key_cache: Cache for storing/retrieving budget data
        parent_otel_span: Optional OpenTelemetry span for tracing

    Returns:
        LiteLLM_BudgetTable if configured and found, None otherwise
    """
    if prisma_client is None or litellm.max_end_user_budget_id is None:
        return None

    cache_key = f"default_end_user_budget:{litellm.max_end_user_budget_id}"

    # Check cache first
    cached_budget = await user_api_key_cache.async_get_cache(
        key=cache_key,
        model_type=LiteLLM_BudgetTable,
    )
    if cached_budget is not None:
        return cached_budget

    # Fetch from database
    try:
        budget_record = await prisma_client.db.litellm_budgettable.find_unique(
            where={"budget_id": litellm.max_end_user_budget_id}
        )

        if budget_record is None:
            verbose_proxy_logger.warning(
                f"Default end user budget not found in database: {litellm.max_end_user_budget_id}"
            )
            return None

        _budget_obj = LiteLLM_BudgetTable(**budget_record.dict())
        # Cache the budget for 60 seconds
        await user_api_key_cache.async_set_cache(
            key=cache_key,
            value=_budget_obj,
            model_type=LiteLLM_BudgetTable,
            ttl=_management_object_cache_ttl(user_api_key_cache),
        )

        return _budget_obj

    except Exception as e:
        verbose_proxy_logger.error(f"Error fetching default end user budget: {str(e)}")
        return None


@log_db_metrics
async def get_team_member_default_budget(
    budget_id: str,
    prisma_client: Optional[PrismaClient],
    user_api_key_cache: UserApiKeyCache,
) -> Optional[LiteLLM_BudgetTable]:
    """
    Fetches the team-level default per-member budget referenced by team.metadata["team_member_budget_id"].

    This budget is applied to team members whose TeamMembership row has no
    linked budget, or whose linked budget has max_budget=NULL. Results are
    cached for performance.

    Args:
        budget_id: The budget_id pulled from team.metadata["team_member_budget_id"]
        prisma_client: Database client instance
        user_api_key_cache: Cache for storing/retrieving budget data

    Returns:
        LiteLLM_BudgetTable if found, None otherwise
    """
    if prisma_client is None:
        return None

    cache_key = f"team_member_default_budget:{budget_id}"

    cached_budget = await user_api_key_cache.async_get_cache(key=cache_key)
    if isinstance(cached_budget, LiteLLM_BudgetTable):
        return cached_budget
    if isinstance(cached_budget, dict):
        return LiteLLM_BudgetTable(**cached_budget)

    try:
        budget_record = await prisma_client.db.litellm_budgettable.find_unique(
            where={"budget_id": budget_id}
        )

        if budget_record is None:
            verbose_proxy_logger.warning(
                f"Team-default member budget not found in database: {budget_id}"
            )
            return None

        await user_api_key_cache.async_set_cache(
            key=cache_key,
            value=budget_record.dict(),
            ttl=_management_object_cache_ttl(user_api_key_cache),
        )

        return LiteLLM_BudgetTable(**budget_record.dict())

    except Exception:
        verbose_proxy_logger.exception(
            f"Error fetching team-default member budget {budget_id}"
        )
        return None


async def _apply_default_budget_to_end_user(
    end_user_obj: LiteLLM_EndUserTable,
    prisma_client: PrismaClient,
    user_api_key_cache: UserApiKeyCache,
    parent_otel_span: Optional[Span] = None,
) -> LiteLLM_EndUserTable:
    """
    Helper function to apply default budget to end user if they don't have a budget assigned.

    Args:
        end_user_obj: The end user object to potentially apply default budget to
        prisma_client: Database client instance
        user_api_key_cache: Cache for storing/retrieving data
        parent_otel_span: Optional OpenTelemetry span for tracing

    Returns:
        Updated end user object with default budget applied if applicable
    """
    # If end user already has a budget assigned, no need to apply default
    if end_user_obj.litellm_budget_table is not None:
        return end_user_obj

    # If no default budget configured, return as-is
    if litellm.max_end_user_budget_id is None:
        return end_user_obj

    # Fetch and apply default budget
    default_budget = await get_default_end_user_budget(
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        parent_otel_span=parent_otel_span,
    )

    if default_budget is not None:
        # Apply default budget to end user object
        end_user_obj.litellm_budget_table = default_budget
        verbose_proxy_logger.debug(
            f"Applied default budget {litellm.max_end_user_budget_id} to end user {end_user_obj.user_id}"
        )

    return end_user_obj


async def _check_end_user_budget(
    end_user_obj: LiteLLM_EndUserTable,
    route: str,
) -> None:
    """
    Check if end user is within their budget limit.

    Args:
        end_user_obj: The end user object to check
        route: The request route

    Raises:
        litellm.BudgetExceededError: If end user has exceeded their budget
    """
    if RouteChecks.is_info_route(route):
        return

    if end_user_obj.litellm_budget_table is None:
        return

    end_user_budget = end_user_obj.litellm_budget_table.max_budget
    if end_user_budget is None:
        return

    from litellm.proxy.proxy_server import get_current_spend

    end_user_spend = await get_current_spend(
        counter_key=f"spend:end_user:{end_user_obj.user_id}",
        fallback_spend=end_user_obj.spend or 0.0,
    )
    if end_user_spend > end_user_budget:
        raise litellm.BudgetExceededError(
            current_cost=end_user_spend,
            max_budget=end_user_budget,
            message=f"ExceededBudget: End User={end_user_obj.user_id} over budget. Spend={end_user_spend}, Budget={end_user_budget}",
        )


@log_db_metrics
async def get_end_user_object(
    end_user_id: Optional[str],
    prisma_client: Optional[PrismaClient],
    user_api_key_cache: UserApiKeyCache,
    route: str,
    parent_otel_span: Optional[Span] = None,
    proxy_logging_obj: Optional[ProxyLogging] = None,
) -> Optional[LiteLLM_EndUserTable]:
    """
    Returns end user object from database or cache.

    If end user exists but has no budget_id, applies the default budget
    (if configured via litellm.max_end_user_budget_id).

    Args:
        end_user_id: The ID of the end user
        prisma_client: Database client instance
        user_api_key_cache: Cache for storing/retrieving data
        route: The request route
        parent_otel_span: Optional OpenTelemetry span for tracing
        proxy_logging_obj: Optional proxy logging object

    Returns:
        LiteLLM_EndUserTable if found, None otherwise
    """
    if prisma_client is None:
        raise Exception("No db connected")

    if end_user_id is None:
        return None

    _key = "end_user_id:{}".format(end_user_id)

    # Check cache first
    cached_user_obj = await user_api_key_cache.async_get_cache(
        key=_key,
        model_type=LiteLLM_EndUserTable,
    )
    if cached_user_obj is not None:
        return_obj = cached_user_obj
        # Apply default budget if needed
        return_obj = await _apply_default_budget_to_end_user(
            end_user_obj=return_obj,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=parent_otel_span,
        )

        # Check budget limits
        await _check_end_user_budget(end_user_obj=return_obj, route=route)

        return return_obj

    # Fetch from database
    try:
        response = await prisma_client.db.litellm_endusertable.find_unique(
            where={"user_id": end_user_id},
            include={"litellm_budget_table": True, "object_permission": True},
        )

        if response is None:
            raise Exception

        # Convert to LiteLLM_EndUserTable object
        _response = LiteLLM_EndUserTable(**response.dict())

        # Apply default budget if needed
        _response = await _apply_default_budget_to_end_user(
            end_user_obj=_response,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=parent_otel_span,
        )

        # Save to cache
        await user_api_key_cache.async_set_cache(
            key="end_user_id:{}".format(end_user_id),
            value=_response,
            model_type=LiteLLM_EndUserTable,
        )

        # Check budget limits
        await _check_end_user_budget(end_user_obj=_response, route=route)

        return _response

    except Exception as e:
        if isinstance(e, litellm.BudgetExceededError):
            raise e
        return None


@log_db_metrics
async def get_tag_objects_batch(
    tag_names: List[str],
    prisma_client: Optional[PrismaClient],
    user_api_key_cache: UserApiKeyCache,
    parent_otel_span: Optional[Span] = None,
    proxy_logging_obj: Optional[ProxyLogging] = None,
) -> Dict[str, LiteLLM_TagTable]:
    """
    Batch fetch multiple tag objects from cache and db.

    Optimizes for latency by:
    1. Fetching all cached tags in parallel
    2. Batch fetching uncached tags in one DB query

    Args:
        tag_names: List of tag names to fetch
        prisma_client: Prisma database client
        user_api_key_cache: Cache for storing tag objects
        parent_otel_span: Optional OpenTelemetry span for tracing
        proxy_logging_obj: Optional proxy logging object

    Returns:
        Dictionary mapping tag_name to LiteLLM_TagTable object
    """
    if prisma_client is None:
        return {}

    if not tag_names:
        return {}

    tag_objects = {}
    uncached_tags = []

    # Try to get all tags from cache first
    for tag_name in tag_names:
        cache_key = f"tag:{tag_name}"
        cached_tag = await user_api_key_cache.async_get_cache(
            key=cache_key,
            model_type=LiteLLM_TagTable,
        )
        if cached_tag is not None:
            tag_objects[tag_name] = cached_tag
        else:
            uncached_tags.append(tag_name)

    # Batch fetch uncached tags from DB in one query
    if uncached_tags:
        try:
            db_tags = await prisma_client.db.litellm_tagtable.find_many(
                where={"tag_name": {"in": uncached_tags}},
                include={"litellm_budget_table": True},
            )

            # Cache and add to tag_objects
            for db_tag in db_tags:
                tag_name = db_tag.tag_name
                cache_key = f"tag:{tag_name}"
                _tag_obj = LiteLLM_TagTable(**db_tag.dict())
                await user_api_key_cache.async_set_cache(
                    key=cache_key,
                    value=_tag_obj,
                    model_type=LiteLLM_TagTable,
                )
                tag_objects[tag_name] = _tag_obj
        except Exception as e:
            verbose_proxy_logger.debug(f"Error batch fetching tags from database: {e}")

    return tag_objects


@log_db_metrics
async def get_tag_object(
    tag_name: Optional[str],
    prisma_client: Optional[PrismaClient],
    user_api_key_cache: UserApiKeyCache,
    parent_otel_span: Optional[Span] = None,
    proxy_logging_obj: Optional[ProxyLogging] = None,
) -> Optional[LiteLLM_TagTable]:
    """
    Returns tag object from cache or db.

    Uses default cache TTL (same as end_user objects) to avoid drift.

    Args:
        tag_name: Name of the tag to fetch
        prisma_client: Prisma database client
        user_api_key_cache: Cache for storing tag objects
        parent_otel_span: Optional OpenTelemetry span for tracing
        proxy_logging_obj: Optional proxy logging object

    Returns:
        LiteLLM_TagTable object if found, None otherwise
    """
    if prisma_client is None or tag_name is None:
        return None

    # Use batch helper for consistency
    tag_objects = await get_tag_objects_batch(
        tag_names=[tag_name],
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        parent_otel_span=parent_otel_span,
        proxy_logging_obj=proxy_logging_obj,
    )

    return tag_objects.get(tag_name)


@log_db_metrics
async def get_team_membership(
    user_id: str,
    team_id: str,
    prisma_client: Optional[PrismaClient],
    user_api_key_cache: UserApiKeyCache,
    parent_otel_span: Optional[Span] = None,
    proxy_logging_obj: Optional[ProxyLogging] = None,
) -> Optional["LiteLLM_TeamMembership"]:
    """
    Returns team membership object if user is member of team.

    Do a isolated check for team membership vs. doing a combined key + team + user + team-membership check, as key might come in frequently for different users/teams. Larger call will slowdown query time. This way we get to cache the constant (key/team/user info) and only update based on the changing value (team membership).
    """
    from litellm.proxy._types import LiteLLM_TeamMembership

    if prisma_client is None:
        raise Exception("No db connected")

    if user_id is None or team_id is None:
        return None

    _key = "team_membership:{}:{}".format(user_id, team_id)

    # check if in cache
    cached_membership_obj = await user_api_key_cache.async_get_cache(
        key=_key,
        model_type=LiteLLM_TeamMembership,
    )
    if cached_membership_obj is not None:
        return cached_membership_obj

    # else, check db
    try:
        response = await prisma_client.db.litellm_teammembership.find_unique(
            where={"user_id_team_id": {"user_id": user_id, "team_id": team_id}},
            include={"litellm_budget_table": True},
        )

        if response is None:
            return None

        _response = LiteLLM_TeamMembership(**response.dict())
        await user_api_key_cache.async_set_cache(
            key=_key,
            value=_response,
            model_type=LiteLLM_TeamMembership,
        )

        return _response
    except Exception:
        verbose_proxy_logger.exception(
            "Error getting team membership for user_id: %s, team_id: %s",
            user_id,
            team_id,
        )
        return None


def model_in_access_group(
    model: str, team_models: Optional[List[str]], llm_router: Optional[Router]
) -> bool:
    from collections import defaultdict

    if team_models is None:
        return True
    if model in team_models:
        return True

    access_groups: dict[str, list[str]] = defaultdict(list)
    if llm_router:
        access_groups = llm_router.get_model_access_groups(model_name=model)

    if len(access_groups) > 0:  # check if token contains any model access groups
        for idx, m in enumerate(
            team_models
        ):  # loop token models, if any of them are an access group add the access group
            if m in access_groups:
                return True

    # Filter out models that are access_groups
    filtered_models = [m for m in team_models if m not in access_groups]

    if model in filtered_models:
        return True

    return False


def _should_check_db(
    key: str, last_db_access_time: LimitedSizeOrderedDict, db_cache_expiry: int
) -> bool:
    """
    Prevent calling db repeatedly for items that don't exist in the db.
    """
    current_time = time.time()
    # if key doesn't exist in last_db_access_time -> check db
    if key not in last_db_access_time:
        return True
    elif (
        last_db_access_time[key][0] is not None
    ):  # check db for non-null values (for refresh operations)
        return True
    elif last_db_access_time[key][0] is None:
        if current_time - last_db_access_time[key] >= db_cache_expiry:
            return True
    return False


def _update_last_db_access_time(
    key: str, value: Optional[Any], last_db_access_time: LimitedSizeOrderedDict
):
    last_db_access_time[key] = (value, time.time())


def _get_role_based_permissions(
    rbac_role: RBAC_ROLES,
    general_settings: dict,
    key: Literal["models", "routes"],
) -> Optional[List[str]]:
    """
    Get the role based permissions from the general settings.
    """
    role_based_permissions = cast(
        Optional[List[RoleBasedPermissions]],
        general_settings.get("role_permissions", []),
    )
    if role_based_permissions is None:
        return None

    for role_based_permission in role_based_permissions:
        if role_based_permission.role == rbac_role:
            return getattr(role_based_permission, key)

    return None


def get_role_based_models(
    rbac_role: RBAC_ROLES,
    general_settings: dict,
) -> Optional[List[str]]:
    """
    Get the models allowed for a user role.

    Used by JWT Auth.
    """

    return _get_role_based_permissions(
        rbac_role=rbac_role,
        general_settings=general_settings,
        key="models",
    )


def get_role_based_routes(
    rbac_role: RBAC_ROLES,
    general_settings: dict,
) -> Optional[List[str]]:
    """
    Get the routes allowed for a user role.
    """

    return _get_role_based_permissions(
        rbac_role=rbac_role,
        general_settings=general_settings,
        key="routes",
    )


async def _get_fuzzy_user_object(
    prisma_client: PrismaClient,
    sso_user_id: Optional[str] = None,
    user_email: Optional[str] = None,
) -> Optional[LiteLLM_UserTable]:
    """
    Checks if sso user is in db.

    Called when user id match is not found in db.

    - Check if sso_user_id is user_id in db
    - Check if sso_user_id is sso_user_id in db
    - Check if user_email is user_email in db
    - If not, create new user with user_email and sso_user_id and user_id = sso_user_id
    """

    response = None
    if sso_user_id is not None:
        response = await prisma_client.db.litellm_usertable.find_unique(
            where={"sso_user_id": sso_user_id},
            include={"organization_memberships": True},
        )

    if response is None and user_email is not None:
        # Use case-insensitive query to handle emails with different casing
        # This matches the pattern used in _check_duplicate_user_email
        response = await prisma_client.db.litellm_usertable.find_first(
            where={"user_email": {"equals": user_email, "mode": "insensitive"}},
            include={"organization_memberships": True},
        )

        if response is not None and sso_user_id is not None:  # update sso_user_id
            asyncio.create_task(  # background task to update user with sso id
                prisma_client.db.litellm_usertable.update(
                    where={"user_id": response.user_id},
                    data={"sso_user_id": sso_user_id},
                )
            )

    return response


@log_db_metrics
async def get_user_object(
    user_id: Optional[str],
    prisma_client: Optional[PrismaClient],
    user_api_key_cache: UserApiKeyCache,
    user_id_upsert: bool,
    parent_otel_span: Optional[Span] = None,
    proxy_logging_obj: Optional[ProxyLogging] = None,
    sso_user_id: Optional[str] = None,
    user_email: Optional[str] = None,
    check_db_only: Optional[bool] = None,
) -> Optional[LiteLLM_UserTable]:
    """
    - Check if user id in proxy User Table
    - if valid, return LiteLLM_UserTable object with defined limits
    - if not, then raise an error
    """

    if user_id is None:
        return None

    # check if in cache
    if not check_db_only:
        cached_user_obj = await user_api_key_cache.async_get_cache(
            key=user_id,
            model_type=LiteLLM_UserTable,
        )
        if cached_user_obj is not None:
            return cached_user_obj
    # else, check db
    if prisma_client is None:
        raise Exception("No db connected")
    try:
        db_access_time_key = "user_id:{}".format(user_id)
        should_check_db = _should_check_db(
            key=db_access_time_key,
            last_db_access_time=last_db_access_time,
            db_cache_expiry=db_cache_expiry,
        )

        if should_check_db:
            response = await prisma_client.db.litellm_usertable.find_unique(
                where={"user_id": user_id}, include={"organization_memberships": True}
            )

            if response is None:
                response = await _get_fuzzy_user_object(
                    prisma_client=prisma_client,
                    sso_user_id=sso_user_id,
                    user_email=user_email,
                )

        else:
            response = None

        if response is None:
            if user_id_upsert:
                new_user_params: Dict[str, Any] = {
                    "user_id": user_id,
                }
                if user_email is not None:
                    new_user_params["user_email"] = user_email
                if litellm.default_internal_user_params is not None:
                    new_user_params.update(litellm.default_internal_user_params)

                response = await prisma_client.db.litellm_usertable.create(
                    data=new_user_params,
                    include={"organization_memberships": True},
                )
            else:
                raise Exception

        if (
            response.organization_memberships is not None
            and len(response.organization_memberships) > 0
        ):
            # dump each organization membership to type LiteLLM_OrganizationMembershipTable
            _dumped_memberships = [
                LiteLLM_OrganizationMembershipTable(**membership.model_dump())
                for membership in response.organization_memberships
                if membership is not None
            ]
            response.organization_memberships = _dumped_memberships

        _response = LiteLLM_UserTable(**dict(response))
        response_dict = _response.model_dump()

        # save the user object to cache
        await user_api_key_cache.async_set_cache(
            key=user_id,
            value=_response,
            model_type=LiteLLM_UserTable,
            ttl=_management_object_cache_ttl(user_api_key_cache),
        )

        # save to db access time
        _update_last_db_access_time(
            key=db_access_time_key,
            value=response_dict,
            last_db_access_time=last_db_access_time,
        )

        return _response
    except Exception as e:  # if user not in db
        _log_budget_lookup_failure("user", e)
        raise ValueError(
            f"User doesn't exist in db. 'user_id'={user_id}. Create user via `/user/new` call. Got error - {e}"
        )


def _management_object_cache_ttl(user_api_key_cache: "UserApiKeyCache") -> float:
    """
    TTL (in seconds) for management-object cache writes.

    ``user_api_key_cache.default_in_memory_ttl`` is the source of truth: it is
    initialised to ``UserAPIKeyCacheTTLEnum.in_memory_cache_ttl.value`` (60s)
    at proxy startup and then overridden by
    ``general_settings.user_api_key_cache_ttl`` when the user configures one
    (see ``proxy_server.py``'s ``user_api_key_cache.update_cache_ttl(...)``
    call).

    Reading it here keeps every management-object cache write
    (``_cache_management_object`` and friends) in sync with the user's
    configured TTL, instead of hard-coding
    ``DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL`` and silently capping the
    cache at 60s. Falls back to ``DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL``
    only when the cache exposes no default (e.g. a bare cache constructed in a
    test).
    """
    configured = getattr(user_api_key_cache, "default_in_memory_ttl", None)
    if configured is not None:
        return float(configured)
    return float(DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL)


async def _cache_management_object(
    key: str,
    value: Union[BaseModel, Dict[str, Any]],
    user_api_key_cache: UserApiKeyCache,
    proxy_logging_obj: Optional[ProxyLogging],
    *,
    model_type: Type[BaseModel],
):
    """
    Persist management objects via ``UserApiKeyCache`` (in-memory + optional Redis).

    ``UserApiKeyCache`` serializes with ``model_type`` so Redis and in-memory
    stay aligned. The TTL honours ``general_settings.user_api_key_cache_ttl``
    when configured; see :func:`_management_object_cache_ttl`.
    """
    await user_api_key_cache.async_set_cache(
        key=key,
        value=value,
        model_type=model_type,
        ttl=_management_object_cache_ttl(user_api_key_cache),
    )


async def _cache_team_object(
    team_id: str,
    team_table: LiteLLM_TeamTableCachedObj,
    user_api_key_cache: UserApiKeyCache,
    proxy_logging_obj: Optional[ProxyLogging],
):
    key = "team_id:{}".format(team_id)

    ## CACHE REFRESH TIME!
    team_table.last_refreshed_at = time.time()

    await _cache_management_object(
        key=key,
        value=team_table,
        user_api_key_cache=user_api_key_cache,
        proxy_logging_obj=proxy_logging_obj,
        model_type=LiteLLM_TeamTableCachedObj,
    )


async def _cache_key_object(
    hashed_token: str,
    user_api_key_obj: UserAPIKeyAuth,
    user_api_key_cache: UserApiKeyCache,
    proxy_logging_obj: Optional[ProxyLogging],
):
    key = hashed_token

    ## CACHE REFRESH TIME
    user_api_key_obj.last_refreshed_at = time.time()

    cached_key_obj = _copy_user_api_key_auth_for_cache(
        user_api_key_obj=user_api_key_obj
    )
    await _cache_management_object(
        key=key,
        value=cached_key_obj,
        user_api_key_cache=user_api_key_cache,
        proxy_logging_obj=proxy_logging_obj,
        model_type=UserAPIKeyAuth,
    )


async def _delete_cache_key_object(
    hashed_token: str,
    user_api_key_cache: UserApiKeyCache,
    proxy_logging_obj: Optional[ProxyLogging],
):
    key = hashed_token

    user_api_key_cache.delete_cache(key=key)

    ## UPDATE REDIS CACHE ##
    if proxy_logging_obj is not None:
        await proxy_logging_obj.internal_usage_cache.dual_cache.async_delete_cache(
            key=key
        )


@log_db_metrics
async def _get_team_db_check(
    team_id: str, prisma_client: PrismaClient, team_id_upsert: Optional[bool] = None
):
    response = await prisma_client.db.litellm_teamtable.find_unique(
        where={"team_id": team_id}
    )

    if response is None and team_id_upsert:
        from litellm.proxy.management_endpoints.team_endpoints import new_team

        new_team_data = NewTeamRequest(team_id=team_id)

        mock_request = Request(scope={"type": "http"})
        system_admin_user = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)

        created_team_dict = await new_team(
            data=new_team_data,
            http_request=mock_request,
            user_api_key_dict=system_admin_user,
        )
        response = LiteLLM_TeamTable(**created_team_dict)
    return response


async def _get_team_object_from_db(team_id: str, prisma_client: PrismaClient):
    return await prisma_client.db.litellm_teamtable.find_unique(
        where={"team_id": team_id}
    )


async def _get_team_object_from_user_api_key_cache(
    team_id: str,
    prisma_client: PrismaClient,
    user_api_key_cache: UserApiKeyCache,
    last_db_access_time: LimitedSizeOrderedDict,
    db_cache_expiry: int,
    proxy_logging_obj: Optional[ProxyLogging],
    key: str,
    team_id_upsert: Optional[bool] = None,
) -> LiteLLM_TeamTableCachedObj:
    db_access_time_key = key
    should_check_db = _should_check_db(
        key=db_access_time_key,
        last_db_access_time=last_db_access_time,
        db_cache_expiry=db_cache_expiry,
    )
    if should_check_db:
        response = await _get_team_db_check(
            team_id=team_id, prisma_client=prisma_client, team_id_upsert=team_id_upsert
        )
    else:
        response = None

    if response is None:
        raise Exception

    _response = LiteLLM_TeamTableCachedObj(**response.dict())

    # Load object_permission if object_permission_id exists but object_permission is not loaded
    if _response.object_permission_id and not _response.object_permission:
        try:
            _response.object_permission = await get_object_permission(
                object_permission_id=_response.object_permission_id,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                parent_otel_span=None,
                proxy_logging_obj=proxy_logging_obj,
            )
        except Exception as e:
            verbose_proxy_logger.debug(
                f"Failed to load object_permission for team {team_id} with object_permission_id={_response.object_permission_id}: {e}"
            )

    # save the team object to cache
    await _cache_team_object(
        team_id=team_id,
        team_table=_response,
        user_api_key_cache=user_api_key_cache,
        proxy_logging_obj=proxy_logging_obj,
    )

    # save to db access time
    _update_last_db_access_time(
        key=db_access_time_key,
        value=_response,
        last_db_access_time=last_db_access_time,
    )

    return _response


async def _get_team_object_from_cache(
    key: str,
    proxy_logging_obj: Optional[ProxyLogging],
    user_api_key_cache: UserApiKeyCache,
    parent_otel_span: Optional[Span],
) -> Optional[LiteLLM_TeamTableCachedObj]:
    ## INTERNAL USAGE CACHE (plain DualCache) — checked before UserApiKeyCache stores ##
    if (
        proxy_logging_obj is not None
        and proxy_logging_obj.internal_usage_cache.dual_cache
    ):
        cached_raw = (
            await proxy_logging_obj.internal_usage_cache.dual_cache.async_get_cache(
                key=key, parent_otel_span=parent_otel_span
            )
        )
        if cached_raw is not None:
            from_internal = CacheCodec.deserialize(
                cached_raw, LiteLLM_TeamTableCachedObj
            )
            if from_internal is not None:
                return from_internal

    decoded = await user_api_key_cache.async_get_cache(
        key=key,
        parent_otel_span=parent_otel_span,
        model_type=LiteLLM_TeamTableCachedObj,
    )
    return decoded


async def get_team_object(
    team_id: str,
    prisma_client: Optional[PrismaClient],
    user_api_key_cache: UserApiKeyCache,
    parent_otel_span: Optional[Span] = None,
    proxy_logging_obj: Optional[ProxyLogging] = None,
    check_cache_only: Optional[bool] = None,
    check_db_only: Optional[bool] = None,
    team_id_upsert: Optional[bool] = None,
) -> LiteLLM_TeamTableCachedObj:
    """
    - Check if team id in proxy Team Table
    - if valid, return LiteLLM_TeamTable object with defined limits
    - if not, then raise an error

    Raises:
        - HTTPException: If team doesn't exist in db or cache (status_code=404)
    """
    if prisma_client is None:
        raise Exception(
            "No DB Connected. See - https://docs.litellm.ai/docs/proxy/virtual_keys"
        )

    # check if in cache
    key = "team_id:{}".format(team_id)

    if not check_db_only:
        cached_team_obj = await _get_team_object_from_cache(
            key=key,
            proxy_logging_obj=proxy_logging_obj,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=parent_otel_span,
        )

        if cached_team_obj is not None:
            return cached_team_obj

        if check_cache_only:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": f"Team doesn't exist in cache + check_cache_only=True. Team={team_id}."
                },
            )

    # else, check db
    try:
        return await _get_team_object_from_user_api_key_cache(
            team_id=team_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
            last_db_access_time=last_db_access_time,
            db_cache_expiry=db_cache_expiry,
            key=key,
            team_id_upsert=team_id_upsert,
        )
    except Exception:
        raise HTTPException(
            status_code=404,
            detail={
                "error": f"Team doesn't exist in db. Team={team_id}. Create team via `/team/new` call."
            },
        )


async def _cache_access_object(
    access_group_id: str,
    access_group_table: LiteLLM_AccessGroupTable,
    user_api_key_cache: UserApiKeyCache,
    proxy_logging_obj: Optional[ProxyLogging] = None,
):
    key = "access_group_id:{}".format(access_group_id)
    await user_api_key_cache.async_set_cache(
        key=key,
        value=access_group_table,
        model_type=LiteLLM_AccessGroupTable,
        ttl=DEFAULT_ACCESS_GROUP_CACHE_TTL,
    )


async def _delete_cache_access_object(
    access_group_id: str,
    user_api_key_cache: UserApiKeyCache,
    proxy_logging_obj: Optional[ProxyLogging] = None,
):
    key = "access_group_id:{}".format(access_group_id)

    user_api_key_cache.delete_cache(key=key)

    ## UPDATE REDIS CACHE ##
    if proxy_logging_obj is not None:
        await proxy_logging_obj.internal_usage_cache.dual_cache.async_delete_cache(
            key=key
        )


@log_db_metrics
async def get_access_object(
    access_group_id: str,
    prisma_client: Optional[PrismaClient],
    user_api_key_cache: UserApiKeyCache,
    proxy_logging_obj: Optional[ProxyLogging] = None,
) -> LiteLLM_AccessGroupTable:
    """
    - Check if access_group_id in proxy AccessGroupTable
    - Always checks cache first, then DB only when not found in cache
    - if valid, return LiteLLM_AccessGroupTable object
    - if not, then raise an error

    Unlike get_team_object, this has no check_cache_only or check_db_only flags;
    it always follows cache-first-then-db semantics.

    Raises:
        - HTTPException: If access group doesn't exist in db or cache (status_code=404)
    """
    if prisma_client is None:
        raise Exception(
            "No DB Connected. See - https://docs.litellm.ai/docs/proxy/virtual_keys"
        )

    key = "access_group_id:{}".format(access_group_id)

    cached_access_obj = await user_api_key_cache.async_get_cache(
        key=key,
        model_type=LiteLLM_AccessGroupTable,
    )
    if cached_access_obj is not None:
        return cached_access_obj

    # Not in cache - fetch from DB
    try:
        response = await prisma_client.db.litellm_accessgrouptable.find_unique(
            where={"access_group_id": access_group_id}
        )

        if response is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": f"Access group doesn't exist in db. Access group={access_group_id}."
                },
            )

        _response = LiteLLM_AccessGroupTable(**response.dict())

        # Save to cache
        await _cache_access_object(
            access_group_id=access_group_id,
            access_group_table=_response,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )

        return _response
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(
            "Error getting access group for access_group_id: %s",
            access_group_id,
        )
        raise HTTPException(
            status_code=404,
            detail={
                "error": f"Access group doesn't exist in db. Access group={access_group_id}. Error: {e}"
            },
        )


@log_db_metrics
async def get_team_object_by_alias(
    team_alias: str,
    prisma_client: Optional[PrismaClient],
    user_api_key_cache: UserApiKeyCache,
    parent_otel_span: Optional["Span"] = None,
    proxy_logging_obj: Optional[ProxyLogging] = None,
) -> LiteLLM_TeamTableCachedObj:
    """
    Look up a team by its team_alias (name) in the database.

    Args:
        team_alias: The team name/alias to look up
        prisma_client: Database client
        user_api_key_cache: Cache for storing results
        parent_otel_span: Optional OpenTelemetry span
        proxy_logging_obj: Optional proxy logging object

    Returns:
        LiteLLM_TeamTableCachedObj: The team object if found

    Raises:
        HTTPException: If team doesn't exist or multiple teams have the same alias
    """
    if prisma_client is None:
        raise Exception(
            "No DB Connected. See - https://docs.litellm.ai/docs/proxy/virtual_keys"
        )

    # Check cache first (keyed by alias)
    cache_key = "team_alias:{}".format(team_alias)

    cached_team_obj = await _get_team_object_from_cache(
        key=cache_key,
        proxy_logging_obj=proxy_logging_obj,
        user_api_key_cache=user_api_key_cache,
        parent_otel_span=parent_otel_span,
    )

    if cached_team_obj is not None:
        return cached_team_obj

    # Query database by team_alias
    try:
        teams = await prisma_client.db.litellm_teamtable.find_many(
            where={"team_alias": team_alias}
        )

        if not teams:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": f"Team with alias '{team_alias}' doesn't exist in db. Create team via `/team/new` call."
                },
            )

        if len(teams) > 1:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": f"Multiple teams found with alias '{team_alias}'. Please use team_id_jwt_field instead or ensure team aliases are unique."
                },
            )

        team = teams[0]
        team_obj = LiteLLM_TeamTableCachedObj(**team.model_dump())

        # Load object_permission if object_permission_id exists but object_permission is not loaded
        if team_obj.object_permission_id and not team_obj.object_permission:
            try:
                team_obj.object_permission = await get_object_permission(
                    object_permission_id=team_obj.object_permission_id,
                    prisma_client=prisma_client,
                    user_api_key_cache=user_api_key_cache,
                    parent_otel_span=parent_otel_span,
                    proxy_logging_obj=proxy_logging_obj,
                )
            except Exception as e:
                verbose_proxy_logger.debug(
                    f"Failed to load object_permission for team {team_obj.team_id} with object_permission_id={team_obj.object_permission_id}: {e}"
                )

        # Cache the result by both alias and team_id
        await user_api_key_cache.async_set_cache(
            key=cache_key,
            value=team_obj,
            model_type=LiteLLM_TeamTableCachedObj,
            ttl=DEFAULT_IN_MEMORY_TTL,
        )
        # Also cache by team_id for consistency
        team_id_cache_key = "team_id:{}".format(team_obj.team_id)
        await user_api_key_cache.async_set_cache(
            key=team_id_cache_key,
            value=team_obj,
            model_type=LiteLLM_TeamTableCachedObj,
            ttl=DEFAULT_IN_MEMORY_TTL,
        )

        return team_obj

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception("Error looking up team by alias: %s", team_alias)
        raise HTTPException(
            status_code=500,
            detail={
                "error": f"Error looking up team by alias '{team_alias}': {str(e)}"
            },
        )


@log_db_metrics
async def get_org_object_by_alias(
    org_alias: str,
    prisma_client: Optional[PrismaClient],
    user_api_key_cache: UserApiKeyCache,
    parent_otel_span: Optional["Span"] = None,
    proxy_logging_obj: Optional[ProxyLogging] = None,
) -> Optional[LiteLLM_OrganizationTable]:
    """
    Look up an organization by its organization_alias in the database.

    Args:
        org_alias: The organization name/alias to look up
        prisma_client: Database client
        user_api_key_cache: Cache for storing results
        parent_otel_span: Optional OpenTelemetry span
        proxy_logging_obj: Optional proxy logging object

    Returns:
        LiteLLM_OrganizationTable if found, None otherwise

    Raises:
        HTTPException: If organization not found or multiple orgs have the same alias
    """
    if prisma_client is None:
        raise Exception(
            "No DB Connected. See - https://docs.litellm.ai/docs/proxy/virtual_keys"
        )

    # Check cache first (keyed by alias)
    cache_key = "org_alias:{}".format(org_alias)
    cached_org_obj = await user_api_key_cache.async_get_cache(
        key=cache_key,
        model_type=LiteLLM_OrganizationTable,
    )
    if cached_org_obj is not None:
        return cached_org_obj

    # Query database by organization_alias
    try:
        orgs = await prisma_client.db.litellm_organizationtable.find_many(
            where={"organization_alias": org_alias}
        )

        if not orgs:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": f"Organization with alias '{org_alias}' doesn't exist in db. Create organization via `/organization/new` call."
                },
            )

        if len(orgs) > 1:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": f"Multiple organizations found with alias '{org_alias}'. Please use org_id_jwt_field instead or ensure organization aliases are unique."
                },
            )

        org = orgs[0]
        org_obj = LiteLLM_OrganizationTable(**org.model_dump())

        # Cache the result
        await user_api_key_cache.async_set_cache(
            key=cache_key,
            value=org_obj,
            model_type=LiteLLM_OrganizationTable,
            ttl=DEFAULT_IN_MEMORY_TTL,
        )
        # Also cache by org_id for consistency
        await user_api_key_cache.async_set_cache(
            key="org_id:{}".format(org_obj.organization_id),
            value=org_obj,
            model_type=LiteLLM_OrganizationTable,
            ttl=DEFAULT_IN_MEMORY_TTL,
        )

        return org_obj

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(
            "Error looking up organization by alias: %s", org_alias
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": f"Error looking up organization by alias '{org_alias}': {str(e)}"
            },
        )


class ExperimentalUIJWTToken:
    @staticmethod
    def get_experimental_ui_login_jwt_auth_token(user_info: LiteLLM_UserTable) -> str:
        from datetime import timedelta

        from litellm.proxy.common_utils.encrypt_decrypt_utils import (
            encrypt_value_helper,
        )

        if user_info.user_role is None:
            raise Exception("User role is required for experimental UI login")

        # Experimental UI flow uses fixed 10-min expiry for security (does not use LITELLM_UI_SESSION_DURATION)
        expiration_time = get_utc_datetime() + timedelta(minutes=10)

        # Format the expiration time as ISO 8601 string
        expires = expiration_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "+00:00"

        valid_token = UserAPIKeyAuth(
            token="ui-token",
            key_name="ui-token",
            key_alias="ui-token",
            max_budget=litellm.max_ui_session_budget,
            rpm_limit=100,  # allow user to have a conversation on test key pane of UI
            expires=expires,
            user_id=user_info.user_id,
            team_id="litellm-dashboard",
            models=user_info.models,
            max_parallel_requests=None,
            user_role=LitellmUserRoles(user_info.user_role),
        )

        return encrypt_value_helper(valid_token.model_dump_json(exclude_none=True))

    @staticmethod
    def get_cli_jwt_auth_token(
        user_info: LiteLLM_UserTable,
        team_id: Optional[str] = None,
        team_alias: Optional[str] = None,
    ) -> str:
        """
        Generate a JWT token for CLI authentication with configurable expiration.

        The expiration time can be controlled via the LITELLM_CLI_JWT_EXPIRATION_HOURS
        environment variable (defaults to 24 hours).

        Args:
            user_info: User information from the database
            team_id: Team ID for the user (optional, uses user's team if available)
            team_alias: Team alias for the selected team, if available

        Returns:
            Encrypted JWT token string
        """
        from datetime import timedelta

        from litellm.proxy.common_utils.encrypt_decrypt_utils import (
            encrypt_value_helper,
        )

        if user_info.user_role is None:
            raise Exception("User role is required for CLI JWT login")

        # Calculate expiration time (configurable via LITELLM_CLI_JWT_EXPIRATION_HOURS env var)
        expiration_time = get_utc_datetime() + timedelta(hours=CLI_JWT_EXPIRATION_HOURS)

        # Format the expiration time as ISO 8601 string
        expires = expiration_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "+00:00"

        # Use provided team_id, or fall back to user's teams if available
        _team_id = team_id
        if _team_id is None and hasattr(user_info, "teams") and user_info.teams:
            # Use first team if user has teams
            _team_id = user_info.teams[0] if len(user_info.teams) > 0 else None

        valid_token = UserAPIKeyAuth(
            token=CLI_JWT_TOKEN_NAME,
            key_name=CLI_JWT_TOKEN_NAME,
            key_alias=CLI_JWT_TOKEN_NAME,
            max_budget=litellm.max_ui_session_budget,
            expires=expires,
            user_id=user_info.user_id,
            team_id=_team_id,
            team_alias=team_alias,
            models=user_info.models,
            max_parallel_requests=None,
            user_role=LitellmUserRoles(user_info.user_role),
        )

        return encrypt_value_helper(valid_token.model_dump_json(exclude_none=True))

    @staticmethod
    def get_key_object_from_ui_hash_key(
        hashed_token: str,
    ) -> Optional[UserAPIKeyAuth]:
        import json

        from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth
        from litellm.proxy.common_utils.encrypt_decrypt_utils import (
            encrypt_value_helper,
        )

        decrypted_token = decrypt_value_helper(
            hashed_token, key="ui_hash_key", exception_type="debug"
        )
        if decrypted_token is None:
            return None
        try:
            return UserAPIKeyAuth(**json.loads(decrypted_token))
        except Exception as e:
            raise Exception(
                f"Invalid hash key. Hash key={hashed_token}. Decrypted token={decrypted_token}. Error: {e}"
            )


async def _fetch_key_object_from_db_with_reconnect(
    hashed_token: str,
    prisma_client: PrismaClient,
    parent_otel_span: Optional[Span],
    proxy_logging_obj: Optional[ProxyLogging],
) -> Optional[BaseModel]:
    """
    Fetch key object from DB and retry once if a DB connection error can be healed.
    """
    try:
        return await prisma_client.get_data(
            token=hashed_token,
            table_name="combined_view",
            parent_otel_span=parent_otel_span,
            proxy_logging_obj=proxy_logging_obj,
        )
    except Exception as e:
        if PrismaDBExceptionHandler.is_database_transport_error(e):
            did_reconnect = False
            if hasattr(prisma_client, "attempt_db_reconnect"):
                auth_reconnect_timeout = getattr(
                    prisma_client, "_db_auth_reconnect_timeout_seconds", 2.0
                )
                if not isinstance(auth_reconnect_timeout, (int, float)):
                    auth_reconnect_timeout = 2.0
                auth_reconnect_lock_timeout = getattr(
                    prisma_client, "_db_auth_reconnect_lock_timeout_seconds", 0.1
                )
                if not isinstance(auth_reconnect_lock_timeout, (int, float)):
                    auth_reconnect_lock_timeout = 0.1
                did_reconnect = await prisma_client.attempt_db_reconnect(
                    reason="auth_get_key_object_lookup_failure",
                    timeout_seconds=auth_reconnect_timeout,
                    lock_timeout_seconds=auth_reconnect_lock_timeout,
                )
            if did_reconnect:
                return await prisma_client.get_data(
                    token=hashed_token,
                    table_name="combined_view",
                    parent_otel_span=parent_otel_span,
                    proxy_logging_obj=proxy_logging_obj,
                )
        raise


@log_db_metrics
async def get_jwt_key_mapping_object(
    jwt_claim_name: str,
    jwt_claim_value: str,
    prisma_client: PrismaClient,
) -> Optional[str]:
    """
    Lookup a JWT-to-virtual-key mapping from the database.

    Returns the hashed token (str) if a matching active mapping is found, else None.
    """
    mapping = await prisma_client.db.litellm_jwtkeymapping.find_first(
        where={
            "jwt_claim_name": jwt_claim_name,
            "jwt_claim_value": jwt_claim_value,
            "is_active": True,
        }
    )
    if mapping is not None:
        return mapping.token
    return None


@log_db_metrics
async def get_key_object(
    hashed_token: str,
    prisma_client: Optional[PrismaClient],
    user_api_key_cache: UserApiKeyCache,
    parent_otel_span: Optional[Span] = None,
    proxy_logging_obj: Optional[ProxyLogging] = None,
    check_cache_only: Optional[bool] = None,
) -> UserAPIKeyAuth:
    """
    - Check if team id in proxy Team Table
    - if valid, return LiteLLM_TeamTable object with defined limits
    - if not, then raise an error
    """
    if prisma_client is None:
        raise Exception(
            "No DB Connected. See - https://docs.litellm.ai/docs/proxy/virtual_keys"
        )

    # check if in cache
    key = "team_id:{}".format(team_id)
