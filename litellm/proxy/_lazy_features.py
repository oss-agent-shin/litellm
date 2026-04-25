"""
Lazy registration for optional feature routers on the proxy app.

Each entry in `LAZY_FEATURES` describes a feature whose router module is
imported only when the first request lands on one of its path prefixes.
Until that happens, the module's Pydantic schemas, FastAPI Dependants, and
TypedDict metaclasses stay out of the process — saving ~700 MB on a typical
deployment that uses none of these features.

Tradeoffs vs. eager registration:
    * First request to a lazy feature pays the import + schema-compile cost
      (typically 1-3 s for heavy modules like MCP).
    * `/openapi.json` is shrunken until each feature is warmed: routes don't
      appear in the spec until their module loads. UI form-builder code that
      introspects schemas via `/openapi.json` (see check_openapi_schema.tsx)
      will see empty schemas until the feature has been hit at least once.
      Operators who need a complete spec post-deploy can warm by issuing a
      throwaway request to each feature prefix.
"""

import asyncio
import importlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Tuple

from starlette.types import Receive, Scope, Send

from litellm._logging import verbose_proxy_logger

if TYPE_CHECKING:
    from fastapi import FastAPI


def _include_router(attr_name: str = "router") -> Callable[["FastAPI", object], None]:
    """Default register_fn factory: include_router(getattr(module, attr_name))."""

    def _register(app: "FastAPI", module: object) -> None:
        router = getattr(module, attr_name)
        app.include_router(router)

    return _register


def _mount_app(
    prefix: str, attr_name: str = "app"
) -> Callable[["FastAPI", object], None]:
    """register_fn for ASGI sub-app mounts (e.g. mcp_app at BASE_MCP_ROUTE)."""

    def _register(app: "FastAPI", module: object) -> None:
        sub_app = getattr(module, attr_name)
        app.mount(path=prefix, app=sub_app)

    return _register


@dataclass(frozen=True)
class LazyFeature:
    """One lazy-loadable feature."""

    name: str
    module_path: str
    path_prefixes: Tuple[str, ...]
    register_fn: Callable[["FastAPI", object], None] = field(
        default_factory=lambda: _include_router("router")
    )


# Order matters only insofar as `path_prefixes` must be specific enough that
# unrelated requests don't trigger the wrong feature. When in doubt, use the
# most-specific prefix(es) — e.g. /v1/mcp/server before /v1/mcp.
LAZY_FEATURES: Tuple[LazyFeature, ...] = (
    LazyFeature(
        name="guardrails",
        module_path="litellm.proxy.guardrails.guardrail_endpoints",
        path_prefixes=("/guardrails",),
    ),
    LazyFeature(
        name="policies",
        module_path="litellm.proxy.management_endpoints.policy_endpoints",
        # Trailing slash so this doesn't also match `/policies/...` paths,
        # which belong to policy_engine below.
        path_prefixes=("/policy/",),
    ),
    LazyFeature(
        name="policy_engine",
        module_path="litellm.proxy.policy_engine.policy_endpoints",
        path_prefixes=("/policies",),
    ),
    LazyFeature(
        name="policy_resolve",
        module_path="litellm.proxy.policy_engine.policy_resolve_endpoints",
        path_prefixes=("/policies/resolve", "/policies/attachments/estimate-impact"),
    ),
    LazyFeature(
        name="agents",
        module_path="litellm.proxy.agent_endpoints.endpoints",
        path_prefixes=("/v1/agents", "/agents"),
    ),
    LazyFeature(
        name="a2a",
        module_path="litellm.proxy.agent_endpoints.a2a_endpoints",
        path_prefixes=("/a2a",),
    ),
    LazyFeature(
        name="vector_stores",
        module_path="litellm.proxy.vector_store_endpoints.endpoints",
        path_prefixes=("/v1/vector_stores", "/vector_stores"),
    ),
    LazyFeature(
        name="vector_store_management",
        module_path="litellm.proxy.vector_store_endpoints.management_endpoints",
        # Trailing slash so this doesn't match `/vector_stores/...` paths,
        # which belong to vector_stores above.
        path_prefixes=("/vector_store/",),
    ),
    LazyFeature(
        name="vector_store_files",
        module_path="litellm.proxy.vector_store_files_endpoints.endpoints",
        # The router exposes both /v1/vector_stores/{id}/files and the
        # un-versioned /vector_stores/{id}/files form, so we must match both
        # prefixes here. Both are also listed in vector_stores above; the
        # middleware loads every feature that matches, so all needed routes
        # register on the first hit.
        path_prefixes=("/v1/vector_stores", "/vector_stores"),
    ),
    LazyFeature(
        name="tools",
        module_path="litellm.proxy.management_endpoints.tool_management_endpoints",
        path_prefixes=("/v1/tool", "/tool"),
    ),
    LazyFeature(
        name="search_tools",
        module_path="litellm.proxy.search_endpoints.search_tool_management",
        path_prefixes=("/search_tools",),
    ),
    # MCP — the heavy chain. Mount the ASGI sub-app at BASE_MCP_ROUTE first
    # since /v1/mcp/server requests should hit the management router (a normal
    # APIRouter), not the mounted ASGI app.
    LazyFeature(
        name="mcp_management",
        module_path="litellm.proxy.management_endpoints.mcp_management_endpoints",
        path_prefixes=("/v1/mcp/server",),
    ),
    LazyFeature(
        name="mcp_byok_oauth",
        module_path="litellm.proxy._experimental.mcp_server.byok_oauth_endpoints",
        path_prefixes=("/v1/mcp/oauth", "/mcp/oauth"),
    ),
    LazyFeature(
        name="mcp_discoverable",
        module_path="litellm.proxy._experimental.mcp_server.discoverable_endpoints",
        path_prefixes=("/v1/mcp/discoverable", "/mcp/discoverable"),
    ),
    LazyFeature(
        name="mcp_rest",
        module_path="litellm.proxy._experimental.mcp_server.rest_endpoints",
        path_prefixes=("/v1/mcp/tools",),
    ),
    LazyFeature(
        name="mcp_app",
        module_path="litellm.proxy._experimental.mcp_server.server",
        # BASE_MCP_ROUTE is "/mcp" today; importing it here would defeat lazy
        # loading. Match on /mcp prefix and let the mount target use whatever
        # the constant resolves to at load time.
        path_prefixes=("/mcp",),
        register_fn=_mount_app("/mcp", attr_name="app"),
    ),
    LazyFeature(
        name="config_overrides",
        module_path="litellm.proxy.management_endpoints.config_override_endpoints",
        path_prefixes=("/config_overrides",),
    ),
    LazyFeature(
        name="realtime",
        module_path="litellm.proxy.realtime_endpoints.endpoints",
        path_prefixes=("/openai/v1/realtime", "/realtime"),
    ),
    LazyFeature(
        name="anthropic_passthrough",
        module_path="litellm.proxy.anthropic_endpoints.endpoints",
        path_prefixes=("/v1/messages", "/anthropic"),
    ),
    LazyFeature(
        name="anthropic_skills",
        module_path="litellm.proxy.anthropic_endpoints.skills_endpoints",
        path_prefixes=("/v1/skills", "/skills"),
    ),
    LazyFeature(
        name="langfuse_passthrough",
        module_path="litellm.proxy.vertex_ai_endpoints.langfuse_endpoints",
        path_prefixes=("/langfuse",),
    ),
    LazyFeature(
        name="evals",
        module_path="litellm.proxy.openai_evals_endpoints.endpoints",
        path_prefixes=("/v1/evals", "/evals"),
    ),
    LazyFeature(
        name="claude_code_marketplace",
        module_path="litellm.proxy.anthropic_endpoints.claude_code_endpoints",
        path_prefixes=("/claude-code",),
        register_fn=_include_router("claude_code_marketplace_router"),
    ),
    LazyFeature(
        name="scim",
        module_path="litellm.proxy.management_endpoints.scim.scim_v2",
        path_prefixes=("/scim",),
        register_fn=_include_router("scim_router"),
    ),
    LazyFeature(
        name="cloudzero",
        module_path="litellm.proxy.spend_tracking.cloudzero_endpoints",
        path_prefixes=("/cloudzero",),
    ),
    LazyFeature(
        name="vantage",
        module_path="litellm.proxy.spend_tracking.vantage_endpoints",
        path_prefixes=("/vantage",),
    ),
    LazyFeature(
        name="usage_ai",
        module_path="litellm.proxy.management_endpoints.usage_endpoints",
        path_prefixes=("/usage/ai",),
    ),
    LazyFeature(
        name="prompts",
        module_path="litellm.proxy.prompts.prompt_endpoints",
        path_prefixes=("/prompts",),
    ),
    LazyFeature(
        name="jwt_mappings",
        module_path="litellm.proxy.management_endpoints.jwt_key_mapping_endpoints",
        path_prefixes=("/jwt/key/mapping",),
    ),
    LazyFeature(
        name="compliance",
        module_path="litellm.proxy.management_endpoints.compliance_endpoints",
        path_prefixes=("/compliance",),
    ),
    LazyFeature(
        name="access_groups",
        module_path="litellm.proxy.management_endpoints.access_group_endpoints",
        path_prefixes=("/access_group", "/v1/access_group", "/v1/unified_access_group"),
    ),
)


class LazyFeatureMiddleware:
    """
    ASGI middleware that lazy-imports + registers feature routers on first
    matching request. Idempotent — once a feature is loaded, subsequent
    requests fall through to the normal router dispatch with no overhead.
    """

    def __init__(
        self,
        app,
        fastapi_app: "FastAPI",
        features: Tuple[LazyFeature, ...] = LAZY_FEATURES,
    ):
        self.app = app
        self._fastapi_app = fastapi_app
        self._features = features
        self._loaded: set = set()
        # Per-feature locks: independent features can load in parallel.
        self._locks: dict = {}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            path = scope.get("path", "")
            for feat in self._features:
                if feat.module_path in self._loaded:
                    continue
                if any(path.startswith(p) for p in feat.path_prefixes):
                    await self._load(feat)
        await self.app(scope, receive, send)

    async def _load(self, feat: LazyFeature) -> None:
        lock = self._locks.setdefault(feat.module_path, asyncio.Lock())
        async with lock:
            if feat.module_path in self._loaded:
                return
            try:
                # Off-load the import to a thread so we don't block the event
                # loop — heavy modules (e.g. MCP) take 1-3 s. register_fn must
                # stay on the loop thread since it mutates app.router.routes.
                loop = asyncio.get_running_loop()
                module = await loop.run_in_executor(
                    None, importlib.import_module, feat.module_path
                )
                feat.register_fn(self._fastapi_app, module)
                self._loaded.add(feat.module_path)
                # Invalidate the cached OpenAPI schema so the next /openapi.json
                # fetch reflects the newly registered routes.
                self._fastapi_app.openapi_schema = None
                verbose_proxy_logger.info(
                    "Lazy-loaded optional feature %r (module: %s)",
                    feat.name,
                    feat.module_path,
                )
            except Exception as exc:
                # Mark loaded anyway to avoid repeated failed imports per request.
                self._loaded.add(feat.module_path)
                verbose_proxy_logger.warning(
                    "Failed to lazy-load optional feature %r (module: %s): %s. "
                    "This feature's endpoints will return 404 until restart.",
                    feat.name,
                    feat.module_path,
                    exc,
                )


def attach_lazy_features(app: "FastAPI") -> None:
    """
    Attach the lazy-feature middleware to a FastAPI app. Called once at the
    end of proxy_server.py module load, after the always-on routers are
    registered.
    """
    app.add_middleware(LazyFeatureMiddleware, fastapi_app=app)
