import enum
import json
import os
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Union

import httpx
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    Json,
    field_validator,
    model_validator,
)
from typing_extensions import Required, TypedDict

from litellm._uuid import uuid
from litellm.constants import MCP_STDIO_ALLOWED_COMMANDS
from litellm.litellm_core_utils.initialize_dynamic_callback_params import (
    validate_no_callback_env_reference,
)
from litellm.types.integrations.slack_alerting import AlertType
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIFileObject,
    ResponsesAPIResponse,
)
from litellm.types.mcp import (
    MCPAuthType,
    MCPCredentials,
    MCPTransport,
    MCPTransportType,
)
from litellm.types.mcp_server.mcp_server_manager import MCPInfo
from litellm.types.router import RouterErrors, UpdateRouterConfig
from litellm.types.secret_managers.main import KeyManagementSystem
from litellm.types.utils import (
    CallTypes,
    CostBreakdown,
    EmbeddingResponse,
    GenericBudgetConfigType,
    ImageResponse,
    LiteLLMBatch,
    LiteLLMFineTuningJob,
    LiteLLMPydanticObjectBase,
    ModelResponse,
    ProviderField,
    StandardCallbackDynamicParams,
    StandardLoggingGuardrailInformation,
    StandardLoggingMCPToolCall,
    StandardLoggingModelInformation,
    StandardLoggingPayloadErrorInformation,
    StandardLoggingPayloadStatus,
    StandardLoggingVectorStoreRequest,
    StandardPassThroughResponseObject,
    TextCompletionResponse,
)
from litellm.types.videos.main import VideoObject

from .types_utils.utils import get_instance_fn, validate_custom_validate_return_type

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    Span = Union[_Span, Any]
else:
    Span = Any


class SupportedDBObjectType(str, enum.Enum):
    """
    Supported database object types for fine-grained DB storage control.
    Use in general_settings.supported_db_objects to specify which objects to load from DB.
    """

    MODELS = "models"
    MCP = "mcp"
    GUARDRAILS = "guardrails"
    POLICIES = "policies"
    VECTOR_STORES = "vector_stores"
    PASS_THROUGH_ENDPOINTS = "pass_through_endpoints"
    PROMPTS = "prompts"
    MODEL_COST_MAP = "model_cost_map"
    TOOLS = "tools"
    CONFIG_OVERRIDES = "config_overrides"

    def __str__(self):
        return str(self.value)


class LiteLLMTeamRoles(enum.Enum):
    # team admin
    TEAM_ADMIN = "admin"
    # team member
    TEAM_MEMBER = "user"


class LitellmUserRoles(str, enum.Enum):
    """
    Admin Roles:
    PROXY_ADMIN: admin over the platform
    PROXY_ADMIN_VIEW_ONLY: can login, view all own keys, view all spend
    ORG_ADMIN: admin over a specific organization, can create teams, users only within their organization

    Internal User Roles:
    INTERNAL_USER: can login, view/create/delete their own keys, view their spend
    INTERNAL_USER_VIEW_ONLY: can login, view their own keys, view their own spend


    Team Roles:
    TEAM: used for JWT auth


    Customer Roles:
    CUSTOMER: External users -> these are customers

    """

    # Admin Roles
    PROXY_ADMIN = "proxy_admin"
    PROXY_ADMIN_VIEW_ONLY = "proxy_admin_viewer"

    # Organization admins
    ORG_ADMIN = "org_admin"

    # Internal User Roles
    INTERNAL_USER = "internal_user"
    INTERNAL_USER_VIEW_ONLY = "internal_user_viewer"

    # Team Roles
    TEAM = "team"

    # Customer Roles - External users of proxy
    CUSTOMER = "customer"

    def __str__(self):
        return str(self.value)

    def values(self) -> List[str]:
        return list(self.__annotations__.keys())

    @property
    def description(self):
        """
        Descriptions for the enum values
        """
        descriptions = {
            "proxy_admin": "admin over litellm proxy, has all permissions",
            "proxy_admin_viewer": "view all keys, view all spend",
            "internal_user": "view/create/delete their own keys, view their own spend",
            "internal_user_viewer": "view their own keys, view their own spend",
            "team": "team scope used for JWT auth",
            "customer": "customer",
        }
        return descriptions.get(self.value, "")

    @property
    def ui_label(self):
        """
        UI labels for the enum values
        """
        ui_labels = {
            "proxy_admin": "Admin (All Permissions)",
            "proxy_admin_viewer": "Admin (View Only)",
            "internal_user": "Internal User (Create/Delete/View)",
            "internal_user_viewer": "Internal User (View Only)",
            "team": "Team",
            "customer": "Customer",
        }
        return ui_labels.get(self.value, "")

    @property
    def is_internal_user_role(self) -> bool:
        """returns true if this role is an `internal_user` or `internal_user_viewer` role"""
        return self.value in [
            self.INTERNAL_USER,
            self.INTERNAL_USER_VIEW_ONLY,
        ]


class LitellmTableNames(str, enum.Enum):
    """
    Enum for Table Names used by LiteLLM
    """

    TEAM_TABLE_NAME = "LiteLLM_TeamTable"
    USER_TABLE_NAME = "LiteLLM_UserTable"
    KEY_TABLE_NAME = "LiteLLM_VerificationToken"
    PROXY_MODEL_TABLE_NAME = "LiteLLM_ProxyModelTable"
    MANAGED_FILE_TABLE_NAME = "LiteLLM_ManagedFileTable"
    TOOL_TABLE_NAME = "LiteLLM_ToolTable"
    CACHE_CONFIG_TABLE_NAME = "LiteLLM_CacheConfig"
    CONFIG_OVERRIDES_TABLE_NAME = "LiteLLM_ConfigOverrides"


class Litellm_EntityType(enum.Enum):
    """
    Enum for types of entities on litellm

    This enum allows specifying the type of entity that is being tracked in the database.
    """

    KEY = "key"
    USER = "user"
    END_USER = "end_user"
    TEAM = "team"
    TEAM_MEMBER = "team_member"
    ORGANIZATION = "organization"
    PROJECT = "project"
    TAG = "tag"
    AGENT = "agent"

    # global proxy level entity
    PROXY = "proxy"


def hash_token(token: str):
    import hashlib

    # Hash the string using SHA-256
    hashed_token = hashlib.sha256(token.encode()).hexdigest()

    return hashed_token


TRUNCATED_PLACEHOLDER_DO_NOT_USE = True
