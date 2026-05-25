import asyncio
import importlib
import json
import os
import socket
import subprocess
import sys
from unittest import mock
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import click
import httpx
import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system-path


def test_audit_log_masking():
    from datetime import datetime

    from litellm.proxy._types import LiteLLM_AuditLogs

    audit_log = LiteLLM_AuditLogs(
        id="123",
        updated_at=datetime.now(),
        changed_by="test",
        changed_by_api_key="test",
        table_name="LiteLLM_VerificationToken",
        object_id="test",
        action="updated",
        updated_values=json.dumps({"key": "sk-1234567890", "token": "1q2132r222"}),
        before_value=json.dumps({"key": "sk-1234567890", "token": "1q2132r222"}),
    )

    print(audit_log.updated_values)
    json_updated_values = json.loads(audit_log.updated_values)
    assert json_updated_values["token"] == "1q2132r222"
    assert json_updated_values["key"] == "sk-1*****7890"
    assert audit_log.before_value
    json_before_value = json.loads(audit_log.before_value)
    assert json_before_value["token"] == "1q2132r222"
    assert json_before_value["key"] == "sk-1*****7890"


def test_internal_jobs_user_has_proxy_admin_role():
    """
    Test that the internal jobs system user has PROXY_ADMIN role.

    This is critical for key rotation to work properly. The system user needs
    PROXY_ADMIN role to bypass team permission checks in
    TeamMemberPermissionChecks.can_team_member_execute_key_management_endpoint()

    Regression test for: https://github.com/BerriAI/litellm/pull/21896
    """
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth

    # Get the system user used for internal jobs like key rotation
    system_user = UserAPIKeyAuth.get_litellm_internal_jobs_user_api_key_auth()

    # Verify the system user has PROXY_ADMIN role
    assert system_user.user_role == LitellmUserRoles.PROXY_ADMIN

    # Verify other expected properties
    assert system_user.user_id == "system"
    assert system_user.team_id == "system"
    assert system_user.team_alias == "system"


def test_proxy_exception_message_propagates_to_str():
    """
    Regression test for LIT-3094.

    ProxyException must pass its message to Python's base Exception so that
    str(exception) and exception.args are populated. Otherwise downstream
    logging that does `error_message = str(original_exception)` (see
    StandardLoggingPayloadSetup.get_error_information) records an empty
    error_message for things like 401 auth failures.
    """
    from litellm.proxy._types import ProxyException

    msg = "key not allowed to access model. This key can only access models=['a']. Tried to access b"
    exc = ProxyException(message=msg, type="auth_error", param="model", code=401)

    # The fix: message must flow to the base Exception.
    assert str(exc) == msg
    assert exc.args == (msg,)

    # Existing attributes must keep working unchanged.
    assert exc.message == msg
    assert exc.type == "auth_error"
    assert exc.param == "model"
    assert exc.code == "401"
    assert exc.to_dict() == {
        "message": msg,
        "type": "auth_error",
        "param": "model",
        "code": "401",
    }


def test_proxy_exception_message_flows_into_logging_payload():
    """
    End-to-end regression for LIT-3094: the auth-failure message that the
    NVIDIA customer saw as empty `error_message` should now be populated in
    the StandardLoggingPayload error information.
    """
    from litellm.litellm_core_utils.litellm_logging import (
        StandardLoggingPayloadSetup,
    )
    from litellm.proxy._types import ProxyException

    msg = "key not allowed to access model. This key can only access models=['gpt-4o']. Tried to access gpt-5"
    exc = ProxyException(message=msg, type="auth_error", param="model", code=401)

    info = StandardLoggingPayloadSetup.get_error_information(exc)

    assert info["error_message"] == msg
    assert info["error_code"] == "401"
    assert info["error_class"] == "ProxyException"
