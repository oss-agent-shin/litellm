import os
import sys
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("../../.."))

import litellm.proxy.proxy_server as ps
import litellm.proxy.utils as proxy_utils
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.proxy_server import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def mock_auth():
    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="test-user"
    )
    yield
    app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.fixture(autouse=True)
def model_list_state(monkeypatch):
    mock_router = AsyncMock()
    monkeypatch.setattr(ps, "llm_router", mock_router)
    monkeypatch.setattr(ps, "general_settings", {})
    monkeypatch.setattr(ps, "prisma_client", None)
    monkeypatch.setattr(ps, "proxy_logging_obj", None)
    monkeypatch.setattr(ps, "user_api_key_cache", None)
    monkeypatch.setattr(ps, "user_model", None)
    monkeypatch.setattr(
        proxy_utils,
        "get_available_models_for_user",
        AsyncMock(return_value=["claude-opus-4-6", "gpt-4o"]),
    )


def test_v1_models_returns_anthropic_format_for_claude_code_user_agent(client):
    response = client.get(
        "/v1/models",
        headers={
            "Authorization": "Bearer sk-test",
            "User-Agent": "claude-code/2.1.128",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["has_more"] is False
    assert body["first_id"] == "claude-opus-4-6"
    assert body["last_id"] == "gpt-4o"
    assert "object" not in body
    assert body["data"][0] == {
        "type": "model",
        "id": "claude-opus-4-6",
        "display_name": "Claude Opus 4.6",
        "created_at": "2023-02-28T18:56:42Z",
    }


def test_v1_models_returns_anthropic_format_for_anthropic_version_header(client):
    response = client.get(
        "/v1/models",
        headers={
            "Authorization": "Bearer sk-test",
            "anthropic-version": "2023-06-01",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["has_more"] is False
    assert body["data"][0]["type"] == "model"
    assert body["data"][0]["created_at"].endswith("Z")


def test_v1_models_keeps_openai_format_by_default(client):
    response = client.get(
        "/v1/models",
        headers={"Authorization": "Bearer sk-test"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert "has_more" not in body
    assert body["data"][0]["object"] == "model"
    assert "display_name" not in body["data"][0]
