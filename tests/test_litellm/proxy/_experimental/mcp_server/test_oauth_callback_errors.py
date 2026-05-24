"""LIT-2750 — /callback handles IdP error responses + missing ``code``
without falling through to FastAPI's Pydantic 422.

Before this fix, the callback endpoint signature was
``async def callback(request: Request, code: str, state: str)``, so the
OAuth-error envelope path (RFC 6749 §4.1.2.1) — ``?error=access_denied&...``
with no ``code`` — and the SSO-redirect-chain-dropped-query case both
short-circuited to a Pydantic 422 JSON response. The user saw an opaque
error page and the MCP client waited on its loopback redirect until it
timed out.

These tests live in a dedicated file (rather than appended to
``test_discoverable_endpoints.py``) so the fix is reviewable in
isolation.
"""

from unittest.mock import MagicMock, patch

import pytest


# Fixture to mock IP address check for all MCP tests. Mirrors the
# autouse fixture in test_discoverable_endpoints.py so the callback
# handler's call to ``IPAddressUtils.get_mcp_client_ip`` doesn't
# 404 us out on the same-origin redirect-URI validator.
@pytest.fixture(autouse=True)
def _mock_mcp_client_ip():
    with patch(
        "litellm.proxy._experimental.mcp_server.discoverable_endpoints.IPAddressUtils.get_mcp_client_ip",
        return_value=None,
    ):
        yield


def _mock_callback_request(base_url: str = "http://localhost:3000/"):
    """Return a MagicMock Request usable by the callback handler.

    The handler only reads ``request.base_url`` and ``request.headers``
    through ``get_request_base_url``, so a plain MagicMock with these
    attributes is sufficient.
    """
    req = MagicMock()
    req.base_url = base_url
    req.headers = {}
    return req


@pytest.mark.asyncio
async def test_callback_propagates_idp_error_to_client_redirect_uri():
    """IdP returned ``?error=access_denied&...&state=<encrypted>`` — the
    callback must 302 the user back to the client's registered
    redirect_uri with the OAuth error params attached (RFC 6749
    §4.1.2.1), NOT 422.
    """
    from urllib.parse import parse_qs, urlparse

    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
        callback,
    )

    with patch(
        "litellm.proxy._experimental.mcp_server.discoverable_endpoints.decode_state_hash"
    ) as mock_decode:
        mock_decode.return_value = {
            "base_url": "http://127.0.0.1:60108/callback",
            "original_state": "client-csrf-xyz",
            "code_challenge": None,
            "code_challenge_method": None,
            "client_redirect_uri": "http://127.0.0.1:60108/callback",
        }

        response = await callback(
            request=_mock_callback_request(),
            code=None,
            state="encrypted_state_value",
            error="access_denied",
            error_description="The user cancelled the sign-in dialog",
            error_uri="https://idp.example.com/errors/access_denied",
        )

    assert response.status_code == 302
    parsed = urlparse(response.headers["location"])
    assert parsed.scheme == "http"
    assert parsed.netloc == "127.0.0.1:60108"
    assert parsed.path == "/callback"

    qs = parse_qs(parsed.query)
    assert qs["error"] == ["access_denied"]
    assert qs["error_description"] == ["The user cancelled the sign-in dialog"]
    assert qs["error_uri"] == ["https://idp.example.com/errors/access_denied"]
    assert qs["state"] == ["client-csrf-xyz"]
    # The error path must NOT leak a code or pretend the auth succeeded.
    assert "code" not in qs


@pytest.mark.asyncio
async def test_callback_missing_code_treated_as_invalid_request():
    """SSO chains sometimes strip the IdP's query params and re-hit
    ``/callback`` with nothing but ``state``. The previous behaviour
    returned a Pydantic 422 blob; the new behaviour normalizes to
    ``error=invalid_request`` and propagates that to the client so the
    MCP client surfaces a real error instead of timing out.
    """
    from urllib.parse import parse_qs, urlparse

    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
        callback,
    )

    with patch(
        "litellm.proxy._experimental.mcp_server.discoverable_endpoints.decode_state_hash"
    ) as mock_decode:
        mock_decode.return_value = {
            "base_url": "http://127.0.0.1:60108/callback",
            "original_state": "csrf-1",
            "code_challenge": None,
            "code_challenge_method": None,
            "client_redirect_uri": "http://127.0.0.1:60108/callback",
        }

        response = await callback(
            request=_mock_callback_request(),
            code=None,
            state="encrypted_state_value",
        )

    assert response.status_code == 302
    parsed = urlparse(response.headers["location"])
    qs = parse_qs(parsed.query)
    assert qs["error"] == ["invalid_request"]
    assert qs["state"] == ["csrf-1"]
    assert "error_description" in qs


@pytest.mark.asyncio
async def test_callback_idp_error_without_state_renders_html_400():
    """If the IdP loses the state too (broken SSO chains that strip every
    query param), we can't redirect anywhere. The callback must render a
    friendly HTML page with the error code instead of 422.

    Also verifies that ``error_description`` — which is attacker-influenced
    (the IdP echoes back whatever it wants) — is HTML-escaped to defeat
    reflected XSS via the error page.
    """
    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
        callback,
    )

    response = await callback(
        request=_mock_callback_request(),
        code=None,
        state=None,
        error="server_error",
        error_description="Upstream IdP exploded <script>alert(1)</script>",
    )

    assert response.status_code == 400
    body = response.body.decode()
    assert "server_error" in body
    assert "<script>" not in body
    assert "&lt;script&gt;" in body


@pytest.mark.asyncio
async def test_callback_idp_error_with_undecryptable_state_renders_html_400():
    """State present but undecryptable (signing key rotated, malformed
    blob, attacker-crafted). We must not 422; render the same friendly
    HTML page so the user knows what happened.
    """
    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
        callback,
    )

    with patch(
        "litellm.proxy._experimental.mcp_server.discoverable_endpoints.decode_state_hash"
    ) as mock_decode:
        mock_decode.side_effect = ValueError("bad state")

        response = await callback(
            request=_mock_callback_request(),
            code=None,
            state="garbage_state_value",
            error="access_denied",
            error_description="user said no",
        )

    assert response.status_code == 400
    body = response.body.decode()
    assert "access_denied" in body
    assert "user said no" in body


@pytest.mark.asyncio
async def test_callback_error_path_does_not_open_redirect_to_attacker():
    """A party with the proxy's state-signing key (or a forged loopback
    flow) could mint a state whose ``client_redirect_uri`` points at
    ``attacker.example.com`` and then invoke
    ``/callback?error=...&state=<minted>`` to bounce the victim's browser.

    The same re-validation that protects the success path must also
    protect the error path (VERIA-57): when redirect-URI validation
    fails, fall through to the HTML page — NOT a 302 to the attacker.
    """
    from fastapi import HTTPException

    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
        callback,
    )

    with patch(
        "litellm.proxy._experimental.mcp_server.discoverable_endpoints.decode_state_hash"
    ) as mock_decode, patch(
        "litellm.proxy._experimental.mcp_server.discoverable_endpoints"
        "._get_validated_client_redirect_uri"
    ) as mock_validate:
        mock_decode.return_value = {
            "base_url": "https://attacker.example.com/steal",
            "original_state": "x",
            "client_redirect_uri": "https://attacker.example.com/steal",
        }
        mock_validate.side_effect = HTTPException(
            status_code=400, detail="untrusted redirect_uri"
        )

        response = await callback(
            request=_mock_callback_request(),
            code=None,
            state="hostile_state",
            error="access_denied",
        )

    # Must NOT have 302-ed to the attacker.
    assert response.status_code == 400
    body = response.body.decode()
    assert "attacker.example.com" not in body


@pytest.mark.asyncio
async def test_callback_code_without_state_renders_html_400():
    """If ``code`` arrives but ``state`` does not, we can't tell which
    client to redirect back to (and CSRF protection would be broken
    anyway). Render the HTML page instead of 422.
    """
    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
        callback,
    )

    response = await callback(
        request=_mock_callback_request(),
        code="some_auth_code",
        state=None,
    )

    assert response.status_code == 400
    body = response.body.decode()
    assert "invalid_request" in body
