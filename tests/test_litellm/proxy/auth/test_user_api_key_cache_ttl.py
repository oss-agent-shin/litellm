"""
Tests for LIT-3338: ``general_settings.user_api_key_cache_ttl`` must be honored
when ``user_api_key_auth`` caches management objects.

Before the fix, every management-object cache write in
``litellm/proxy/auth/auth_checks.py`` passed
``ttl=DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL`` explicitly, which
overrode the configured ``default_in_memory_ttl`` on every write -- so the
in-memory cache was pinned to 60s regardless of ``general_settings``.

These tests cover both the helper resolver (``_resolve_management_object_ttl``)
and the actual cache-write path exercised by ``_cache_key_object`` /
``_cache_management_object``.
"""
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.constants import DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL
from litellm.proxy._types import LiteLLM_UserTable, UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import (
    _cache_key_object,
    _cache_management_object,
    _resolve_management_object_ttl,
)
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache


def _make_cache(default_in_memory_ttl=60):
    return UserApiKeyCache(default_in_memory_ttl=default_in_memory_ttl)


def _entry_ttl(cache: UserApiKeyCache, key: str) -> float:
    """Return the effective in-memory TTL (seconds) the cache wrote for ``key``."""
    im = cache.in_memory_cache
    expires_at = im.ttl_dict[key]
    return expires_at - time.time()


# ----- resolver -----


def test_resolve_uses_cache_default_when_configured():
    """When user sets ``user_api_key_cache_ttl``, proxy_server.load_config
    propagates it onto ``user_api_key_cache.default_in_memory_ttl`` via
    ``update_cache_ttl``. The resolver must return that value."""
    cache = _make_cache(default_in_memory_ttl=60)
    cache.update_cache_ttl(default_in_memory_ttl=300.0, default_redis_ttl=300.0)
    assert _resolve_management_object_ttl(cache) == 300.0


def test_resolve_uses_cache_default_when_unconfigured():
    """No explicit user override -> cache.default_in_memory_ttl is still the
    constructor value (60s by default). Resolver returns it."""
    cache = _make_cache(default_in_memory_ttl=60)
    assert _resolve_management_object_ttl(cache) == 60.0


def test_resolve_falls_back_to_constant_when_cache_default_is_none():
    """Defensive: if some caller hands us a cache with no in-memory ttl set,
    fall back to the env-overridable constant rather than passing None."""
    cache = _make_cache(default_in_memory_ttl=60)
    cache.default_in_memory_ttl = None
    assert (
        _resolve_management_object_ttl(cache)
        == float(DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL)
    )


# ----- end-to-end: _cache_management_object writes the resolved ttl -----


@pytest.mark.asyncio
async def test_cache_management_object_honors_configured_ttl():
    """LIT-3338 regression: configured ttl=300 must reach the in-memory cache."""
    cache = _make_cache(default_in_memory_ttl=60)
    cache.update_cache_ttl(default_in_memory_ttl=300.0, default_redis_ttl=300.0)

    await _cache_management_object(
        key="user_id:abc",
        value=LiteLLM_UserTable(user_id="abc"),
        user_api_key_cache=cache,
        proxy_logging_obj=None,
        model_type=LiteLLM_UserTable,
    )

    ttl = _entry_ttl(cache, "user_id:abc")
    # Allow ~5s of clock slack; what matters is it's not pinned to 60.
    assert 290.0 <= ttl <= 305.0, f"expected ~300s, got {ttl:.1f}s"


@pytest.mark.asyncio
async def test_cache_management_object_default_is_unchanged():
    """No user override -> cache writes still use ~60s (no regression)."""
    cache = _make_cache(default_in_memory_ttl=60)
    await _cache_management_object(
        key="user_id:def",
        value=LiteLLM_UserTable(user_id="def"),
        user_api_key_cache=cache,
        proxy_logging_obj=None,
        model_type=LiteLLM_UserTable,
    )
    ttl = _entry_ttl(cache, "user_id:def")
    assert 55.0 <= ttl <= 65.0, f"expected ~60s, got {ttl:.1f}s"


# ----- reported call path: _cache_key_object (the API-key auth pin) -----


@pytest.mark.asyncio
async def test_cache_key_object_honors_configured_ttl():
    """End-to-end reproduction of the reported bug: get_key_object ->
    _cache_key_object -> _cache_management_object must write a 300s entry
    when ``general_settings.user_api_key_cache_ttl: 300``."""
    cache = _make_cache(default_in_memory_ttl=60)
    cache.update_cache_ttl(default_in_memory_ttl=300.0, default_redis_ttl=300.0)

    hashed_token = "sk-hash-LIT-3338"
    await _cache_key_object(
        hashed_token=hashed_token,
        user_api_key_obj=UserAPIKeyAuth(token=hashed_token),
        user_api_key_cache=cache,
        proxy_logging_obj=None,
    )

    ttl = _entry_ttl(cache, hashed_token)
    assert 290.0 <= ttl <= 305.0, f"expected ~300s, got {ttl:.1f}s"
