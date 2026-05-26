"""
Tests for LIT-3338: ``general_settings.user_api_key_cache_ttl`` must be honored
when ``user_api_key_auth`` caches management objects.

The bug: every management-object cache write in
``litellm/proxy/auth/auth_checks.py`` calls
``user_api_key_cache.async_set_cache(..., ttl=DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL)``.
``DualCache.async_set_cache`` then forwards that explicit kwarg verbatim,
overriding ``self.default_in_memory_ttl`` (the value
``proxy_server.load_config`` seeds from ``general_settings.user_api_key_cache_ttl``
via ``DualCache.update_cache_ttl``). So the operator-configured TTL was
silently ignored on every API-key auth pin.

The fix lives in ``UserApiKeyCache._honor_user_api_key_cache_ttl`` and is called
at the top of both ``set_cache`` and ``async_set_cache``.
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
)
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache


def _make_cache(default_in_memory_ttl=60):
    return UserApiKeyCache(default_in_memory_ttl=default_in_memory_ttl)


def _entry_ttl(cache: UserApiKeyCache, key: str) -> float:
    """Effective in-memory TTL (seconds) the cache wrote for ``key``."""
    im = cache.in_memory_cache
    expires_at = im.ttl_dict[key]
    return expires_at - time.time()


# ----- _honor_user_api_key_cache_ttl: the swap helper -----


def test_honor_swaps_sentinel_to_configured_default():
    """When the caller passed ``ttl=DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL``
    and the cache's ``default_in_memory_ttl`` has been overridden (e.g. by
    ``general_settings.user_api_key_cache_ttl``), swap the explicit kwarg for
    the configured default."""
    cache = _make_cache(default_in_memory_ttl=60)
    cache.update_cache_ttl(default_in_memory_ttl=300.0, default_redis_ttl=300.0)
    kwargs = {"ttl": DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL}
    cache._honor_user_api_key_cache_ttl(kwargs)
    assert kwargs["ttl"] == 300.0


def test_honor_noop_when_configured_default_matches_sentinel():
    """If the operator left ``user_api_key_cache_ttl`` at the default 60s, the
    cache default equals the sentinel and there's nothing to swap."""
    cache = _make_cache(default_in_memory_ttl=60)
    # default_in_memory_ttl is already 60 -> equals sentinel -> no swap
    kwargs = {"ttl": DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL}
    cache._honor_user_api_key_cache_ttl(kwargs)
    assert kwargs["ttl"] == DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL


def test_honor_does_not_swap_other_explicit_ttls():
    """Callers that pass a TTL that is NOT the management-object sentinel must
    keep their value, even when the operator has configured a different default."""
    cache = _make_cache(default_in_memory_ttl=60)
    cache.update_cache_ttl(default_in_memory_ttl=300.0, default_redis_ttl=300.0)
    kwargs = {"ttl": 5}
    cache._honor_user_api_key_cache_ttl(kwargs)
    assert kwargs["ttl"] == 5


def test_honor_noop_when_no_explicit_ttl():
    """No ttl kwarg -> nothing to swap. DualCache will fill it from
    default_in_memory_ttl on its own."""
    cache = _make_cache(default_in_memory_ttl=60)
    cache.update_cache_ttl(default_in_memory_ttl=300.0, default_redis_ttl=300.0)
    kwargs = {}
    cache._honor_user_api_key_cache_ttl(kwargs)
    assert "ttl" not in kwargs


def test_honor_noop_when_cache_default_is_none():
    """Defensive: if ``default_in_memory_ttl`` was somehow cleared, leave the
    explicit kwarg alone (the constant is still a sensible fallback)."""
    cache = _make_cache(default_in_memory_ttl=60)
    cache.default_in_memory_ttl = None
    kwargs = {"ttl": DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL}
    cache._honor_user_api_key_cache_ttl(kwargs)
    assert kwargs["ttl"] == DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL


# ----- end-to-end: the reported get_key_object -> _cache_key_object path -----


@pytest.mark.asyncio
async def test_cache_management_object_honors_configured_ttl():
    """LIT-3338 regression: configured ttl=300 must reach the in-memory cache
    even when the caller explicitly passes ``DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL``."""
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
    assert 290.0 <= ttl <= 305.0, f"expected ~300s, got {ttl:.1f}s"


@pytest.mark.asyncio
async def test_cache_management_object_default_is_unchanged():
    """No operator override -> cache writes still use ~60s (no regression)."""
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


@pytest.mark.asyncio
async def test_cache_key_object_honors_configured_ttl():
    """End-to-end reproduction: get_key_object -> _cache_key_object ->
    _cache_management_object must write a 300s entry when
    ``general_settings.user_api_key_cache_ttl: 300``."""
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
