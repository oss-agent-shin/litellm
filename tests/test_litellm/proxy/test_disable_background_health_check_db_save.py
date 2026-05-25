"""Unit tests for the `general_settings.disable_background_health_check_db_save` opt-out.

Bug: LIT-3078 -- `_schedule_background_health_check_db_save` runs after every
background health check cycle. On clusters with many models this drives a
`get_all_latest_health_checks()` SELECT + per-model INSERTs into
`LiteLLM_HealthCheckTable` every cycle, which can push Aurora ACU to 100%.

Fix: read the new `general_settings.disable_background_health_check_db_save`
flag (default False to preserve current behavior). When True, the scheduler
helper short-circuits before importing or scheduling the DB save task, so
both the SELECT-for-diff and the INSERTs are skipped from the background
loop. Explicit `GET /health` writes are unaffected.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.proxy import proxy_server


def _make_prisma():
    """Cheap stand-in for a prisma client. Only identity is checked."""
    return MagicMock(name="prisma_client")


def _reset_general_settings(value):
    """Set `general_settings` to a known dict for the duration of the test."""
    return patch.object(proxy_server, "general_settings", value, create=True)


class TestIsBackgroundHealthCheckDbSaveDisabled:
    def test_default_false_when_key_missing(self):
        with _reset_general_settings({}):
            assert proxy_server._is_background_health_check_db_save_disabled() is False

    def test_default_false_when_general_settings_is_none(self):
        # In some startup paths `general_settings` may briefly be None.
        with _reset_general_settings(None):
            assert proxy_server._is_background_health_check_db_save_disabled() is False

    @pytest.mark.parametrize("truthy", [True, "true", "True", "TRUE", " true "])
    def test_truthy_values_disable(self, truthy):
        with _reset_general_settings(
            {"disable_background_health_check_db_save": truthy}
        ):
            assert proxy_server._is_background_health_check_db_save_disabled() is True

    @pytest.mark.parametrize("falsey", [False, "false", "False", "", "no", 0])
    def test_falsey_values_do_not_disable(self, falsey):
        with _reset_general_settings(
            {"disable_background_health_check_db_save": falsey}
        ):
            assert (
                proxy_server._is_background_health_check_db_save_disabled() is False
            )


class TestScheduleBackgroundHealthCheckDbSave:
    """Behavior tests for the gate inside `_schedule_background_health_check_db_save`."""

    @patch("litellm.proxy.proxy_server.asyncio.create_task")
    def test_no_prisma_client_is_noop(self, mock_create_task):
        # Sanity: when prisma is None the scheduler must not touch asyncio.
        with _reset_general_settings({}):
            proxy_server._schedule_background_health_check_db_save(
                prisma_client=None,
                shared_health_manager=None,
                model_list=[{"model_name": "gpt-4"}],
                healthy_endpoints=[{"model": "gpt-4"}],
                unhealthy_endpoints=[],
            )
        mock_create_task.assert_not_called()

    @patch("litellm.proxy.proxy_server.asyncio.create_task")
    def test_default_settings_still_schedule_db_save(self, mock_create_task):
        # Backward compat: without the flag, we still schedule the DB save.
        with _reset_general_settings({}):
            proxy_server._schedule_background_health_check_db_save(
                prisma_client=_make_prisma(),
                shared_health_manager=None,
                model_list=[{"model_name": "gpt-4"}],
                healthy_endpoints=[{"model": "gpt-4"}],
                unhealthy_endpoints=[],
            )
        assert mock_create_task.call_count == 1

    @patch("litellm.proxy.proxy_server.asyncio.create_task")
    def test_flag_true_skips_db_save(self, mock_create_task):
        # The fix: flag is honored end-to-end and no task is scheduled.
        with _reset_general_settings(
            {"disable_background_health_check_db_save": True}
        ):
            proxy_server._schedule_background_health_check_db_save(
                prisma_client=_make_prisma(),
                shared_health_manager=None,
                model_list=[{"model_name": "gpt-4"}],
                healthy_endpoints=[{"model": "gpt-4"}],
                unhealthy_endpoints=[],
            )
        mock_create_task.assert_not_called()

    @patch("litellm.proxy.proxy_server.asyncio.create_task")
    def test_flag_string_true_skips_db_save(self, mock_create_task):
        # YAML loads sometimes hand us strings; "true" must work too.
        with _reset_general_settings(
            {"disable_background_health_check_db_save": "true"}
        ):
            proxy_server._schedule_background_health_check_db_save(
                prisma_client=_make_prisma(),
                shared_health_manager=None,
                model_list=[{"model_name": "gpt-4"}],
                healthy_endpoints=[{"model": "gpt-4"}],
                unhealthy_endpoints=[],
            )
        mock_create_task.assert_not_called()

    @patch("litellm.proxy.proxy_server.asyncio.create_task")
    def test_flag_false_does_not_skip(self, mock_create_task):
        with _reset_general_settings(
            {"disable_background_health_check_db_save": False}
        ):
            proxy_server._schedule_background_health_check_db_save(
                prisma_client=_make_prisma(),
                shared_health_manager=None,
                model_list=[{"model_name": "gpt-4"}],
                healthy_endpoints=[{"model": "gpt-4"}],
                unhealthy_endpoints=[],
            )
        assert mock_create_task.call_count == 1
