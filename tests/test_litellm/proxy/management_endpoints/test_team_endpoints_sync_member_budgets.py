"""
Tests for LIT-3223: opt-in auto-sync of team member budgets when the team
budget is updated via /team/update.

The default behaviour of ``backfill_team_member_budget_entries`` is unchanged:
rows with a non-NULL ``budget_id`` (per-member overrides) are left alone, and
only rows with ``budget_id IS NULL`` are healed onto the new team default.

When the caller passes ``sync_member_budgets=True`` on the
``UpdateTeamRequest``, the proxy forwards
``sync_existing_member_budgets=True`` into the backfill and EVERY membership
row for the team is re-pointed at the new team default budget — including
rows that previously held a per-member override. This is the auto-sync
behaviour the customer requested in the ticket.
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.proxy._types import Member, UpdateTeamRequest
from litellm.proxy.management_endpoints.team_endpoints import (
    TeamMemberBudgetHandler,
)


@pytest.mark.asyncio
async def test_backfill_preserves_overrides_by_default():
    """Default call (no flag): the {"budget_id": None} filter MUST be kept so
    rows with a per-member override are not touched."""
    team_id = "team-preserve"
    new_budget_id = "team-default-budget-new"

    existing_a = MagicMock(); existing_a.user_id = "user-A"
    existing_b = MagicMock(); existing_b.user_id = "user-B"

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_teammembership.find_many = AsyncMock(
        return_value=[existing_a, existing_b]
    )
    mock_prisma.db.litellm_teammembership.create_many = AsyncMock(return_value=None)
    mock_prisma.db.litellm_teammembership.update_many = AsyncMock(return_value=1)

    await TeamMemberBudgetHandler.backfill_team_member_budget_entries(
        team_id=team_id,
        members_with_roles=[
            Member(user_id="user-A", role="user"),
            Member(user_id="user-B", role="user"),
        ],
        team_member_budget_id=new_budget_id,
        prisma_client=mock_prisma,
    )

    mock_prisma.db.litellm_teammembership.create_many.assert_not_awaited()
    mock_prisma.db.litellm_teammembership.update_many.assert_awaited_once_with(
        where={"team_id": team_id, "budget_id": None},
        data={"budget_id": new_budget_id},
    )


@pytest.mark.asyncio
async def test_backfill_sync_flag_overrides_all_rows():
    """sync_existing_member_budgets=True must drop the budget_id filter so
    every team membership (including overrides) is re-pointed at the new
    team default — this is the LIT-3223 auto-sync behaviour."""
    team_id = "team-sync"
    new_budget_id = "team-default-budget-new"

    existing_a = MagicMock(); existing_a.user_id = "user-A"
    existing_b = MagicMock(); existing_b.user_id = "user-B"

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_teammembership.find_many = AsyncMock(
        return_value=[existing_a, existing_b]
    )
    mock_prisma.db.litellm_teammembership.create_many = AsyncMock(return_value=None)
    mock_prisma.db.litellm_teammembership.update_many = AsyncMock(return_value=2)

    await TeamMemberBudgetHandler.backfill_team_member_budget_entries(
        team_id=team_id,
        members_with_roles=[
            Member(user_id="user-A", role="user"),
            Member(user_id="user-B", role="user"),
        ],
        team_member_budget_id=new_budget_id,
        prisma_client=mock_prisma,
        sync_existing_member_budgets=True,
    )

    mock_prisma.db.litellm_teammembership.create_many.assert_not_awaited()
    mock_prisma.db.litellm_teammembership.update_many.assert_awaited_once_with(
        where={"team_id": team_id},
        data={"budget_id": new_budget_id},
    )


@pytest.mark.asyncio
async def test_backfill_sync_flag_still_creates_missing_rows():
    """sync_existing_member_budgets=True must NOT regress the existing
    behaviour of creating membership rows for members who don't have one."""
    team_id = "team-mix"
    new_budget_id = "budget-xyz"

    existing_a = MagicMock(); existing_a.user_id = "user-A"  # already has a row

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_teammembership.find_many = AsyncMock(
        return_value=[existing_a]
    )
    mock_prisma.db.litellm_teammembership.create_many = AsyncMock(return_value=None)
    mock_prisma.db.litellm_teammembership.update_many = AsyncMock(return_value=1)

    await TeamMemberBudgetHandler.backfill_team_member_budget_entries(
        team_id=team_id,
        members_with_roles=[
            Member(user_id="user-A", role="user"),
            Member(user_id="user-B", role="user"),  # missing -> create_many
        ],
        team_member_budget_id=new_budget_id,
        prisma_client=mock_prisma,
        sync_existing_member_budgets=True,
    )

    mock_prisma.db.litellm_teammembership.create_many.assert_awaited_once()
    call_kwargs = mock_prisma.db.litellm_teammembership.create_many.call_args.kwargs
    assert call_kwargs["data"] == [
        {"team_id": team_id, "user_id": "user-B", "budget_id": new_budget_id}
    ]
    assert call_kwargs["skip_duplicates"] is True

    mock_prisma.db.litellm_teammembership.update_many.assert_awaited_once_with(
        where={"team_id": team_id},
        data={"budget_id": new_budget_id},
    )


def test_update_team_request_accepts_sync_member_budgets():
    """The new opt-in field on UpdateTeamRequest must be a typed Optional[bool]
    that defaults to None and round-trips through Pydantic correctly."""
    # Not set -> not present in exclude_unset payload (and value is None)
    req_unset = UpdateTeamRequest(team_id="t1")
    payload_unset = req_unset.json(exclude_unset=True)
    assert "sync_member_budgets" not in payload_unset
    assert req_unset.sync_member_budgets is None

    # Set True -> present and True
    req_true = UpdateTeamRequest(team_id="t1", sync_member_budgets=True)
    assert req_true.sync_member_budgets is True
    assert req_true.json(exclude_unset=True).get("sync_member_budgets") is True

    # Set False -> present and False
    req_false = UpdateTeamRequest(team_id="t1", sync_member_budgets=False)
    assert req_false.sync_member_budgets is False
    assert req_false.json(exclude_unset=True).get("sync_member_budgets") is False
