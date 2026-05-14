"""Regression tests for issue #149: auto-retire stale agents.

Long-running multi-agent projects accumulate "active" agents whose
sessions ended without an explicit `retire_agent` call. After ~30+
of these, every new agent broadcast triggers contact_approval for
all of them and silently fails delivery. The `sweep_stale_agents`
helper retires agents whose `last_active_ts` is older than a
caller-configurable threshold so the contact-wall stops piling up.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastmcp import Client

from mcp_agent_mail.app import build_mcp_server, sweep_stale_agents
from mcp_agent_mail.db import get_session
from mcp_agent_mail.models import Agent


def _naive_utc(when: datetime | None = None) -> datetime:
    target = when or datetime.now(timezone.utc)
    if target.tzinfo is not None:
        target = target.astimezone(timezone.utc).replace(tzinfo=None)
    return target


@pytest.mark.asyncio
async def test_sweep_retires_only_agents_past_threshold(isolated_env):
    server = build_mcp_server()
    async with Client(server) as client:
        await client.call_tool("ensure_project", {"human_key": "/staleagents"})

        stale_result = await client.call_tool(
            "register_agent",
            {
                "project_key": "Staleagents",
                "program": "claude-code",
                "model": "opus-4",
                "name": "DustyMountain",
                "task_description": "Long-since-quiet agent",
            },
        )
        active_result = await client.call_tool(
            "register_agent",
            {
                "project_key": "Staleagents",
                "program": "claude-code",
                "model": "opus-4",
                "name": "BrightForest",
                "task_description": "Currently working agent",
            },
        )

        stale_name = stale_result.data["name"]
        active_name = active_result.data["name"]

        # Backdate the stale agent's last_active_ts to two days ago.
        async with get_session() as session:
            stale_agent = (
                await session.execute(
                    Agent.__table__.select().where(Agent.name == stale_name)
                )
            ).first()
            assert stale_agent is not None
            stale_id = stale_agent.id
            two_days_ago = _naive_utc(
                datetime.now(timezone.utc) - timedelta(hours=48)
            )
            await session.execute(
                Agent.__table__.update()
                .where(Agent.id == stale_id)
                .values(last_active_ts=two_days_ago, retired_at=None)
            )
            await session.commit()

        # 24h threshold: stale agent retires, active agent stays.
        retired = await sweep_stale_agents(threshold_seconds=86400)
        assert [entry["agent_name"] for entry in retired] == [stale_name]

        async with get_session() as session:
            stale_after = (
                await session.execute(
                    Agent.__table__.select().where(Agent.name == stale_name)
                )
            ).first()
            active_after = (
                await session.execute(
                    Agent.__table__.select().where(Agent.name == active_name)
                )
            ).first()
            assert stale_after is not None
            assert active_after is not None
            assert stale_after.retired_at is not None
            assert active_after.retired_at is None


@pytest.mark.asyncio
async def test_sweep_is_idempotent(isolated_env):
    server = build_mcp_server()
    async with Client(server) as client:
        await client.call_tool("ensure_project", {"human_key": "/staleidempo"})

        result = await client.call_tool(
            "register_agent",
            {
                "project_key": "Staleidempo",
                "program": "claude-code",
                "model": "opus-4",
                "name": "QuietRiver",
                "task_description": "Will be backdated",
            },
        )
        target_name = result.data["name"]

        async with get_session() as session:
            target_id = (
                (
                    await session.execute(
                        Agent.__table__.select().where(Agent.name == target_name)
                    )
                )
                .first()
                .id
            )
            two_days_ago = _naive_utc(
                datetime.now(timezone.utc) - timedelta(hours=48)
            )
            await session.execute(
                Agent.__table__.update()
                .where(Agent.id == target_id)
                .values(last_active_ts=two_days_ago, retired_at=None)
            )
            await session.commit()

        first = await sweep_stale_agents(threshold_seconds=86400)
        assert len(first) == 1
        # Second pass must not re-retire the same agent.
        second = await sweep_stale_agents(threshold_seconds=86400)
        assert second == []


@pytest.mark.asyncio
async def test_sweep_threshold_floor(isolated_env):
    """Threshold values below 60s are clamped — sweep must still execute."""
    server = build_mcp_server()
    async with Client(server) as client:
        await client.call_tool("ensure_project", {"human_key": "/stalefloor"})

        await client.call_tool(
            "register_agent",
            {
                "project_key": "Stalefloor",
                "program": "claude-code",
                "model": "opus-4",
                "name": "FuzzyCloud",
                "task_description": "Just registered",
            },
        )

        # last_active_ts was set just now; even a 0s argument is clamped to 60s,
        # so the just-registered agent must still NOT retire.
        retired = await sweep_stale_agents(threshold_seconds=0)
        assert retired == []


@pytest.mark.asyncio
async def test_sweep_include_deregistered_retires_regardless_of_age(isolated_env):
    """include_deregistered=True retires agents carrying the [DEREGISTERED ...]
    marker even when they were active well within the threshold — the
    regression safety net / historical backfill path."""
    server = build_mcp_server()
    async with Client(server) as client:
        await client.call_tool("ensure_project", {"human_key": "/staledereg"})
        names = {}
        for label, requested in (("fresh", "BrightForest"), ("ghost", "DustyMountain")):
            res = await client.call_tool(
                "register_agent",
                {
                    "project_key": "Staledereg",
                    "program": "claude-code",
                    "model": "opus-4",
                    "name": requested,
                    "task_description": f"work for {label}",
                },
            )
            names[label] = res.data["name"]

        # Simulate the historical bug: an agent that recorded its own
        # deregistration marker but was left retired_at=NULL and recently active.
        async with get_session() as session:
            ghost = (
                await session.execute(
                    Agent.__table__.select().where(Agent.name == names["ghost"])
                )
            ).first()
            await session.execute(
                Agent.__table__.update()
                .where(Agent.id == ghost.id)
                .values(
                    task_description="[DEREGISTERED at 2026-01-01T00:00:00+00:00] old work",
                    retired_at=None,
                )
            )
            await session.commit()

        # Default sweep (age-only) leaves the recently-active ghost alone.
        assert await sweep_stale_agents(threshold_seconds=86400) == []

        # include_deregistered=True catches it regardless of age.
        retired = await sweep_stale_agents(
            threshold_seconds=86400, include_deregistered=True
        )
        assert [e["agent_name"] for e in retired] == [names["ghost"]]
        assert retired[0]["reason"] == "deregistered"

        # The fresh agent (no marker, recently active) is untouched.
        async with get_session() as session:
            fresh = (
                await session.execute(
                    Agent.__table__.select().where(Agent.name == names["fresh"])
                )
            ).first()
            assert fresh.retired_at is None


@pytest.mark.asyncio
async def test_sweep_dry_run_reports_without_writing(isolated_env):
    """dry_run=True returns the candidate list but writes no retired_at."""
    server = build_mcp_server()
    async with Client(server) as client:
        await client.call_tool("ensure_project", {"human_key": "/staledry"})
        result = await client.call_tool(
            "register_agent",
            {
                "project_key": "Staledry",
                "program": "claude-code",
                "model": "opus-4",
                "name": "DozyBadger",
                "task_description": "Will be backdated",
            },
        )
        target_name = result.data["name"]

        async with get_session() as session:
            target_id = (
                (
                    await session.execute(
                        Agent.__table__.select().where(Agent.name == target_name)
                    )
                )
                .first()
                .id
            )
            two_days_ago = _naive_utc(datetime.now(timezone.utc) - timedelta(hours=48))
            await session.execute(
                Agent.__table__.update()
                .where(Agent.id == target_id)
                .values(last_active_ts=two_days_ago, retired_at=None)
            )
            await session.commit()

        # Dry run: reported as a candidate...
        preview = await sweep_stale_agents(threshold_seconds=86400, dry_run=True)
        assert [e["agent_name"] for e in preview] == [target_name]

        # ...but retired_at was NOT written.
        async with get_session() as session:
            row = (
                await session.execute(
                    Agent.__table__.select().where(Agent.id == target_id)
                )
            ).first()
            assert row.retired_at is None, "dry_run must not write retired_at"

        # A real sweep still retires it.
        retired = await sweep_stale_agents(threshold_seconds=86400)
        assert [e["agent_name"] for e in retired] == [target_name]
