"""Regression tests: deregister_agent must actually retire the agent.

The historical bug: `deregister_agent` set `contact_policy="block_all"` and
stamped a `[DEREGISTERED at ...]` marker into `task_description`, but left
`retired_at = NULL`. So every well-behaved session that *did* clean up still
showed as "active" forever — polluting `resource://agents/{project}` and the
auto-retire sweep's working set, and (for auto-policy agents) walling
broadcast via the contact-approval check.

Fix: `deregister_agent` now sets `retired_at` directly.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastmcp import Client

from mcp_agent_mail.app import build_mcp_server, sweep_stale_agents
from mcp_agent_mail.db import get_session
from mcp_agent_mail.models import Agent


def _parse_resource_json(blocks) -> Any:
    if not blocks:
        return None
    text = "".join(b.text or "" for b in blocks)
    return json.loads(text) if text else None


async def _register(client: Client, project_key: str, name: str) -> tuple[str, str]:
    """Register an agent; return (actual_name, registration_token).

    register_agent coerces names that don't fit the adjective+noun whitelist,
    so callers must use the returned name, not the requested one.
    """
    res = await client.call_tool(
        "register_agent",
        {
            "project_key": project_key,
            "program": "claude-code",
            "model": "opus-4",
            "name": name,
            "task_description": f"work for {name}",
        },
    )
    return res.data["name"], res.data["registration_token"]


@pytest.mark.asyncio
async def test_deregister_agent_sets_retired_at(isolated_env):
    server = build_mcp_server()
    async with Client(server) as client:
        await client.call_tool("ensure_project", {"human_key": "/deregproj"})
        name, token = await _register(client, "Deregproj", "BlueLake")

        await client.call_tool(
            "deregister_agent",
            {
                "project_key": "Deregproj",
                "agent_name": name,
                "registration_token": token,
            },
        )

        async with get_session() as session:
            row = (
                await session.execute(
                    Agent.__table__.select().where(Agent.name == name)
                )
            ).first()
        assert row is not None
        # The core fix: deregister leaves the active roster via retired_at.
        assert row.retired_at is not None, "deregister_agent must set retired_at"
        # And the pre-existing side effects are preserved.
        assert row.contact_policy == "block_all"
        assert row.task_description.startswith("[DEREGISTERED at ")


@pytest.mark.asyncio
async def test_deregistered_agent_moves_to_retired_roster(isolated_env):
    """resource://agents/{project} must list a deregistered agent under
    retired_agents, not the active agents list."""
    server = build_mcp_server()
    async with Client(server) as client:
        await client.call_tool("ensure_project", {"human_key": "/rosterproj"})
        kept_name, _ = await _register(client, "Rosterproj", "GreenField")
        gone_name, token = await _register(client, "Rosterproj", "RedStone")

        await client.call_tool(
            "deregister_agent",
            {
                "project_key": "Rosterproj",
                "agent_name": gone_name,
                "registration_token": token,
            },
        )

        blocks = await client.read_resource("resource://agents/rosterproj")
        data = _parse_resource_json(blocks)

        active_names = {a["name"] for a in data["agents"]}
        retired_names = {a["name"] for a in data["retired_agents"]}
        assert active_names == {kept_name}, f"unexpected active roster: {active_names}"
        assert gone_name in retired_names, "deregistered agent missing from retired_agents"


@pytest.mark.asyncio
async def test_deregistered_agent_skipped_by_stale_sweep(isolated_env):
    """A freshly-deregistered agent is already retired, so the stale sweep
    must not pick it up again (its WHERE filter is retired_at IS NULL)."""
    server = build_mcp_server()
    async with Client(server) as client:
        await client.call_tool("ensure_project", {"human_key": "/sweepskip"})
        name, token = await _register(client, "Sweepskip", "QuietRiver")
        await client.call_tool(
            "deregister_agent",
            {
                "project_key": "Sweepskip",
                "agent_name": name,
                "registration_token": token,
            },
        )

    # Even with a near-zero threshold (clamped to 60s) the agent is not a
    # candidate — it is already retired.
    swept = await sweep_stale_agents(threshold_seconds=1)
    assert [e["agent_name"] for e in swept] == []
