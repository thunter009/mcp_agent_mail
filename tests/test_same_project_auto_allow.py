"""Tests for same-project contact-wall bypass (CONTACT_SAME_PROJECT_AUTO_ALLOW).

Agents that share a project_key are, by construction, the same operator's
cooperating sessions on one checkout — there is no trust boundary between
them. The contact-approval wall exists for CROSS-project spam prevention;
applied within a project it is pure friction: a fresh, live sister session
on the default `auto` policy would otherwise wall every broadcast until a
handshake completes (and a handshake across two separate MCP sessions
cannot auto-complete).

These tests register the recipient(s) in a *separate* MCP session from the
sender, so the in-session auto-handshake path cannot mask the behavior —
this is the real-world multi-session scenario.

Covers:
- same-project broadcast succeeds without any prior handshake
- same-project direct send succeeds without any prior handshake
- `contacts_only` is still an enforced, explicit opt-out
- `block_all` is still an enforced, explicit opt-out
- CONTACT_SAME_PROJECT_AUTO_ALLOW=false restores the old wall
"""

from __future__ import annotations

import contextlib

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from mcp_agent_mail import config as _config
from mcp_agent_mail.app import build_mcp_server

PROJECT = "/test/same-project-auto-allow"


async def _register(client, project_key: str, name: str) -> str:
    """Register an agent and return its registration token."""
    result = await client.call_tool(
        "register_agent",
        {"project_key": project_key, "program": "codex", "model": "gpt-5", "name": name},
    )
    return result.data["registration_token"]


@pytest.mark.asyncio
async def test_same_project_broadcast_succeeds_without_handshake(isolated_env):
    """A broadcast reaches every same-project peer with no contact handshake."""
    server = build_mcp_server()
    async with Client(server) as bootstrap:
        await bootstrap.call_tool("ensure_project", {"human_key": PROJECT})
        sender_token = await _register(bootstrap, PROJECT, "GreenCastle")
        await _register(bootstrap, PROJECT, "BlueLake")
        await _register(bootstrap, PROJECT, "RedStone")

    # Separate MCP session: in-session auto-handshake is NOT available here,
    # so success proves the same-project bypass — not the handshake fallback.
    async with Client(server) as sender:
        result = await sender.call_tool(
            "send_message",
            {
                "project_key": PROJECT,
                "sender_name": "GreenCastle",
                "sender_token": sender_token,
                "to": [],
                "subject": "Broadcast without handshake",
                "body_md": "should reach every same-project peer",
                "broadcast": True,
            },
        )
        payload = result.data["deliveries"][0]["payload"]
        recipients = payload.get("to", [])
        assert "BlueLake" in recipients
        assert "RedStone" in recipients
        assert "GreenCastle" not in recipients


@pytest.mark.asyncio
async def test_same_project_direct_send_succeeds_without_handshake(isolated_env):
    """A direct send to a same-project peer needs no contact handshake."""
    server = build_mcp_server()
    async with Client(server) as bootstrap:
        await bootstrap.call_tool("ensure_project", {"human_key": PROJECT})
        sender_token = await _register(bootstrap, PROJECT, "GreenCastle")
        await _register(bootstrap, PROJECT, "BlueLake")

    async with Client(server) as sender:
        result = await sender.call_tool(
            "send_message",
            {
                "project_key": PROJECT,
                "sender_name": "GreenCastle",
                "sender_token": sender_token,
                "to": ["BlueLake"],
                "subject": "Direct without handshake",
                "body_md": "ping",
            },
        )
        assert result.data["count"] == 1


@pytest.mark.asyncio
async def test_same_project_contacts_only_still_enforced(isolated_env):
    """`contacts_only` remains an explicit opt-out even for same-project peers."""
    server = build_mcp_server()
    async with Client(server) as bootstrap:
        await bootstrap.call_tool("ensure_project", {"human_key": PROJECT})
        sender_token = await _register(bootstrap, PROJECT, "GreenCastle")
        await _register(bootstrap, PROJECT, "BlueLake")
        await bootstrap.call_tool(
            "set_contact_policy",
            {"project_key": PROJECT, "agent_name": "BlueLake", "policy": "contacts_only"},
        )

    async with Client(server) as sender:
        with pytest.raises(ToolError) as exc_info:
            await sender.call_tool(
                "send_message",
                {
                    "project_key": PROJECT,
                    "sender_name": "GreenCastle",
                    "sender_token": sender_token,
                    "to": ["BlueLake"],
                    "subject": "Should be walled",
                    "body_md": "contacts_only must still block",
                    "auto_contact_if_blocked": False,
                },
            )
        assert "Contact approval required" in str(exc_info.value)


@pytest.mark.asyncio
async def test_same_project_block_all_still_enforced(isolated_env):
    """`block_all` remains an explicit opt-out even for same-project peers."""
    server = build_mcp_server()
    async with Client(server) as bootstrap:
        await bootstrap.call_tool("ensure_project", {"human_key": PROJECT})
        sender_token = await _register(bootstrap, PROJECT, "GreenCastle")
        await _register(bootstrap, PROJECT, "BlueLake")
        await bootstrap.call_tool(
            "set_contact_policy",
            {"project_key": PROJECT, "agent_name": "BlueLake", "policy": "block_all"},
        )

    async with Client(server) as sender:
        with pytest.raises(ToolError) as exc_info:
            await sender.call_tool(
                "send_message",
                {
                    "project_key": PROJECT,
                    "sender_name": "GreenCastle",
                    "sender_token": sender_token,
                    "to": ["BlueLake"],
                    "subject": "Should be blocked",
                    "body_md": "block_all must still block",
                },
            )
        assert "not accepting messages" in str(exc_info.value)


@pytest.mark.asyncio
async def test_same_project_auto_allow_disabled_restores_wall(isolated_env, monkeypatch):
    """CONTACT_SAME_PROJECT_AUTO_ALLOW=false restores the pre-fix contact wall."""
    monkeypatch.setenv("CONTACT_SAME_PROJECT_AUTO_ALLOW", "false")
    with contextlib.suppress(Exception):
        _config.clear_settings_cache()

    server = build_mcp_server()
    async with Client(server) as bootstrap:
        await bootstrap.call_tool("ensure_project", {"human_key": PROJECT})
        sender_token = await _register(bootstrap, PROJECT, "GreenCastle")
        await _register(bootstrap, PROJECT, "BlueLake")

    async with Client(server) as sender:
        with pytest.raises(ToolError) as exc_info:
            await sender.call_tool(
                "send_message",
                {
                    "project_key": PROJECT,
                    "sender_name": "GreenCastle",
                    "sender_token": sender_token,
                    "to": ["BlueLake"],
                    "subject": "Walled when disabled",
                    "body_md": "default auto policy should block again",
                    "auto_contact_if_blocked": False,
                },
            )
        assert "Contact approval required" in str(exc_info.value)
