"""Bare-name project resolution in _get_project_record (bd-i3q).

Operators call e.g. `sweep-stale-agents --project off-earth-data`, but slugs
are derived from full paths ('users-thom-10-19-projects-off-earth-data').
_get_project_record resolves a bare name when its slug suffix is unique,
errors listing candidates when ambiguous, and never regresses exact matches.
"""

import asyncio

import pytest

from mcp_agent_mail.cli import _get_project_record
from mcp_agent_mail.db import ensure_schema, get_session
from mcp_agent_mail.models import Project


def _seed(projects: list[tuple[str, str]]) -> None:
    async def inner() -> None:
        await ensure_schema()
        async with get_session() as session:
            for slug, human_key in projects:
                session.add(Project(slug=slug, human_key=human_key))
            await session.commit()

    asyncio.run(inner())


def test_exact_slug_still_wins(isolated_env):
    _seed([
        ("users-thom-projects-off-earth-data", "/Users/thom/projects/off-earth-data"),
        ("off-earth-data", "/elsewhere/off-earth-data"),
    ])
    # 'off-earth-data' is itself a real slug -> exact match, no fallback
    project = asyncio.run(_get_project_record("off-earth-data"))
    assert project.human_key == "/elsewhere/off-earth-data"


def test_bare_name_resolves_unique_suffix(isolated_env):
    _seed([
        ("users-thom-projects-off-earth-data", "/Users/thom/projects/off-earth-data"),
        ("users-thom-projects-agent-skills", "/Users/thom/projects/agent-skills"),
    ])
    project = asyncio.run(_get_project_record("off-earth-data"))
    assert project.slug == "users-thom-projects-off-earth-data"


def test_bare_name_ambiguous_lists_candidates(isolated_env):
    _seed([
        ("users-thom-projects-off-earth-data", "/Users/thom/projects/off-earth-data"),
        ("users-alice-work-off-earth-data", "/Users/alice/work/off-earth-data"),
    ])
    with pytest.raises(ValueError) as excinfo:
        asyncio.run(_get_project_record("off-earth-data"))
    message = str(excinfo.value)
    assert "ambiguous" in message
    assert "/Users/thom/projects/off-earth-data" in message
    assert "/Users/alice/work/off-earth-data" in message


def test_unknown_name_still_not_found(isolated_env):
    _seed([
        ("users-thom-projects-agent-skills", "/Users/thom/projects/agent-skills"),
    ])
    with pytest.raises(ValueError, match="not found"):
        asyncio.run(_get_project_record("no-such-project"))


def test_full_human_key_unaffected(isolated_env):
    _seed([
        ("users-thom-projects-off-earth-data", "/Users/thom/projects/off-earth-data"),
    ])
    project = asyncio.run(_get_project_record("/Users/thom/projects/off-earth-data"))
    assert project.slug == "users-thom-projects-off-earth-data"


def test_substring_without_boundary_does_not_match(isolated_env):
    # 'data' must not suffix-match '...-off-earth-data' ('-data' would, but
    # only across the '-' boundary: slug endswith('-data') IS true here —
    # the guard under test is that 'earth-data' does not match 'earth-data'
    # embedded mid-slug, only as a full trailing segment sequence.
    _seed([
        ("users-thom-projects-off-earth-database", "/Users/thom/projects/off-earth-database"),
    ])
    with pytest.raises(ValueError, match="not found"):
        asyncio.run(_get_project_record("earth-data"))
