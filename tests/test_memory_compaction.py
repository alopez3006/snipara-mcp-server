"""Regression tests for memory compaction hygiene rules."""

from __future__ import annotations

import importlib
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest


@pytest.fixture(scope="module")
def agent_memory_module():
    """Import the service from the apps/mcp-server package context."""
    project_root = Path(__file__).resolve().parents[1]
    previous_cwd = Path.cwd()
    os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test")
    os.chdir(project_root)
    try:
        sys.modules.pop("src.services.agent_memory", None)
        sys.modules.pop("src.services", None)
        module = importlib.import_module("src.services.agent_memory")
        yield importlib.reload(module)
    finally:
        os.chdir(previous_cwd)


def _memory(**overrides):
    now = datetime.now(UTC)
    payload = {
        "id": "mem_default",
        "content": "durable memory",
        "type": "FACT",
        "scope": "PROJECT",
        "category": "general",
        "confidence": 1.0,
        "accessCount": 0,
        "documentRefs": [],
        "createdAt": now,
        "lastAccessedAt": None,
        "reviewStatus": "APPROVED",
        "tier": "DAILY",
    }
    payload.update(overrides)
    return type("MemoryRow", (), payload)()


def test_classify_low_signal_memory_targets_safe_noise_patterns(agent_memory_module):
    """Compaction should only prune patterns we have explicitly classified as noise."""
    assert (
        agent_memory_module._classify_low_signal_memory(
            _memory(
                id="superseded",
                type="LEARNING",
                category="workspace-learning-0000:superseded:superseded",
                content="Old operational learning",
            )
        )
        == agent_memory_module.LOW_SIGNAL_REASON_SUPERSEDED_WORKSPACE_LEARNING
    )
    assert (
        agent_memory_module._classify_low_signal_memory(
            _memory(
                id="tombstone",
                type="FACT",
                scope="AGENT",
                category="agent-jarvis",
                content="[DELETED memory mem_123]",
            )
        )
        == agent_memory_module.LOW_SIGNAL_REASON_DELETED_TOMBSTONE
    )
    assert (
        agent_memory_module._classify_low_signal_memory(
            _memory(
                id="sync",
                type="LEARNING",
                category="workspace-learning-0000",
                content="SYNCTEST-prod-103420 simple task: Created by backend-only sync test.",
            )
        )
        == agent_memory_module.LOW_SIGNAL_REASON_SYNC_TEST
    )
    assert (
        agent_memory_module._classify_low_signal_memory(
            _memory(
                id="task",
                type="LEARNING",
                category="task-learning",
                content='Task "Publish next social post" completed by Max: Liens utiles: - [Open file in Drive](...)',
            )
        )
        == agent_memory_module.LOW_SIGNAL_REASON_TASK_JOURNAL
    )
    assert (
        agent_memory_module._classify_low_signal_memory(
            _memory(
                id="keep",
                type="LEARNING",
                category="workspace-learning-0000",
                content="Use Bun because startup time and install surface are lower than Node.",
            )
        )
        is None
    )


@pytest.mark.asyncio
async def test_compact_memories_prunes_low_signal_noise_and_cleans_embeddings(
    monkeypatch,
    agent_memory_module,
):
    """Compaction should drop known-noise memories before the generic phases run."""

    memories = [
        _memory(
            id="superseded",
            type="LEARNING",
            category="workspace-learning-0000:superseded:superseded",
            content="Old operational learning",
        ),
        _memory(
            id="tombstone",
            type="FACT",
            scope="AGENT",
            category="agent-jarvis",
            content="[DELETED memory mem_123]",
        ),
        _memory(
            id="sync",
            type="LEARNING",
            category="workspace-learning-0000",
            content="SYNCTEST-prod-103420 simple task: Created by backend-only sync test.",
        ),
        _memory(
            id="task",
            type="LEARNING",
            category="task-learning",
            content='Task "Publish next social post" completed by Max: Liens utiles: - [Open file in Drive](...)',
        ),
        _memory(
            id="keep",
            type="FACT",
            category="models",
            content="Use Haiku for lightweight sub-agents.",
        ),
    ]

    find_many = AsyncMock(side_effect=[memories, [], []])
    delete_many = AsyncMock(return_value=4)
    update = AsyncMock()
    mock_db = type(
        "MockDb",
        (),
        {
            "agentmemory": type(
                "AgentMemoryRepo",
                (),
                {
                    "find_many": find_many,
                    "delete_many": delete_many,
                    "update": update,
                },
            )()
        },
    )()

    delete_embedding = AsyncMock()

    monkeypatch.setattr(agent_memory_module, "get_db", AsyncMock(return_value=mock_db))
    monkeypatch.setattr(agent_memory_module, "_delete_memory_embedding", delete_embedding)

    result = await agent_memory_module.compact_memories(
        project_id="proj_test",
        deduplicate=False,
        promote_threshold=99,
        archive_older_than_days=3650,
        dry_run=False,
        normalize_dates=False,
        validate_refs=False,
        conflict_strategy="",
    )

    delete_many.assert_awaited_once_with(
        where={"id": {"in": ["superseded", "tombstone", "sync", "task"]}}
    )
    assert delete_embedding.await_count == 4
    assert result["noise_pruned"] == 4
    assert result["superseded_workspace_learning_removed"] == 1
    assert result["deleted_tombstones_removed"] == 1
    assert result["sync_test_noise_removed"] == 1
    assert result["task_journals_removed"] == 1
    assert result["message"].startswith("Successfully: pruned 4 low-signal memories")


@pytest.mark.asyncio
async def test_compact_memories_dry_run_reports_noise_without_deleting(
    monkeypatch,
    agent_memory_module,
):
    """Dry runs should surface the same hygiene counts without mutating storage."""

    memories = [
        _memory(
            id="tombstone",
            type="FACT",
            scope="AGENT",
            category="agent-jarvis",
            content="[DELETED memory mem_123]",
        ),
        _memory(
            id="keep",
            type="FACT",
            category="models",
            content="Use Haiku for lightweight sub-agents.",
        ),
    ]

    find_many = AsyncMock(side_effect=[memories, [], []])
    delete_many = AsyncMock()
    mock_db = type(
        "MockDb",
        (),
        {
            "agentmemory": type(
                "AgentMemoryRepo",
                (),
                {
                    "find_many": find_many,
                    "delete_many": delete_many,
                    "update": AsyncMock(),
                },
            )()
        },
    )()

    delete_embedding = AsyncMock()

    monkeypatch.setattr(agent_memory_module, "get_db", AsyncMock(return_value=mock_db))
    monkeypatch.setattr(agent_memory_module, "_delete_memory_embedding", delete_embedding)

    result = await agent_memory_module.compact_memories(
        project_id="proj_test",
        deduplicate=False,
        promote_threshold=99,
        archive_older_than_days=3650,
        dry_run=True,
        normalize_dates=False,
        validate_refs=False,
        conflict_strategy="",
    )

    delete_many.assert_not_awaited()
    delete_embedding.assert_not_awaited()
    assert result["noise_pruned"] == 1
    assert result["deleted_tombstones_removed"] == 1
    assert result["message"].startswith("Would have: pruned 1 low-signal memories")
