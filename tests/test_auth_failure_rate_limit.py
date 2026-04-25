"""Tests for failed-auth throttling on MCP/API auth surfaces."""

import pytest


@pytest.mark.asyncio
async def test_auth_failure_rate_limit_blocks_after_threshold(monkeypatch):
    from src import usage

    usage._local_auth_failure_limits.clear()

    async def no_redis():
        return None

    monkeypatch.setattr(usage, "get_redis", no_redis)
    monkeypatch.setattr(usage.settings, "auth_failure_rate_limit_requests", 2)
    monkeypatch.setattr(usage.settings, "auth_failure_rate_limit_window", 300)

    assert await usage.check_auth_failure_rate_limit("203.0.113.1", "rlm_invalid") is True
    assert await usage.check_auth_failure_rate_limit("203.0.113.1", "rlm_invalid") is True
    assert await usage.check_auth_failure_rate_limit("203.0.113.1", "rlm_invalid") is False


@pytest.mark.asyncio
async def test_auth_failure_rate_limit_is_scoped_by_ip(monkeypatch):
    from src import usage

    usage._local_auth_failure_limits.clear()

    async def no_redis():
        return None

    monkeypatch.setattr(usage, "get_redis", no_redis)
    monkeypatch.setattr(usage.settings, "auth_failure_rate_limit_requests", 1)
    monkeypatch.setattr(usage.settings, "auth_failure_rate_limit_window", 300)

    assert await usage.check_auth_failure_rate_limit("203.0.113.1", "rlm_invalid") is True
    assert await usage.check_auth_failure_rate_limit("203.0.113.1", "rlm_invalid") is False
    assert await usage.check_auth_failure_rate_limit("203.0.113.2", "rlm_invalid") is True
