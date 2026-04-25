"""Tests for benchmark client authentication fallback."""

from datetime import UTC, datetime, timedelta

import pytest

from benchmarks.snipara_client import SniparaClient


class _FakeRefreshResponse:
    status_code = 400
    text = '{"error":"invalid_grant"}'


class _FakeRefreshClient:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, *args, **kwargs):
        return _FakeRefreshResponse()


@pytest.mark.asyncio
async def test_snipara_client_falls_back_to_api_key_on_refresh_failure(monkeypatch):
    """Expired OAuth tokens should not block benchmark runs when an API key exists."""
    client = SniparaClient(
        api_key="rlm_test_key",
        access_token="expired_access_token",
        project_slug="snipara",
    )
    client._refresh_token = "expired_refresh_token"
    client._expires_at = datetime.now(UTC) - timedelta(hours=1)

    monkeypatch.setattr(
        "benchmarks.snipara_client.httpx.AsyncClient", lambda *args, **kwargs: _FakeRefreshClient()
    )

    await client._ensure_token_valid()

    assert client.access_token is None
    assert client._refresh_token is None
    assert client._expires_at is None


def test_snipara_client_can_prefer_api_key_over_local_oauth():
    """Benchmarks should be able to skip local OAuth state explicitly."""
    client = SniparaClient(
        api_key="rlm_test_key",
        access_token="oauth_access_token",
        project_slug="snipara",
        prefer_api_key=True,
    )

    assert client.access_token is None
