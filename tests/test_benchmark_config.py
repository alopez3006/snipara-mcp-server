import asyncio
from pathlib import Path

import pytest

from benchmarks.config import BenchmarkConfig, load_oauth_access_token
from benchmarks.runner import BenchmarkRunner


def test_load_oauth_access_token_matches_project_id(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    token_dir = tmp_path / ".snipara"
    token_dir.mkdir()
    (token_dir / "tokens.json").write_text(
        """
        {
          "proj_1": {
            "project_id": "proj_1",
            "project_slug": "alpha",
            "access_token": "token-alpha"
          },
          "proj_2": {
            "project_id": "proj_2",
            "project_slug": "beta",
            "access_token": "token-beta"
          }
        }
        """.strip()
    )

    assert load_oauth_access_token("proj_2") == "token-beta"
    assert load_oauth_access_token("alpha") == "token-alpha"


def test_benchmark_config_uses_project_id_and_api_key_env(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("SNIPARA_PROJECT_SLUG", raising=False)
    monkeypatch.setenv("SNIPARA_PROJECT_ID", "proj_env_123")
    monkeypatch.setenv("SNIPARA_API_KEY", "rlm_test_key")

    cfg = BenchmarkConfig()

    assert cfg.snipara_project_slug == "proj_env_123"
    assert cfg.snipara_api_key == "rlm_test_key"
    assert cfg.snipara_oauth_token is None


def test_runner_passes_api_key_to_client(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    captured = {}

    async def fake_create_client(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("benchmarks.runner.create_client", fake_create_client)

    cfg = BenchmarkConfig(
        snipara_project_slug="proj_runner_123",
        snipara_api_key="rlm_runner_key",
        snipara_oauth_token=None,
    )
    runner = BenchmarkRunner(config=cfg, use_api=True)

    asyncio.run(runner._get_api_client())

    assert captured["use_real_api"] is True
    assert captured["api_key"] == "rlm_runner_key"
    assert captured["access_token"] is None
    assert captured["project_slug"] == "proj_runner_123"
