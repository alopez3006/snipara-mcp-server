"""Tests for the deterministic companion benchmark helpers."""

import json
from pathlib import Path

from benchmarks.companion_benchmark import (
    build_comparisons,
    compute_percentile,
    default_cases,
    evaluate_success,
    load_project_api_key,
    normalize_companion_api_url,
    render_markdown_report,
)


def test_default_cases_cover_direct_and_companion_paths():
    cases = default_cases()

    ids = {case.id for case in cases}
    modes = {case.mode for case in cases}

    assert "direct_code_callers" in ids
    assert "companion_code_callers" in ids
    assert "direct_context_query" in ids
    assert "companion_workflow_auto" in ids
    assert modes == {"direct", "companion"}


def test_normalize_companion_api_url_strips_trailing_mcp():
    assert normalize_companion_api_url("https://api.snipara.com/mcp") == "https://api.snipara.com"
    assert normalize_companion_api_url("http://localhost:8000/mcp") == "http://localhost:8000"
    assert normalize_companion_api_url("https://api.snipara.com") == "https://api.snipara.com"


def test_load_project_api_key_prefers_exact_project_match(monkeypatch, tmp_path):
    home = tmp_path / "home"
    tokens_dir = home / ".snipara"
    tokens_dir.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: home)
    tokens = {
        "proj_other": {"project_slug": "other", "api_key": "other-key"},
        "proj_snipara": {
            "project_slug": "snipara",
            "project_id": "proj_snipara",
            "api_key": "snipara-key",
        },
    }
    (tokens_dir / "tokens.json").write_text(json.dumps(tokens))

    assert load_project_api_key("snipara") == "snipara-key"
    assert load_project_api_key("proj_snipara") == "snipara-key"
    assert load_project_api_key("missing") is None


def test_compute_percentile_and_success_helpers():
    values = [900, 1100, 1000]
    payload = '{"toolName":"rlm_code_callers","qualified_name":"src.rlm_engine.RLMEngine._handle_context_query"}'

    assert compute_percentile(values, 0.5) == 1000
    assert compute_percentile(values, 0.95) == 1100
    assert evaluate_success(payload, ("rlm_code_callers", "_handle_context_query")) is True
    assert evaluate_success(payload, ("rlm_code_callers", "missing_symbol")) is False


def test_build_comparisons_and_markdown_rendering():
    cases = [
        {
            "id": "direct_code_callers",
            "mode": "direct",
            "comparison_group": "code_callers",
            "success_rate": 1.0,
            "latency_ms": {"p50": 1000, "p95": 1100, "mean": 1000},
            "response_bytes": {"mean": 900},
        },
        {
            "id": "companion_code_callers",
            "mode": "companion",
            "comparison_group": "code_callers",
            "success_rate": 1.0,
            "latency_ms": {"p50": 1150, "p95": 1300, "mean": 1200},
            "response_bytes": {"mean": 900},
        },
    ]
    comparisons = build_comparisons(cases)

    assert comparisons[0]["group"] == "code_callers"
    assert comparisons[0]["p50_delta_ms"] == 150
    assert comparisons[0]["mean_overhead_pct"] == 20.0

    markdown = render_markdown_report(
        {
            "project_ref": "snipara",
            "base_url": "https://api.snipara.com/mcp",
            "companion_bin": "rlm-hook",
            "runs": 3,
            "generated_at": "2026-04-24T00:00:00+00:00",
            "cases": cases,
            "comparisons": comparisons,
        }
    )

    assert "# Companion Benchmark" in markdown
    assert "direct_code_callers" in markdown
    assert "code_callers" in markdown
    assert "20.0%" in markdown
