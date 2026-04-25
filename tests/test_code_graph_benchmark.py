"""Tests for the deterministic code graph benchmark helpers."""

from benchmarks.code_graph_benchmark import (
    compute_percentile,
    default_cases,
    evaluate_success,
    render_markdown_report,
)


def test_default_cases_cover_context_and_code_tools():
    cases = default_cases()

    tool_names = {case.tool_name for case in cases}
    case_ids = {case.id for case in cases}

    assert "rlm_context_query" in tool_names
    assert "rlm_code_callers" in tool_names
    assert "rlm_code_imports" in tool_names
    assert "rlm_code_neighbors" in tool_names
    assert "rlm_code_shortest_path" in tool_names
    assert "hybrid_context_neighbors_doc_first" in case_ids


def test_compute_percentile_handles_small_samples():
    values = [900, 1100, 1000]

    assert compute_percentile(values, 0.5) == 1000
    assert compute_percentile(values, 0.95) == 1100
    assert compute_percentile([], 0.5) == 0


def test_evaluate_success_requires_all_fragments():
    payload = '{"graph_hybrid_used": true, "recommended_tool": "rlm_code_callers"}'

    assert evaluate_success(payload, ("graph_hybrid_used", "rlm_code_callers")) is True
    assert evaluate_success(payload, ("graph_hybrid_used", "src.rlm_engine.RLMEngine.run")) is False


def test_evaluate_success_supports_doc_first_hybrid_cases():
    payload = (
        '{"graph_hybrid_used": true, "recommended_tool": "rlm_code_neighbors", '
        '"sections": [{"title": "Request Handler Flow"}, {"title": "Code Graph: Neighborhood of '
        'src.rlm_engine.RLMEngine._handle_context_query"}]}'
    )

    assert evaluate_success(
        payload,
        ("graph_hybrid_used", "rlm_code_neighbors", "Code Graph: Neighborhood of"),
    ) is True


def test_render_markdown_report_summarizes_cases():
    report = {
        "project_ref": "snipara",
        "base_url": "https://api.snipara.com/mcp",
        "runs": 3,
        "generated_at": "2026-04-23T00:00:00+00:00",
        "cases": [
            {
                "id": "code_callers",
                "tool_name": "rlm_code_callers",
                "success_rate": 1.0,
                "latency_ms": {"p50": 1000, "p95": 1100},
                "response_bytes": {"mean": 2048},
            }
        ],
        "runtime_probe": {"attempted": False, "reason": "OPENAI_API_KEY is not set"},
    }

    markdown = render_markdown_report(report)

    assert "# Code Graph Benchmark" in markdown
    assert "code_callers" in markdown
    assert "100%" in markdown
    assert "Skipped: OPENAI_API_KEY is not set" in markdown
