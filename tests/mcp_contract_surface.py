"""Shared MCP contract surfaces for local and remote tests."""

ESSENTIAL_TOOL_SURFACE = {
    "rlm_context_query",
    "rlm_help",
    "rlm_plan",
    "rlm_remember",
    "rlm_remember_if_novel",
    "rlm_end_of_task_commit",
    "rlm_recall",
    "rlm_session_memories",
    "rlm_memory_daily_brief",
    "rlm_tenant_profile_get",
}

CODE_GRAPH_TOOL_SURFACE = {
    "rlm_code_callers",
    "rlm_code_imports",
    "rlm_code_neighbors",
    "rlm_code_shortest_path",
}

INDEX_TOOL_SURFACE = {
    "rlm_index_health",
    "rlm_index_recommendations",
    "rlm_reindex",
}
