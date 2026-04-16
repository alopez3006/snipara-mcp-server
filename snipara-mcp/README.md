# Snipara MCP Server

[![PyPI version](https://badge.fury.io/py/snipara-mcp.svg)](https://pypi.org/project/snipara-mcp/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

MCP server for [Snipara](https://snipara.com) - Context optimization and Agent infrastructure for LLMs.

**Two Products in One:**

- **Snipara** - Context optimization with 90% token reduction
- **Snipara Agents** - Multi-agent memory, swarms, and coordination

The stdio package keeps full parity with the hosted backend contract. The packaged tool surface is generated from `apps/mcp-server/src/mcp/tool_defs.py`.

Works with any MCP-compatible client including Claude Desktop, Cursor, Windsurf, Claude Code, Gemini, GPT, and more.

**LLM-agnostic**: Snipara optimizes context delivery - you use your own LLM (Claude, GPT, Gemini, Llama, etc.).

## Installation

### Option 1: uvx (Recommended - No Install)

```bash
uvx snipara-mcp
```

### Option 2: pip

```bash
pip install snipara-mcp
```

### Option 3: With RLM Runtime Integration

```bash
pip install snipara-mcp[rlm]
```

This installs `rlm-runtime` as a dependency, enabling programmatic access to Snipara tools within the RLM orchestrator.

## Configuration

### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "snipara": {
      "command": "uvx",
      "args": ["snipara-mcp"],
      "env": {
        "SNIPARA_API_KEY": "sk-your-api-key",
        "SNIPARA_PROJECT_ID": "your-project-id"
      }
    }
  }
}
```

### Cursor

Add to `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "snipara": {
      "command": "uvx",
      "args": ["snipara-mcp"],
      "env": {
        "SNIPARA_API_KEY": "sk-your-api-key",
        "SNIPARA_PROJECT_ID": "your-project-id"
      }
    }
  }
}
```

### Claude Code

```bash
claude mcp add snipara -- uvx snipara-mcp
```

Then set environment variables in your shell or `.env` file.

### Windsurf

Add to `~/.codeium/windsurf/mcp_config.json`:

```json
{
  "mcpServers": {
    "snipara": {
      "command": "uvx",
      "args": ["snipara-mcp"],
      "env": {
        "SNIPARA_API_KEY": "sk-your-api-key",
        "SNIPARA_PROJECT_ID": "your-project-id"
      }
    }
  }
}
```

## Quick Setup (Recommended)

### Option A: Initialize in Your Project (New!)

The fastest way to get started — run `snipara-init` in your project directory:

```bash
# Install
pip install snipara-mcp

# Initialize Snipara in your project
snipara-init
```

**What happens:**

1. Detects your project type (Node.js, Python, Go, Rust, Java)
2. Extracts project slug from git remote (or uses directory name)
3. Creates `.mcp.json` with Snipara server configuration
4. Adds `SNIPARA_API_KEY` to `.env.example`
5. Uploads CLAUDE.md, README.md, and docs/\*.md (if authenticated)
6. Tests API connection

**Options:**

```bash
snipara-init                    # Auto-detect and initialize
snipara-init --slug my-project  # Use specific slug
snipara-init --dry-run          # Preview what would be done
snipara-init --no-upload        # Skip doc upload
snipara-init --skip-test        # Skip connection test
```

### Option B: Device Flow Login

Alternatively, sign in via browser with `snipara-mcp-login`. A free account and project are created automatically if you don't have one.

```bash
# Install
pip install snipara-mcp

# Sign in (opens browser, auto-creates account + project)
snipara-mcp-login
```

**What happens:**

1. The CLI opens your browser to the Snipara authorization page (code pre-filled in URL)
2. Sign in with GitHub or Google — a free account is created automatically if needed
3. Select your project and click **Authorize**
4. Return to your terminal — the CLI receives the token automatically (no copying needed)
5. The CLI prints a `.mcp.json` snippet with your API key and MCP endpoint

Tokens are stored securely in `~/.snipara/tokens.json`.

### CLI Commands

| Command              | Description                                               |
| -------------------- | --------------------------------------------------------- |
| `snipara-init`       | Initialize Snipara in current project (creates .mcp.json) |
| `snipara-mcp-login`  | Sign in via browser (auto-creates free account + project) |
| `snipara-mcp-logout` | Clear all stored tokens                                   |
| `snipara-mcp-status` | Show current auth status and stored tokens                |

## Environment Variables

| Variable             | Required | Description                                |
| -------------------- | -------- | ------------------------------------------ |
| `SNIPARA_API_KEY`    | Yes\*    | Your Snipara API key                       |
| `SNIPARA_PROJECT_ID` | Yes\*    | Your project ID                            |
| `SNIPARA_API_URL`    | No       | API URL (default: https://api.snipara.com) |

\* Not required if you use `snipara-mcp-login` (OAuth tokens from `~/.snipara/tokens.json` are used automatically).

Get your API key and project ID from [snipara.com/dashboard](https://snipara.com/dashboard) or run `snipara-mcp-login` for automatic setup.

## Available Tools

The current stdio surface includes:

- Retrieval and query tools such as `rlm_context_query`, `rlm_ask`, `rlm_search`, `rlm_multi_query`, `rlm_plan`, `rlm_get_chunk`, `rlm_load_document`, and `rlm_load_project`
- Shared context and template tools such as `rlm_shared_context`, `rlm_list_templates`, `rlm_get_template`, `rlm_list_collections`, and `rlm_upload_shared_document`
- Summary and memory automation tools such as `rlm_store_summary`, `rlm_remember_if_novel`, `rlm_end_of_task_commit`, `rlm_memory_compact`, `rlm_journal_append`, and `rlm_tenant_profile_get`
- Swarm and coordination tools such as `rlm_swarm_create`, `rlm_claim`, `rlm_state_poll`, `rlm_task_bulk_create`, `rlm_task_reassign`, and `rlm_agent_status`
- Hierarchical task tools such as `rlm_htask_create_feature`, `rlm_htask_tree`, `rlm_htask_recommend_batch`, `rlm_htask_policy_update`, and `rlm_htask_audit_trail`
- Decision and operational tools such as `rlm_decision_create`, `rlm_index_health`, `rlm_search_analytics`, `rlm_query_trends`, and `rlm_request_access`

For the full current list of tools, required parameters, optional parameters, and descriptions, see the generated [MCP Tool Index](https://github.com/Snipara/snipara-server/blob/main/docs/reference/MCP_TOOL_INDEX.md).

## Example Usage

Once configured, ask your LLM:

> "Use snipara to find how authentication works in my codebase"

The LLM will call `rlm_context_query` and return relevant documentation sections.

### Agent Memory Example

> "Remember that the user prefers TypeScript over JavaScript"

> "What do you remember about the user's preferences?"

### Multi-Agent Swarm Example

> "Create a swarm called 'refactoring-team' for coordinating the auth refactor"

> "Claim the file src/auth.ts so other agents don't modify it"

> "Create a task to update the login flow, depending on the token-refresh task"

## Alternative: Direct HTTP (No Local Install)

For clients that support HTTP transport (Claude Code, Cursor v0.48+), you can connect directly without installing anything:

**Claude Code:**

```json
{
  "mcpServers": {
    "snipara": {
      "type": "http",
      "url": "https://api.snipara.com/mcp/YOUR_PROJECT_ID",
      "headers": {
        "Authorization": "Bearer sk-your-api-key"
      }
    }
  }
}
```

## CI/CD Integration

Sync docs automatically on git push using the webhook endpoint:

```bash
curl -X POST "https://api.snipara.com/v1/YOUR_PROJECT_ID/webhook/sync" \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"documents": [{"path": "CLAUDE.md", "content": "..."}]}'
```

See [GitHub Action example](https://github.com/Snipara/snipara-server#github-action-example) for automated sync on push.

## Development and Validation

The packaged MCP contract is generated from the hosted backend source of truth at `apps/mcp-server/src/mcp/tool_defs.py`.

When backend MCP tools change, regenerate the packaged contract before committing:

```bash
uv run --project apps/mcp-server python scripts/sync_snipara_mcp_contract.py
```

For deterministic local validation, use Python 3.11, which matches CI:

```bash
cd apps/mcp-server/snipara-mcp
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
ruff check src/snipara_mcp tests
pytest tests -v --tb=short
python -m build
```

Repository CI now validates both:

- backend MCP tests in `apps/mcp-server/tests/`
- package-local lint, tests, and wheel/sdist builds in `apps/mcp-server/snipara-mcp/`

## Upgrading

When a new version is released on PyPI, follow these steps to get the latest tools:

### 1. Clear the uvx cache

```bash
# macOS/Linux
rm -rf ~/.cache/uv/tools/snipara-mcp
rm -rf ~/Library/Caches/uv/tools/snipara-mcp

# Windows
rmdir /s %LOCALAPPDATA%\uv\tools\snipara-mcp
```

### 2. Restart your MCP client

MCP tool definitions are loaded at startup. You **must restart** Claude Desktop, Cursor, Claude Code, or your MCP client to load the new tools.

### 3. Verify the version

After restart, the new tools should be available. You can check by asking:

> "Use snipara to show settings"

If `rlm_settings` works, you have the latest version.

### Important: Use uvx, not local Python

Always configure with `uvx` to get automatic updates from PyPI:

```json
{
  "command": "uvx",
  "args": ["snipara-mcp"]
}
```

**Do NOT use local Python paths** like:

```json
{
  "command": "/usr/bin/python3",
  "args": ["-m", "snipara_mcp"],
  "env": { "PYTHONPATH": "/local/path" }
}
```

This bypasses PyPI and you won't get updates.

## Troubleshooting

### MCP tools not showing up

1. **Restart your MCP client** - Tool definitions are cached at startup
2. **Clear uvx cache** - Old version may be cached (see Upgrading section)
3. **Check config syntax** - Ensure valid JSON in your MCP config file

### "Invalid API key" error

- Verify your API key is correct in the dashboard
- Check the key hasn't been rotated
- Ensure no extra whitespace in the config

### MCP server not connecting

- Check that `uvx` is installed: `which uvx` or `uvx --version`
- Install uv if missing: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Check Claude Code output panel for connection errors

## RLM Runtime Integration (New in 1.4.0)

Snipara MCP can be used as a tool provider for [rlm-runtime](https://github.com/Snipara/rlm-runtime), enabling LLMs to query your documentation during autonomous code execution.

### Installation

```bash
pip install snipara-mcp[rlm]
```

### Usage with RLM Runtime

```python
from rlm import RLM

# Snipara tools are auto-registered when credentials are set
rlm = RLM(
    model="claude-sonnet-4-20250514",
    snipara_api_key="rlm_your_key",
    snipara_project_slug="your-project"
)

# The LLM can now query your docs during execution
result = rlm.run("Implement the auth flow following our coding standards")
```

### Manual Tool Registration

```python
from snipara_mcp import get_snipara_tools

# Get tools as RLM-compatible Tool objects
tools = get_snipara_tools(
    api_key="rlm_your_key",
    project_slug="your-project"
)

# Register with RLM
from rlm import RLM
rlm = RLM(model="claude-sonnet-4-20250514", tools=tools)
```

### Available Tools (Programmatic API)

When using `get_snipara_tools()`, the programmatic wrappers now expose the full hosted `rlm_*` contract generated from `tool_contract.py`, while preserving the legacy high-level aliases defined in `src/snipara_mcp/rlm_tools.py` (`context_query`, `remember`, `task_create`, etc.).

See [`src/snipara_mcp/rlm_tools.py`](./src/snipara_mcp/rlm_tools.py) for the exact wrapper surface and signatures.

### CLI Tool Introspection

The `snipara` CLI also exposes generic MCP inspection helpers:

```bash
snipara tools list --slug your-project
snipara tools call rlm_help --slug your-project --args '{"query":"session automation"}'
```

### Environment Variables

```bash
export SNIPARA_API_KEY="rlm_your_key"
export SNIPARA_PROJECT_SLUG="your-project"
export SNIPARA_API_URL="https://api.snipara.com"  # Optional
```

## Version History

| Version | Date       | Changes                                                |
| ------- | ---------- | ------------------------------------------------------ |
| 2.6.1   | 2026-04-16 | Contract parity, package validation, mirror hardening  |
| 2.4.0   | 2026-02-11 | Add `snipara-init` CLI for project initialization      |
| 2.3.1   | 2026-01-31 | Fix device flow CLI: remove misleading code entry step |
| 1.8.1   | 2025-01-25 | Add multi_project_query for cross-project search       |
| 1.8.0   | 2025-01-25 | Full tool parity with FastAPI server (21 new tools)    |
| 1.7.6   | 2025-01-24 | Fix Redis URL protocol support, graceful env handling  |
| 1.7.5   | 2025-01-23 | CI/CD improvements, production environment secrets     |
| 1.7.1   | 2025-01-22 | OAuth device flow fixes                                |
| 1.7.0   | 2025-01-21 | OAuth device flow authentication (`snipara-mcp-login`) |
| 1.6.0   | 2025-01-20 | Agent Memory and Multi-Agent Swarms (14 new tools)     |
| 1.5.0   | 2025-01-18 | Auto-inject Snipara usage instructions                 |
| 1.4.0   | 2025-01-15 | RLM Runtime integration                                |
| 1.3.0   | 2025-01-10 | Shared Context tools (Team+)                           |
| 1.2.0   | 2025-01-05 | Document upload and sync tools                         |
| 1.1.0   | 2024-12-20 | Session context management                             |
| 1.0.0   | 2024-12-15 | Initial release with core context optimization         |

## Support

- Website: [snipara.com](https://snipara.com)
- Issues: [github.com/Snipara/snipara-server/issues](https://github.com/Snipara/snipara-server/issues)
- Email: support@starbox-group.com

## License

MIT
