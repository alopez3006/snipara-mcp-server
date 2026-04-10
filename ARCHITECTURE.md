# Snipara MCP Architecture

## Repositories

| Repo | Role | Runtime |
| --- | --- | --- |
| `Snipara/snipara` | Main monorepo, product source of truth | Local development + Infomaniak deployment scripts |
| `Snipara/snipara-server` | Backend deployment mirror | Backend build source for Infomaniak VPS |
| `snipara-mcp` (PyPI) | MCP client package | Installed by users locally |

## Deployment Model

Snipara production runs on the Infomaniak VPS.

- `snipara.com` and `www.snipara.com` run in Docker behind Traefik
- `api.snipara.com` runs the FastAPI MCP backend in Docker behind Traefik
- PostgreSQL lives on Vaultbrix
- Redis lives on Upstash

This repo does not directly orchestrate production. The operational deployment entry point is:

- `/Users/alopez/Devs/Snipara/deploy/infomaniak/deploy-zero-downtime.sh`

## Backend Source Flow

The FastAPI backend source of truth is still maintained in the monorepo:

- `/Users/alopez/Devs/Snipara/apps/mcp-server/src/`

This repo receives synchronized backend code before deployment so the backend build context remains isolated and reproducible.

Typical sync flow:

```bash
cd /Users/alopez/Devs/Snipara
./scripts/sync-python-prisma-schema.py --target /Users/alopez/Devs/snipara-fastapi/prisma/schema.prisma

cp apps/mcp-server/src/config.py /Users/alopez/Devs/snipara-fastapi/src/config.py
cp apps/mcp-server/src/rlm_engine.py /Users/alopez/Devs/snipara-fastapi/src/rlm_engine.py
cp apps/mcp-server/src/services/agent_memory.py /Users/alopez/Devs/snipara-fastapi/src/services/agent_memory.py
```

## Runtime Data Flow

```text
LLM client
  -> snipara-mcp / HTTP MCP
  -> api.snipara.com
  -> FastAPI MCP backend
  -> Vaultbrix PostgreSQL + Upstash Redis
```

## Schema Ownership

Prisma schema ownership stays in the monorepo:

- JS source of truth:
  - `/Users/alopez/Devs/Snipara/packages/database/prisma/schema.prisma`
- Python deployment copy:
  - `/Users/alopez/Devs/snipara-fastapi/prisma/schema.prisma`

The backend Python schema must stay aligned with the monorepo schema and use the Python Prisma generator with multi-schema support.

## Production Rule

If you change backend code or schema:

1. update the monorepo source
2. sync this backend mirror
3. validate locally
4. push this repo
5. deploy from the monorepo Infomaniak scripts
