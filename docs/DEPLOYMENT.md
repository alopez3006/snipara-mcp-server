# Snipara Backend Deployment

## Overview

`snipara-fastapi` is the production backend mirror for Snipara's FastAPI MCP server.

The current production topology is:

- `Snipara/snipara` monorepo
  - source of truth for `apps/mcp-server/src`
  - source of truth for `packages/database/prisma/schema.prisma`
  - contains the Infomaniak deployment scripts in `deploy/infomaniak/`
- `Snipara/snipara-server` (this repo)
  - backend deployment mirror
  - receives synchronized FastAPI backend code and Python Prisma schema
  - used as the backend build context for the Infomaniak VPS deployment
- Production runtime
  - `api.snipara.com` runs on the Infomaniak VPS in Docker behind Traefik
  - PostgreSQL is hosted on Vaultbrix
  - Redis is hosted on Upstash

Railway is no longer the production deployment target for this backend.

## Deployment Flow

### 1. Make backend changes in the monorepo

Edit the backend in:

- `/Users/alopez/Devs/Snipara/apps/mcp-server/src/`
- `/Users/alopez/Devs/Snipara/apps/mcp-server/prisma/schema.prisma`
- `/Users/alopez/Devs/Snipara/packages/database/prisma/schema.prisma`

### 2. Keep Prisma schema in sync

The Prisma source of truth remains:

- `/Users/alopez/Devs/Snipara/packages/database/prisma/schema.prisma`

Before backend deployment:

1. sync the Python schema used by the backend
2. regenerate the Python Prisma client
3. apply schema changes to Vaultbrix if needed

Example:

```bash
cd /Users/alopez/Devs/Snipara
./scripts/sync-python-prisma-schema.py --target /Users/alopez/Devs/snipara-fastapi/prisma/schema.prisma
npx -y prisma@5.17.0 generate --schema /Users/alopez/Devs/snipara-fastapi/prisma/schema.prisma
```

### 3. Sync backend code into this repo

Copy the backend files from the monorepo into this deployment repo.

Typical files:

- `src/config.py`
- `src/rlm_engine.py`
- `src/mcp/tool_defs.py`
- `src/engine/handlers/*`
- `src/models/*`
- `src/services/*`
- `prisma/schema.prisma`

### 4. Validate locally

Recommended checks:

```bash
python3 -m py_compile src/config.py src/rlm_engine.py src/services/agent_memory.py
python3 -m py_compile src/models/*.py src/services/*.py
```

For memory changes, also run a real smoke test against Vaultbrix with a minimal environment.

### 5. Commit and push this mirror repo

```bash
cd /Users/alopez/Devs/snipara-fastapi
git add -A
git commit -m "feat(memory): sync backend changes from monorepo"
git push origin main
```

### 6. Deploy production on Infomaniak

Production deployment is triggered from the monorepo:

```bash
cd /Users/alopez/Devs/Snipara/deploy/infomaniak
./deploy-zero-downtime.sh backend
```

That script:

- syncs the web and/or backend build contexts to the VPS
- builds fresh Docker images on the VPS
- performs a rolling update for `mcp-backend`
- verifies `https://api.snipara.com/health`

## Current Production Infrastructure

| Component | Runtime |
| --- | --- |
| Web app | Infomaniak VPS + Docker + Traefik |
| FastAPI backend | Infomaniak VPS + Docker + Traefik |
| Database | Vaultbrix PostgreSQL |
| Cache / rate limit | Upstash Redis |
| Client package | PyPI |

## Operational Notes

- Production deploys are manual and explicit.
- A Git push alone does not deploy production.
- The backend repo is still important, but it is a mirror and build source, not the place where production orchestration lives.
- The deployment scripts under `/Users/alopez/Devs/Snipara/deploy/infomaniak/` are the operational source of truth.
