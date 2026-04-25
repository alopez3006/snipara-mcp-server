# Snipara Backend Mirror Architecture

## Role Of This Folder

- `snipara-fastapi/` in this monorepo mirrors the standalone backend repository `Snipara/snipara-server`.
- The production source of truth remains `/Users/alopez/Devs/Snipara/apps/mcp-server/`.
- This mirror exists to keep backend code, tests, and packaging surfaces aligned with the public repo.

## Runtime Topology

```text
snipara-mcp (local client)
  -> api.snipara.com
  -> FastAPI backend on Infomaniak VPS
  -> Vaultbrix PostgreSQL + Upstash Redis
```

## Key Paths

```text
snipara-fastapi/
├── src/                 # Mirror of apps/mcp-server/src/
├── prisma/              # Mirror of apps/mcp-server/prisma/
├── tests/               # Mirror of apps/mcp-server/tests/
├── snipara-mcp/         # Mirror of the client package
└── mcp-stdio-bridge/    # Mirror of the bridge entrypoint
```

## Sync Workflow

Refresh the mirror from the monorepo source:

```bash
uv run --project apps/mcp-server python apps/mcp-server/scripts/sync_snipara_fastapi_mirror.py
uv run --project apps/mcp-server python apps/mcp-server/scripts/sync_snipara_fastapi_mirror.py --check
```

## Deployment Truth

- Production does not deploy from old PaaS configs anymore.
- Production rollout is driven from the monorepo with:

```bash
./deploy/infomaniak/deploy-zero-downtime.sh backend
```

- The backend code synced on the VPS lives in `/opt/snipara/mcp-backend`.
- Hosted MCP verification is done with:

```bash
python deploy/infomaniak/check_hosted_mcp.py
```

## Packaging Notes

- `snipara-mcp/` remains the installable PyPI thin client.
- Prisma schema changes still need to stay aligned between JS and Python schemas.
- When changing backend code here, keep the standalone `Snipara/snipara-server` repo in sync.
