#!/bin/bash
set -e

echo "Running prisma generate..."
prisma generate

echo "Starting uvicorn..."
exec uvicorn src.server:app --host 0.0.0.0 --port 8000
