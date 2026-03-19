-- Migration: Enable pgvector extension
-- Run this on Vaultbrix (tenant_snipara schema)
-- Required for embedding-based semantic search

CREATE EXTENSION IF NOT EXISTS vector;
