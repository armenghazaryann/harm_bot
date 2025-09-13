-- Initialize extensions for the RAG ETL database
-- This script is executed automatically by the official Postgres entrypoint
-- when mounted at /docker-entrypoint-initdb.d and the data directory is empty.

-- Enable pgvector for vector similarity search (required for VECTOR type and HNSW indexes)
CREATE EXTENSION IF NOT EXISTS vector;

-- Optional: Enable pg_trgm for text similarity operations (useful for FTS/ranking)
-- CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Enable pgcrypto for gen_random_uuid() used by many ORMs / libraries
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Also enable uuid-ossp for uuid_generate_v4() compatibility (some tools prefer it)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
