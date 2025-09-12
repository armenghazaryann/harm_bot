-- Initialize extensions for the RAG ETL database
-- This script is executed automatically by the official Postgres entrypoint
-- when mounted at /docker-entrypoint-initdb.d and the data directory is empty.

-- Enable pgvector for vector similarity search (required for VECTOR type and HNSW indexes)
CREATE EXTENSION IF NOT EXISTS vector;

-- Optional: Enable pg_trgm for text similarity operations (useful for FTS/ranking)
-- CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Optional: Enable uuid-ossp if you rely on DB-side UUID generation
-- CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
