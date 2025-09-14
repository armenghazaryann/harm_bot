# Experiment RAG ETL

## Overview

This repository implements a **Retrieval‑Augmented Generation (RAG) ETL pipeline** for processing large collections of PDFs, CSVs, and plain text. The system extracts content, chunks it, generates embeddings via OpenAI, stores vectors, and serves query endpoints that combine keyword search, vector similarity, and optional reranking with Jina AI. It is designed to start as a lean MVP and grow into a production‑grade, multi‑tenant service.

## Setup

1. **Environment** – copy the example environment file and edit as needed:
   ```bash
   cp .env.example .env
   ```
2. **Install dependencies** – install the base Python packages:
   ```bash
   make install
   ```
   For development with PDF extras, run:
   ```bash
   make install-dev
   ```
3. **Build Docker images** (optional, if you want to use Docker for the full stack):
   ```bash
   make build
   ```
4. **Run the application** – start the required services and the API:
   ```bash
   make run          # launch FastAPI, workers, and Streamlit UI
   ```
   You can also run the API locally without Docker:
   ```bash
   make api
   ```
   And start the Celery worker and Flower UI separately if needed:
   ```bash
   make worker
   make flower
   ```

## Quick Start

*See the **Setup** section above for detailed steps to get the project running.*

## Makefile Targets

- `make install` – install base dependencies.
- `make install-dev` – install development extras (including PDF support).
- `make compose-up` – start the Docker services (Postgres, MinIO, Redis).
- `make compose-down` – stop the Docker services.
- `make build` – build Docker images.
- `make run` – run the full stack with Docker Compose.
- `make stop` – stop Docker Compose.
- `make api` – run FastAPI locally (outside Docker).
- `make worker` – run Celery worker locally.
- `make flower` – run Flower UI locally.
- `make migrate` – apply Alembic migrations.
- `make revision m="msg"` – create a new Alembic revision.
- `make docker-migrate` – run migrations inside the Docker container.
- `make docker-revision` – create Alembic revision inside Docker.
- `make docker-downgrade` – downgrade the database inside Docker.
- `make fmt` – format code with Ruff and Black.
- `make test` – run the test suite.
- `make kill-ports` – kill processes on common dev ports.

## Architecture

See the detailed component diagram and data flow in **[ARCHITECTURE.md](ARCHITECTURE.md)**.

## Scaling

Guidance for scaling the system to handle thousands of concurrent uploads and multi‑TB corpora is provided in **[SCALING.md](SCALING.md)**.

## Development

Common make targets:
- `make test` – run the test suite
- `make fmt` – run linters and formatters
- `make compose-down` – stop all containers

## License

MIT License. See `LICENSE` for details.
