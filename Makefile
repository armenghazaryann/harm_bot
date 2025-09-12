SHELL := /bin/bash

PY ?= uv run
UV ?= uv
DOCKER_EXEC_LOCAL=docker exec rag_api
NONE_IMAGES=docker images -f "dangling=true" -q
COMPOSE_FILE := docker-compose.yml

export PYTHONUNBUFFERED=1

.PHONY: help install install-dev api worker flower migrate revision compose-up compose-down fmt test build run stop docker-migrate docker-revision kill-ports

help:
	@echo "Targets:"
	@echo "  install         - install base deps"
	@echo "  install-dev     - install dev + pdf extras"
	@echo "  api             - run FastAPI server"
	@echo "  worker          - run Celery worker"
	@echo "  flower          - run Flower UI"
	@echo "  migrate         - Alembic upgrade head"
	@echo "  revision m=msg  - Alembic autogenerate revision"
	@echo "  compose-up      - start Postgres, Redis, MinIO"
	@echo "  compose-down    - stop stack"
	@echo "  build           - build Docker images"
	@echo "  run             - run with Docker Compose"
	@echo "  stop            - stop Docker Compose"
	@echo "  docker-migrate  - run migrations in Docker"
	@echo "  docker-revision - create revision in Docker"
	@echo "  fmt             - format & lint"
	@echo "  test            - run tests"
	@echo "  kill-ports      - kill processes on common dev ports or use PORTS=\"8000 5432\" make kill-ports"


install:
	$(UV) pip install -e .

install-dev:
	$(UV) pip install -e .[dev,pdf]

api:
	$(PY) uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

worker:
	$(PY) celery -A workers.celery_app.app worker --loglevel=info --queues=ingest,chunk,embed,index,eval

flower:
	$(PY) celery -A workers.celery_app.app flower --port=5555

migrate:
	$(PY) alembic upgrade head

revision:
	$(PY) alembic revision --autogenerate -m "$(m)"

compose-up:
	docker compose up -d

compose-down:
	docker compose down

build:
	docker compose -f ${COMPOSE_FILE} build
	@DANGLING=$$(${NONE_IMAGES}) && [ -n "$$DANGLING" ] && docker rmi $$DANGLING -f || echo "No dangling images to remove."

run:
	docker compose -f ${COMPOSE_FILE} --env-file .env up

stop:
	docker compose -f ${COMPOSE_FILE} down

docker-migrate:
	@echo "Starting database migration..."
	@if ! docker ps | grep -q rag_api; then \
		echo "Error: 'rag_api' container is not running. Please start the containers first with 'make run'"; \
		exit 1; \
	fi
	${DOCKER_EXEC_LOCAL} alembic upgrade head

docker-revision:
	docker compose --env-file .env run --rm api alembic revision --autogenerate -m "$(m)"

fmt:
	$(PY) ruff --fix .
	$(PY) black .

test:
	$(PY) pytest -q

kill-ports:
	@chmod +x ./kill-port.sh || true
	@./kill-port.sh $(PORTS)
