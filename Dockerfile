FROM python:3.13-slim AS builder
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV VIRTUAL_ENV=/opt/venv

# Install system dependencies for PDF processing
RUN apt-get update && apt-get install -y \
    ghostscript \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
WORKDIR /app
COPY pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
     uv venv /opt/venv && \
     uv pip install -e .[pdf]
ADD . /app

FROM python:3.13-slim
WORKDIR /app

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    ghostscript \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ADD . .
CMD ["uvicorn", "--lifespan=on", "api.main:app", "--host", "0.0.0.0", "--workers", "1", "--port", "8000"]
