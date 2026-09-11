# syntax=docker/dockerfile:1
# Multi-stage image for the rag_service workers (non-root, uv-based).

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY src ./src
COPY alembic.ini ./
COPY migrations ./migrations
RUN uv sync --frozen --no-dev


FROM python:3.12-slim-bookworm

RUN groupadd --system app && useradd --system --gid app --create-home app

WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/src /app/src

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

USER app

# sentence-transformers stores the downloaded model in ~/.cache
# (already writable by the app user because --create-home was used).

CMD ["python", "src/main.py"]