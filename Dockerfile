# syntax=docker/dockerfile:1

FROM python:3.14-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --frozen --no-dev --extra server --no-install-project

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra server --no-editable


FROM python:3.14-slim AS runtime

LABEL org.opencontainers.image.source="https://github.com/kweinmeister/gemini-decision-router"

WORKDIR /app

RUN groupadd --system --gid 1001 appgroup && \
    useradd --system --create-home --uid 1001 --gid appgroup appuser

COPY --from=builder --link --chown=1001:1001 /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

USER appuser

EXPOSE 8080

CMD ["sh", "-c", "exec uvicorn gemini_decision_router.server:app --host 0.0.0.0 --port ${PORT:-8080} --no-access-log"]
