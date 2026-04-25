# Multi-stage build for the Daytrader NiceGUI app.
#
# Stage 1 builds a venv with all runtime deps. Stage 2 carries only the
# venv + the app source, so the final image stays slim and the build
# step's apt cache doesn't bloat what we ship.
FROM python:3.13-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# build-essential is needed for argon2-cffi + a couple of ML deps;
# libgomp1 is XGBoost's OpenMP runtime (kept in the final image too).
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install .


FROM python:3.13-slim

# Build-time provenance — CI passes these as --build-arg so the running
# container can report which commit it's running. Default "dev" for
# plain `docker build` without args (local testing).
ARG APP_VERSION=dev
ARG APP_BUILT_AT=unknown

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    APP_HOST=0.0.0.0 \
    APP_PORT=8080 \
    APP_VERSION=${APP_VERSION} \
    APP_BUILT_AT=${APP_BUILT_AT}

WORKDIR /app

# postgresql-client gives us pg_dump for the backup script and psql for
# the entrypoint readiness check. libgomp1 stays for XGBoost runtime.
RUN apt-get update \
 && apt-get install -y --no-install-recommends postgresql-client libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY src ./src
COPY pyproject.toml alembic.ini README.md ./
COPY scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh \
 && mkdir -p /app/data

EXPOSE 8080

ENTRYPOINT ["/entrypoint.sh"]
