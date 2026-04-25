#!/bin/sh
# Container entrypoint: wait for postgres, run migrations, launch app.
#
# Watchtower replaces the running container on each new image; the new
# container starts here, blocks until the DB is reachable, applies any
# pending migrations, then hands off to the NiceGUI process. A failing
# migration aborts startup so we don't run the new app against a stale
# schema — Watchtower will keep the container in restart-loop until you
# investigate.
set -eu

if [ -z "${DATABASE_URL:-}" ]; then
    echo "[entrypoint] DATABASE_URL not set — refusing to start." >&2
    exit 2
fi

# Parse host + port + user + db from the URL for the readiness probe.
# The URL is of the form postgresql+asyncpg://USER:PASS@HOST:PORT/DB.
# We don't need PASS here — pg_isready uses TCP only.
url="${DATABASE_URL#*//}"
creds_host="${url%%/*}"
db_part="${url#*/}"
db_name="${db_part%%\?*}"
host_port="${creds_host#*@}"
db_host="${host_port%%:*}"
db_port="${host_port#*:}"
[ "$db_port" = "$host_port" ] && db_port=5432
db_user="${creds_host%%:*}"

echo "[entrypoint] waiting for postgres at $db_host:$db_port..."
i=0
until pg_isready -h "$db_host" -p "$db_port" -U "$db_user" -d "$db_name" -q; do
    i=$((i + 1))
    if [ "$i" -ge 60 ]; then
        echo "[entrypoint] postgres not ready after 60s — giving up." >&2
        exit 3
    fi
    sleep 1
done
echo "[entrypoint] postgres ready."

echo "[entrypoint] running alembic upgrade head..."
alembic upgrade head

echo "[entrypoint] launching app (version=${APP_VERSION:-dev}, built=${APP_BUILT_AT:-unknown})"
exec python -m daytrader
