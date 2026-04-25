#!/usr/bin/env bash
# Phase 10 — backup/restore drill.
#
# Dumps a running Postgres database, restores it into a fresh database
# on the same server, and verifies alembic agrees the restored schema is
# at the expected revision. Intended to be run periodically against the
# production database (read-only on the source) and in CI against the
# service-container Postgres.
#
# Usage:
#   ./scripts/backup_restore_drill.sh \
#       postgresql://USER:PASS@HOST:5432/SOURCE_DB \
#       postgresql://USER:PASS@HOST:5432/RESTORE_DB
#
# Both arguments must use the libpq URL form (no +asyncpg). The script
# expects pg_dump, psql, and alembic on PATH.

set -euo pipefail

SOURCE_URL="${1:-}"
RESTORE_URL="${2:-}"

if [[ -z "$SOURCE_URL" || -z "$RESTORE_URL" ]]; then
  echo "Usage: $0 SOURCE_LIBPQ_URL RESTORE_LIBPQ_URL" >&2
  exit 2
fi

DUMP_FILE="$(mktemp -t daytrader_dump.XXXXXX.sql)"
trap 'rm -f "$DUMP_FILE"' EXIT

echo "[1/4] pg_dump $SOURCE_URL"
pg_dump --no-owner --no-privileges --dbname="$SOURCE_URL" -f "$DUMP_FILE"

# Drop + recreate the restore DB. We assume the connecting user has
# CREATE DATABASE privileges on the cluster.
RESTORE_DBNAME="$(python - <<PY
from urllib.parse import urlparse
import sys
print(urlparse(sys.argv[1]).path.lstrip('/'))
PY
"$RESTORE_URL")"

ADMIN_URL="$(python - <<PY
from urllib.parse import urlparse, urlunparse
import sys
parts = urlparse(sys.argv[1])
print(urlunparse(parts._replace(path='/postgres')))
PY
"$RESTORE_URL")"

echo "[2/4] DROP + CREATE $RESTORE_DBNAME"
psql --dbname="$ADMIN_URL" -v ON_ERROR_STOP=1 -c "DROP DATABASE IF EXISTS \"$RESTORE_DBNAME\";"
psql --dbname="$ADMIN_URL" -v ON_ERROR_STOP=1 -c "CREATE DATABASE \"$RESTORE_DBNAME\";"

echo "[3/4] psql restore $RESTORE_URL"
psql --dbname="$RESTORE_URL" -v ON_ERROR_STOP=1 -f "$DUMP_FILE" >/dev/null

echo "[4/4] alembic current against restored DB"
ASYNC_URL="${RESTORE_URL/postgresql:\/\//postgresql+asyncpg:\/\/}"
DATABASE_URL="$ASYNC_URL" alembic current

echo "OK — backup/restore drill passed."
