#!/usr/bin/env bash
# Load a dasanyuan.dump from a teammate (Mac / Linux).
#
#   bash db/import_db.sh path/to/dasanyuan.dump
#
# Creates the dsy user (password dsy) and the dasanyuan database if they are missing,
# then loads the dump. Running it again replaces the tables with the dump's contents.
# Works with Postgres.app or Homebrew PostgreSQL.
set -euo pipefail

DUMP="${1:-dasanyuan.dump}"
[ -f "$DUMP" ] || { echo "File not found: $DUMP"; exit 1; }

# Postgres.app keeps its tools here and does not always put them on PATH.
for d in /Applications/Postgres.app/Contents/Versions/latest/bin /opt/homebrew/opt/postgresql@*/bin /usr/local/opt/postgresql@*/bin; do
  [ -x "$d/psql" ] && PATH="$d:$PATH"
done
command -v pg_restore >/dev/null || { echo "pg_restore not found. Install Postgres.app or 'brew install postgresql@18'."; exit 1; }

# Admin connection: your Mac user on Postgres.app / Homebrew, or set ADMIN_URL.
ADMIN_URL="${ADMIN_URL:-postgres:///postgres}"
psql "$ADMIN_URL" -v ON_ERROR_STOP=1 -q <<'SQL'
DO $$ BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'dsy') THEN
    CREATE ROLE dsy LOGIN PASSWORD 'dsy';
  END IF;
END $$;
SQL
if ! psql "$ADMIN_URL" -Atc "select 1 from pg_database where datname='dasanyuan'" | grep -q 1; then
  psql "$ADMIN_URL" -q -c "CREATE DATABASE dasanyuan OWNER dsy"
fi

echo "Loading $DUMP into dasanyuan (a minute or two for the full data)..."
PGPASSWORD=dsy pg_restore -h localhost -U dsy -d dasanyuan --clean --if-exists --no-owner --no-privileges "$DUMP"

PGPASSWORD=dsy psql -h localhost -U dsy -d dasanyuan -c \
  "select r.name as run, count(*) as hands from runs r join hands h using (run_id) group by r.name order by r.name"
echo "Done. Connect with: postgresql://dsy:dsy@localhost:5432/dasanyuan"
