#!/usr/bin/env bash
# Export the dasanyuan database to one file (Mac / Linux).
#
#   bash db/export_db.sh                 # writes dasanyuan.dump
#   bash db/export_db.sh --hands-only    # small file, no decisions table
set -euo pipefail

OUT="dasanyuan.dump"
EXTRA=()
for a in "$@"; do
  case "$a" in
    --hands-only) EXTRA+=(--exclude-table-data=decisions) ;;
    *) OUT="$a" ;;
  esac
done

export PGPASSWORD="${PGPASSWORD:-dsy}"
pg_dump -h localhost -U dsy -d dasanyuan \
  --format=custom --no-owner --no-privileges --file="$OUT" "${EXTRA[@]+"${EXTRA[@]}"}"
echo "Wrote $OUT ($(du -h "$OUT" | cut -f1))."
