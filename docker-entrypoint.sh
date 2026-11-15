#!/bin/sh
# Prepare the SQLite data directory (a mounted volume may be root-owned),
# then drop privileges to the unprivileged "nobody" account.
set -e

db_path="${CATALOG_DB:-/data/catalog.db}"
data_dir=$(dirname "$db_path")
mkdir -p "$data_dir"

if [ "$(id -u)" = "0" ]; then
    chown -R nobody:nogroup "$data_dir"
    exec setpriv --reuid=nobody --regid=nogroup --clear-groups "$@"
fi

exec "$@"
