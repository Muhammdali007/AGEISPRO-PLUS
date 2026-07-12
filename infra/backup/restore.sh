#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 /var/backups/aegispro/<timestamp>" >&2
  exit 2
fi

PROJECT_DIR=${PROJECT_DIR:-/opt/aegispro}
BACKUP_DIR=$1
COMPOSE="docker compose --env-file ${PROJECT_DIR}/.env.production -f ${PROJECT_DIR}/docker-compose.yml -f ${PROJECT_DIR}/docker-compose.prod.yml"

test -f "${BACKUP_DIR}/database.dump"
test -f "${BACKUP_DIR}/storage.tar.gz"
(cd "${BACKUP_DIR}" && sha256sum -c SHA256SUMS)

echo "This replaces the current database and storage. Set CONFIRM_RESTORE=yes to continue."
test "${CONFIRM_RESTORE:-no}" = "yes"

${COMPOSE} stop api ai web
${COMPOSE} exec -T postgres sh -c \
  'dropdb --if-exists --username="$POSTGRES_USER" "$POSTGRES_DB" &&
   createdb --username="$POSTGRES_USER" "$POSTGRES_DB"'
${COMPOSE} exec -T postgres sh -c \
  'pg_restore --no-owner --username="$POSTGRES_USER" --dbname="$POSTGRES_DB"' \
  < "${BACKUP_DIR}/database.dump"

rm -rf "${PROJECT_DIR}/storage"
tar -C "${PROJECT_DIR}" -xzf "${BACKUP_DIR}/storage.tar.gz"
${COMPOSE} up -d

echo "Restore completed. Verify health, authentication, incidents, and evidence playback."
