#!/bin/sh
set -eu

PROJECT_DIR=${PROJECT_DIR:-/opt/aegispro}
BACKUP_ROOT=${BACKUP_ROOT:-/var/backups/aegispro}
RETENTION_DAYS=${RETENTION_DAYS:-30}
COMPOSE="docker compose --env-file ${PROJECT_DIR}/.env.production -f ${PROJECT_DIR}/docker-compose.yml -f ${PROJECT_DIR}/docker-compose.prod.yml"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
DEST="${BACKUP_ROOT}/${STAMP}"

mkdir -p "${DEST}"

${COMPOSE} exec -T postgres sh -c \
  'pg_dump --format=custom --no-owner --username="$POSTGRES_USER" "$POSTGRES_DB"' \
  > "${DEST}/database.dump"

tar -C "${PROJECT_DIR}" -czf "${DEST}/storage.tar.gz" storage
cp "${PROJECT_DIR}/.env.production" "${DEST}/environment.snapshot"
chmod 600 "${DEST}/environment.snapshot"

(cd "${DEST}" && sha256sum database.dump storage.tar.gz environment.snapshot > SHA256SUMS)
find "${BACKUP_ROOT}" -mindepth 1 -maxdepth 1 -type d -mtime "+${RETENTION_DAYS}" -exec rm -rf -- {} +

printf 'AegisPro backup completed: %s\n' "${DEST}"
