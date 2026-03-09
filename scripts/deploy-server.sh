#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
APP_DIR="$(cd "${REPO_DIR}/.." && pwd)"

ENV_FILE="${RUNTIME_ENV_FILE:-${APP_DIR}/.env}"
COMPOSE_FILE="${REPO_DIR}/deploy/compose.prod.yml"
HEALTHCHECK_URL="${HEALTHCHECK_URL:-http://127.0.0.1:7865/}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing runtime env file: ${ENV_FILE}" >&2
  exit 1
fi

cd "${REPO_DIR}"
docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" up -d --build --remove-orphans
curl -fsS "${HEALTHCHECK_URL}" > /dev/null

echo "Service deploy completed."
