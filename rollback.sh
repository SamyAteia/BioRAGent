#!/usr/bin/env bash
set -euo pipefail

TARGET_SHA="${1:-}"
if [[ -z "${TARGET_SHA}" ]]; then
  echo "Usage: ./rollback.sh <commit-sha>"
  exit 1
fi

if ! [[ "${TARGET_SHA}" =~ ^[0-9a-fA-F]{7,40}$ ]]; then
  echo "Invalid commit SHA format: ${TARGET_SHA}" >&2
  exit 1
fi

DEPLOY_USER="${DEPLOY_USER:-samy}"
DEPLOY_HOST="${DEPLOY_HOST:-new.samyateia.de}"
REMOTE_APP_DIR="${REMOTE_APP_DIR:-apps/bioragent}"
BRANCH="${BRANCH:-main}"
REMOTE="${DEPLOY_USER}@${DEPLOY_HOST}"

echo "Rolling back ${REMOTE}:${REMOTE_APP_DIR} to ${TARGET_SHA}"

ssh "${REMOTE}" "bash -s" <<EOF
set -euo pipefail

APP_DIR_INPUT="${REMOTE_APP_DIR}"
if [[ "\${APP_DIR_INPUT}" == "~/"* ]]; then
  APP_DIR="\$HOME/\${APP_DIR_INPUT#~/}"
elif [[ "\${APP_DIR_INPUT}" == /* ]]; then
  APP_DIR="\${APP_DIR_INPUT}"
else
  APP_DIR="\$HOME/\${APP_DIR_INPUT}"
fi
REPO_DIR="\${APP_DIR}/repo"
TARGET_SHA="${TARGET_SHA}"
BRANCH="${BRANCH}"

if [[ ! -d "\${REPO_DIR}/.git" ]]; then
  echo "Repository not found at \${REPO_DIR}" >&2
  exit 1
fi

cd "\${REPO_DIR}"
git fetch origin "\${BRANCH}" --prune
git cat-file -e "\${TARGET_SHA}^{commit}"
CURRENT_SHA="\$(git rev-parse --short HEAD)"
git checkout "\${TARGET_SHA}"
ROLLED_SHA="\$(git rev-parse --short HEAD)"

if [[ ! -f "./scripts/deploy-server.sh" || ! -f "./deploy/compose.prod.yml" ]]; then
  echo "Deployment files are missing on remote checkout. Push latest repository changes first." >&2
  exit 1
fi

bash ./scripts/deploy-server.sh

printf "%s rollback from=%s to=%s\n" "\$(date -Is)" "\${CURRENT_SHA}" "\${ROLLED_SHA}" >> "\${APP_DIR}/releases.log"
echo "Rollback successful: \${CURRENT_SHA} -> \${ROLLED_SHA}"
EOF
