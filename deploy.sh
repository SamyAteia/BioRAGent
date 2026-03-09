#!/usr/bin/env bash
set -euo pipefail

DEPLOY_USER="${DEPLOY_USER:-samy}"
DEPLOY_HOST="${DEPLOY_HOST:-new.samyateia.de}"
REMOTE_APP_DIR="${REMOTE_APP_DIR:-apps/bioragent}"
BRANCH="${BRANCH:-main}"
DEPLOY_REPO_URL="${DEPLOY_REPO_URL:-git@github.com:SamyAteia/BioRAGent.git}"
REMOTE="${DEPLOY_USER}@${DEPLOY_HOST}"

if [[ "${DEPLOY_REPO_URL}" =~ ^https://[^/@]+:[^@]+@ ]]; then
  echo "DEPLOY_REPO_URL must not contain embedded credentials." >&2
  exit 1
fi

echo "Deploying ${BRANCH} to ${REMOTE}:${REMOTE_APP_DIR}"

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
BRANCH="${BRANCH}"
REPO_URL="${DEPLOY_REPO_URL}"

mkdir -p "\${APP_DIR}"

if [[ ! -d "\${REPO_DIR}/.git" ]]; then
  git clone "\${REPO_URL}" "\${REPO_DIR}"
fi

cd "\${REPO_DIR}"
git fetch origin "\${BRANCH}" --prune

PREV_SHA="\$(git rev-parse --short HEAD || echo none)"
git checkout "\${BRANCH}"
git reset --hard "origin/\${BRANCH}"
NEW_SHA="\$(git rev-parse --short HEAD)"

if [[ ! -f "./scripts/deploy-server.sh" || ! -f "./deploy/compose.prod.yml" ]]; then
  echo "Deployment files are missing on remote checkout. Push latest repository changes first." >&2
  exit 1
fi

bash ./scripts/deploy-server.sh

printf "%s deploy prev=%s new=%s\n" "\$(date -Is)" "\${PREV_SHA}" "\${NEW_SHA}" >> "\${APP_DIR}/releases.log"
echo "Deploy successful: \${PREV_SHA} -> \${NEW_SHA}"
EOF
