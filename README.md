# BioRAGent
A retrieval augmented generation system for showcasing generative query expansion and domain-specific search for scientific Q&amp;A

## CI

GitHub Actions only runs a Docker build validation on pushes and pull requests.

- Workflow: `.github/workflows/ci.yml`

## Deployment

Deployment is executed from this repository using SSH to the target server.

Run from your local machine:

```bash
./deploy.sh
```

Rollback from your local machine:

```bash
./rollback.sh <commit-sha>
```

Defaults:

- server: `samy@new.samyateia.de`
- app dir on server: `~/apps/bioragent`
- branch: `main`

Optional overrides per run:

```bash
DEPLOY_HOST=example.com DEPLOY_USER=samy BRANCH=main ./deploy.sh
```

Server requirements:

- `~/apps/bioragent/.env` exists with runtime secrets:
  - `GEMINI_API_KEY`
  - `ELASTICSEARCH_HOST`
  - `ELASTICSEARCH_USER`
  - `ELASTICSEARCH_PASSWORD`
- server user has Docker permissions.
