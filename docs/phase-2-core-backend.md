# Phase 2 - Database & Core Backend

## Architectural Decisions

Phase 2 keeps the backend modular and metadata-focused. FastAPI owns API
contracts and RBAC, SQLAlchemy owns persistence, and future AI inference remains
outside the business API. Evidence files are still stored by path only; the
database stores metadata for cameras, incidents, and alerts.

RBAC is enforced through reusable FastAPI dependencies. Administrators manage
all user and operational records, supervisors manage operational records,
operators can create and update incident/alert workflows, and viewers are
read-only for operational data.

## Database Schema

Phase 2 adds:

- `cameras`: camera source metadata, health state, detection settings, tags, and
  extra JSON metadata.
- `incidents`: detection type, priority, status, confidence, camera link,
  evidence paths, bounding boxes, assignment, and operator notes.
- `alerts`: incident-linked alert messages, acknowledgement state, and operator
  acknowledgement metadata.

The migration is `20260701_0002_phase_2_core_backend.py` and is the current
Alembic head.

## API Contracts

New API groups:

- `/api/v1/users`: user listing, creation, read, update, and soft deactivation.
- `/api/v1/cameras`: camera CRUD plus a Phase 4 placeholder connection test.
- `/api/v1/incidents`: incident creation, filtering, read, and workflow updates.
- `/api/v1/alerts`: alert creation, listing, and acknowledgement.

All routes require authentication. Role-specific permissions are implemented in
`app.api.deps.require_roles`.

## Validation

Validated with:

```bash
cd apps/api
pip install -e ".[dev]"
python -m pytest
python -m ruff check .
python -m alembic heads
```

The test suite covers metadata registration, user management, camera CRUD,
incident filtering and updates, alert acknowledgement, RBAC rejection, existing
configuration behavior, and token/password security.
