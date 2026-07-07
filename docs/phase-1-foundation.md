# Phase 1 - Project Foundation

## Architectural Decisions

The system starts as a multi-app workspace rather than a monolith. The FastAPI
backend owns authentication, HTTP API contracts, persistence access, and health
checks. The Next.js frontend owns operator-facing UI and talks to the backend
through REST. The future AI service remains independent and will communicate
through API and event boundaries in later phases.

Authentication uses a local JWT provider for Phase 1 because it keeps the system
working without external credentials. The auth layer is isolated behind service
and repository classes so a mature provider such as Clerk, Supabase Auth, or
Auth.js can replace it without rewriting business modules.

PostgreSQL is the source of truth for metadata. Redis is introduced now for
readiness checks and future pub/sub, alert fanout, rate limiting, and background
task coordination. Media directories are created under `storage/`, but images
and clips are not stored in PostgreSQL.

## Directory Structure

```text
apps/api
  app/api            FastAPI route modules and dependencies
  app/core           settings, security, logging
  app/db             database session and bootstrap
  app/models         SQLAlchemy models
  app/repositories   persistence adapters
  app/schemas        Pydantic contracts
  app/services       application services
  alembic            database migration environment
  tests              backend tests

apps/web
  app                Next.js App Router pages
  components         reusable UI
  lib                API client and client state

infra/nginx          reverse proxy configuration
storage              filesystem roots for future evidence
```

## Database Schema

Phase 1 creates the `users` table:

| Column | Type | Notes |
| --- | --- | --- |
| id | UUID | Primary key |
| email | varchar(320) | Unique login identity |
| full_name | varchar(160) | Display name |
| role | enum | administrator, supervisor, operator, viewer |
| password_hash | varchar(255) | Bcrypt hash |
| is_active | boolean | Login eligibility |
| created_at | timestamptz | Audit timestamp |
| updated_at | timestamptz | Audit timestamp |

## API Contracts

`POST /api/v1/auth/login`

```json
{
  "email": "admin@aegispro.local",
  "password": "ChangeMe123!"
}
```

Returns an access token and refresh token.

`POST /api/v1/auth/refresh`

```json
{
  "refresh_token": "..."
}
```

Returns a new token pair.

`GET /api/v1/auth/me`

Requires `Authorization: Bearer <access_token>` and returns the current user.

`GET /api/v1/health`

Returns API liveness.

`GET /api/v1/health/ready`

Checks PostgreSQL and Redis readiness.

## Validation

Phase 1 can be validated by:

1. Starting `postgres` and `redis` with Docker Compose.
2. Running the FastAPI app.
3. Logging in through the frontend with the bootstrap administrator.
4. Confirming `/api/v1/auth/me` returns the administrator profile.
5. Confirming `/api/v1/health/ready` reports database and Redis status.

`STORAGE_ROOT=storage` is the preferred local setting. The API resolves that
path from the project root, so it works regardless of the current working
directory.
