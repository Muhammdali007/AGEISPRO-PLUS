# AegisPro

AegisPro is an AI surveillance management platform for live monitoring, alerting,
incident workflows, and future plug-in AI inference services.

## Phase Status

Phase 1 establishes the project foundation:

- FastAPI backend with JWT authentication, RBAC-ready user roles, health checks,
  typed settings, SQLAlchemy, and Alembic.
- Next.js App Router frontend with a login page and protected infrastructure
  dashboard.
- PostgreSQL, Redis, frontend, and Nginx Docker Compose definitions.
- Environment examples, storage layout, and project documentation.

The AI service is intentionally not containerized in Phase 1 so later development
can access local webcams and GPU resources directly.

Phase 2 adds the functional core backend APIs:

- SQLAlchemy and Alembic schema for cameras, incidents, and alerts.
- User management APIs with role-aware access control.
- Camera CRUD APIs and a Phase 4 placeholder for connection testing.
- Incident filtering/update workflows and alert acknowledgement.
- Repository tests, RBAC tests, lint validation, and migration head validation.

Phase 3 adds the navigable frontend dashboard:

- Protected application shell with persistent navigation and system readiness.
- API-backed overview, camera registry, incident queue, user inventory, and
  analytics placeholder pages.
- Camera and incident detail routes prepared for streaming and evidence
  workflows in later phases.
- Typed React Query data layer aligned with the Phase 2 FastAPI contracts.

Phase 4 adds camera streaming foundations:

- Source-aware camera health checks for USB, RTSP, HTTP, and file inputs.
- Stream descriptor APIs that keep browser playback concerns out of core camera
  CRUD contracts.
- Dashboard camera preview support for browser cameras, direct HTTP/HLS feeds,
  protected file playback, and RTSP relay guidance.

Phase 5 adds the independent AI inference layer:

- A separate `apps/ai` FastAPI service that keeps detection execution outside
  the core business API.
- A pluggable inference pipeline with both a development-safe simulated backend
  and a production-capable Ultralytics YOLO11 + ByteTrack path.
- Detection event ingestion on the main API that turns AI outputs into
  incidents and alerts without coupling the dashboard backend to inference code.

Phase 6 adds known person recognition:

- A persistent `/persons` domain in the FastAPI backend for known identities,
  enrollment metadata, and recognition history.
- Recognition-aware inference contracts in the independent AI service, including
  face-region extraction, production-capable `InsightFace` embedding support,
  deterministic hash fallback for development, and track deduping.
- Dashboard support for person management plus known-person and unknown-person
  incident context.

Phase 7 adds the incident and alert operations layer:

- Detection ingestion can now persist inline snapshot, clip, and face evidence
  into storage-backed incident records.
- Incident detail workflows now include evidence retrieval, operator notes,
  assignment updates, alert history, and alert acknowledgement or clearing.
- Authenticated websocket event streaming publishes live incident and alert
  lifecycle updates to connected operators.
- End-to-end backend tests cover evidence persistence, protected media access,
  alert workflow transitions, and websocket broadcast behavior.

Phase 8 adds operational monitoring:

- Backend-backed monitoring APIs for incident trends, camera health, system
  health, and audit activity.
- Persistent audit logs for login, camera, person, incident, and alert
  workflows.
- AI runtime telemetry for inference backend and best-effort GPU visibility.
- A real analytics dashboard that replaces the earlier placeholder route.

Phase 9 adds production hardening:

- Database-backed aggregation for monitoring KPI calculations instead of
  Python-side full-table filtering.
- Configurable database connection pool tuning and recycle controls.
- Composite indexes for monitoring, alert, camera, and audit-log hot paths.
- A dashboard-visible optimization report covering Redis latency, row volumes,
  and AI runtime capacity signals.

Phase 10 adds production deployment:

- HTTPS Nginx ingress with security headers, request limits, WebSocket proxying,
  and internal-only application services.
- A production Compose overlay with Gunicorn, health checks, bounded log
  rotation, Prometheus exporters, HTTP probes, and Grafana.
- Checksum-verified PostgreSQL and evidence backup/restore tooling.
- A deployment, certificate renewal, monitoring, rollback, and recovery runbook.

## Project Layout

```text
apps/
  api/      FastAPI backend
  ai/       Independent AI inference service
  web/      Next.js frontend
docs/       Architecture and phase documentation
infra/      Nginx and deployment support
storage/    Local evidence and media roots
```

## Local Setup

Copy environment files before running services:

```bash
cp .env.example .env
cp apps/api/.env.example apps/api/.env
cp apps/web/.env.example apps/web/.env.local
```

Start infrastructure:

```bash
docker compose up -d postgres redis
```

Run the API:

```bash
cd apps/api
python -m venv .venv
.venv/Scripts/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload
```

Run the AI service:

```bash
cd apps/ai
python -m venv .venv
.venv/Scripts/activate
pip install -e .
uvicorn app.main:app --reload --port 8100
```

To enable the model-backed detector instead of the simulated fallback:

```bash
cd apps/ai
pip install -e ".[model]"
```

Then set `AI_MODEL_BACKEND=ultralytics` and point `AI_MODEL_WEIGHTS_PATH` at a
YOLO11-compatible checkpoint. ByteTrack is enabled through
`AI_MODEL_TRACKER_CONFIG=bytetrack.yaml`.

To enable production-style face recognition embeddings instead of the
development-safe hash fallback:

```bash
cd apps/api
pip install -e ".[recognition]"

cd ../ai
pip install -e ".[recognition]"
```

Then set both services to `InsightFace`:

```bash
AI_RECOGNITION_BACKEND=insightface
API_RECOGNITION_BACKEND=insightface
AI_RECOGNITION_ALLOW_FALLBACK=false
API_RECOGNITION_ALLOW_FALLBACK=false
```

## Production deployment

The repo now includes a root [`.env.production`](C:/Users/Hp/Desktop/AegisPro/.env.production) for the Docker stack and a
template at [`.env.production.example`](C:/Users/Hp/Desktop/AegisPro/.env.production.example).

- `docker-compose.yml` now starts `postgres`, `redis`, `api`, `ai`, `web`, and `nginx`.
- The API container runs `alembic upgrade head` before serving traffic.
- Nginx proxies `/backend/*` directly to the API so browser calls and websocket events work behind one origin.
- Production mode now fails fast if simulated inference, hash recognition, fallback backends, default secrets, or a missing weapon checkpoint are configured.

Important:
- Real knife or weapon detection requires a trained checkpoint at `AI_MODEL_WEAPON_WEIGHTS_PATH`.
- The current repository contains `storage/models/yolo11n.pt`, but that is not a dedicated weapon model.
- Until you place a real weapon checkpoint at the configured path, the production AI service will refuse to start.

The default `CPUExecutionProvider` works for local CPU inference. In production,
you can swap the ONNX Runtime provider list and model settings to match your GPU
or deployment target.

Run the frontend after Node.js is available:

```bash
cd apps/web
npm install
npm run dev
```

Default development credentials:

```text
Email: admin@aegispro.local
Password: ChangeMe123!
```

`STORAGE_ROOT` is resolved relative to the repo root, so `storage` points at
`C:\\Users\\Hp\\Desktop\\AegisPro\\storage` in this workspace.

## API Checks

```bash
curl http://localhost:8000/api/v1/health
curl http://localhost:8000/api/v1/health/ready
```

## Quality Gates

```bash
cd apps/web
npm run build
npm run lint
npm test
```

## Phase Boundary

Phase 10 is now the implemented boundary. The platform ships with:

- A development-safe default and an opt-in production-grade recognition path.
- Storage-backed incident evidence capture and protected retrieval routes.
- Operator-facing incident workflow updates, incident-linked alert history, and
  alert acknowledgement or clearing flows.
- Authenticated websocket notifications for incident and alert lifecycle events.
- Monitoring APIs for incident trends, camera health, system readiness, AI
  runtime telemetry, and audit activity.
- A Phase 9 optimization dashboard that layers database, Redis, and runtime
  hardening telemetry on top of the operational monitoring experience.
- Database-side monitoring aggregates, connection-pool tuning, and composite
  indexing for production-readiness work.
- A hardened production ingress, private service network, operational
  monitoring, bounded logs, and a documented backup/restore strategy.

Documentation for earlier phases remains useful for architecture context, but
Phase 10 is the current end-to-end production deployment baseline. See
[`docs/phase-10-production-deployment.md`](docs/phase-10-production-deployment.md)
for the deployment and operations runbook.

Model roles, continuous camera processing, deduplication, checkpoint requirements, and current
validation limits are documented in
[`docs/detection-runtime.md`](docs/detection-runtime.md).
