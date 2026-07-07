# Phase 8: Analytics & Monitoring

Phase 8 turns the former analytics placeholder into an operational monitoring
surface backed by real API contracts, persisted audit history, and AI runtime
health telemetry.

## Architecture decisions

- The main FastAPI service remains the aggregation layer for monitoring data so
  the dashboard does not couple directly to the AI service.
- Existing cameras, incidents, and alerts tables remain the source of truth for
  KPI cards and trend charts; Phase 8 does not add a separate analytics store.
- AI runtime telemetry is best-effort and nullable. Unsupported hosts report
  missing GPU metrics without failing the overall monitoring response.
- Operator and administrator workflow actions now create persistent audit log
  entries that can be filtered from the monitoring dashboard.

## Delivered API surface

- Core API: `GET /api/v1/monitoring/overview`
- Core API: `GET /api/v1/monitoring/camera-health`
- Core API: `GET /api/v1/monitoring/system-health`
- Core API: `GET /api/v1/monitoring/audit-logs`
- AI service: `GET /health/runtime`

## Monitoring and audit capabilities

- Monitoring overview returns KPI cards, incident trend buckets, detection mix,
  camera health rollups, and aggregated system health.
- Camera health reports summarize online, offline, degraded, stale, and
  detection-enabled camera counts while exposing per-camera health entries.
- System health combines API readiness, Redis status, PostgreSQL status, and
  AI runtime health in one response.
- Audit logs persist login, user creation, camera lifecycle changes, camera
  health checks, person creation, face enrollment, incident updates, and alert
  acknowledgement or clearing.

## Frontend outcomes

- `/dashboard/analytics` now serves as a real operational monitoring dashboard
  instead of a Phase 3 placeholder route.
- Operators can switch reporting windows, inspect incident trends, review
  detection mix, monitor camera health, inspect AI/runtime readiness, and view
  recent audit activity from one screen.
- Dashboard shell copy now reflects the current monitoring phase.

## Validation

- Added API tests for monitoring overview, camera health, degraded AI runtime
  handling, and audit log filtering.
- Added AI service coverage for runtime health payloads with nullable GPU
  metrics.
- Added a standard `npm test` command and aligned the smoke suite with the
  current Phase 8 frontend experience.
