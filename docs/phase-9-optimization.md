# Phase 9: Optimization

Phase 9 hardens the existing Phase 8 monitoring stack for production-oriented
load patterns without changing the core service boundaries established in the
initial plan.

## Architectural Decisions

- Monitoring overview aggregation now shifts incident volume, alert counts, and
  detection mix calculations into filtered SQL queries instead of loading full
  incident and alert rows into Python memory.
- Database pool sizing and recycle settings are configurable so deployment
  environments can tune connection reuse without changing application code.
- A dedicated optimization report endpoint exposes database pool posture, Redis
  latency and memory signals, persisted row volumes, and AI runtime telemetry in
  one operator-facing contract.
- Composite indexes were added on the most common monitoring and operational
  access paths so trend and audit queries scale more predictably as incident
  volume grows.

## API Surface

- `GET /api/v1/monitoring/optimization`
- Existing `GET /api/v1/monitoring/overview` now uses database-backed
  aggregation for KPI and detection-mix computations.

## Data and Indexing

Added composite indexes:

- `cameras(status, group)`
- `incidents(occurred_at, detection_type)`
- `incidents(status, occurred_at)`
- `alerts(status, created_at)`
- `audit_logs(action, created_at)`
- `audit_logs(resource_type, created_at)`

## Frontend

- `/dashboard/analytics` now presents Phase 9 optimization telemetry alongside
  the existing monitoring dashboard.
- The dashboard shell and smoke tests now reflect the Phase 9 boundary.

## Validation

- Added backend tests for windowed aggregate monitoring behavior and the new
  optimization report.
- Existing frontend smoke coverage was extended to assert optimization telemetry
  rendering.
