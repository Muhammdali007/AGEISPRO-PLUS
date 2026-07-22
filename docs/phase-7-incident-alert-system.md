# Phase 7: Incident & Alert System

Phase 7 turns detections into a fuller operator workflow with persisted
evidence, alert lifecycle actions, and live event fan-out.

## Architecture decisions

- Detection ingestion remains centralized at `POST /api/v1/detections/ingest`,
  but it now accepts inline evidence payloads that are validated before any
  incident rows are committed.
- Evidence files stay on disk under the configured storage root while incidents
  and alerts keep only metadata and relative paths in PostgreSQL.
- Incident and alert workflow state remains in the main FastAPI application so
  assignment, notes, acknowledgement, and audit-friendly history live beside
  RBAC enforcement.
- Live operator notifications use an authenticated websocket stream instead of
  coupling dashboard refresh behavior to direct inference callbacks.

## Delivered API surface

- Core API: extended `POST /api/v1/detections/ingest`
- Core API: `GET /api/v1/incidents`
- Core API: `GET /api/v1/incidents/{incident_id}`
- Core API: `PATCH /api/v1/incidents/{incident_id}`
- Core API: `GET /api/v1/incidents/{incident_id}/alerts`
- Core API: `GET /api/v1/incidents/{incident_id}/snapshot`
- Core API: `GET /api/v1/incidents/{incident_id}/clip`
- Core API: `GET /api/v1/alerts`
- Core API: `POST /api/v1/alerts/{alert_id}/acknowledge`
- Core API: `POST /api/v1/alerts/{alert_id}/clear`
- Core API: `WS /api/v1/ws/events?token=...`

## Evidence capture and retrieval

- Detection ingestion can persist inline snapshot and clip payloads directly
  into incident-scoped storage locations.
- Recognition-aware detections can also persist cropped face evidence and link
  it back through `recognized_identity.face_image_path`.
- Invalid inline evidence is rejected with a `422` before incident creation, so
  broken payloads do not leave partial workflow data behind.
- Snapshot and clip download routes resolve paths through
  `EvidenceStorageService`, which rejects files outside the configured storage
  root before returning a `FileResponse`.

## Retention and deletion

- Incidents use documented retention classes: `standard` archives after
  `API_INCIDENT_RETENTION_HOURS`, `extended` after
  `API_INCIDENT_EXTENDED_RETENTION_HOURS`, `compliance` after
  `API_INCIDENT_COMPLIANCE_RETENTION_HOURS`, and `manual` never archives
  automatically.
- Critical incidents default to `compliance`, but automatic cleanup still skips
  every critical incident and every incident that is not `resolved` or
  `dismissed`.
- Legal holds exclude incidents from automatic archive and evidence deletion.
- Archive is a soft-delete: incident rows stay in the database with
  `archived_at` and `deletion_requested_at`, related alerts are cleared rather
  than removed, and evidence files are deleted later by the retention worker.
- Evidence deletion is idempotent and asynchronous. A retry can safely process
  a previously removed incident directory, then clears evidence paths and marks
  `deletion_completed_at`.

## Alert workflow and live events

- Incident-linked alerts can be listed from either the global alerts feed or
  the incident detail route.
- Operators, supervisors, and administrators can acknowledge and clear alerts.
- Incident updates publish `incident.updated` events.
- Alert creation, acknowledgement, and clearing publish alert lifecycle events.
- Websocket clients receive an initial `system.connected` payload after
  authentication, then live `incident.*`, `alert.*`, and confirmed `sound.alert`
  events as they occur.
- Weapon, fire, and smoke sound events are immediate and rate-limited. Unknown
  people produce a sound event only after three consecutive person scans.

## Frontend outcomes

- The dashboard overview now consumes the live incident and alert feeds from the
  Phase 7 backend contracts.
- Incident detail pages expose snapshot and clip evidence when present, show
  recognition context, and support operator notes, assignment, and status
  changes from one screen.
- Related alerts are listed on the incident detail route with acknowledge and
  clear actions aligned to the backend workflow.

## Validation

- Added backend tests for inline evidence persistence across snapshot, clip, and
  face image storage.
- Added backend tests that reject malformed evidence without persisting
  incidents.
- Added API tests for incident-linked alert listing, snapshot retrieval, alert
  acknowledgement, alert clearing, and incident workflow patching.
- Added safety coverage for rejecting evidence paths outside the configured
  storage root.
- Added websocket tests that verify connected clients receive
  `incident.created` and `alert.created` broadcasts from ingestion events.
