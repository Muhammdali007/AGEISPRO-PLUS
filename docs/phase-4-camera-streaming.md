# Phase 4: Camera Streaming

Phase 4 introduces the first live-operations layer on top of the existing camera
registry. The implementation keeps AI inference separate from business logic
while giving operators a usable streaming and health surface today.

## Architecture decisions

- FastAPI now exposes source-aware camera health checks and stream descriptors
  instead of hard-coding playback behavior in the frontend.
- USB cameras are previewed with browser `getUserMedia`, which works in local
  development without forcing the API to own webcam access before the AI service
  exists.
- HTTP and browser-compatible relay outputs are rendered directly by the web
  client.
- RTSP sources are health-checked at the TCP layer and require a future relay
  URL for browser playback, avoiding a misleading broken RTSP player.
- File-backed sources are safely constrained to the configured `storage_root`
  before the API serves them.

## Delivered API surface

- `POST /api/v1/cameras/{camera_id}/test-connection`
- `GET /api/v1/cameras/{camera_id}/stream`
- `GET /api/v1/cameras/{camera_id}/stream/file`

## Frontend outcomes

- Camera detail pages now show live preview behavior tailored to the camera
  source type.
- Operators can trigger health checks and immediately see updated status,
  latency, relay requirements, and stream notes.
- The camera registry reflects Phase 4 readiness instead of the earlier stub
  messaging.
