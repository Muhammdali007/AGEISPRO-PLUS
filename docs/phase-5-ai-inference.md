# Phase 5: AI Inference Engine

Phase 5 introduces the first independent inference service and a production-safe
event ingestion path. The application API still owns business workflows, while
the AI service owns detection execution and publishing.

## Architecture decisions

- AI inference lives in a new `apps/ai` FastAPI service so camera CRUD,
  incidents, and alerts never execute model code directly.
- The pipeline is intentionally pluggable: the default backend remains a
  simulated detector for local development, and a production-capable
  Ultralytics YOLO11 backend now supports ByteTrack-based tracked detections
  when model dependencies and weights are present.
- Detection event publishing terminates at `POST /api/v1/detections/ingest`,
  where the core API converts accepted detections into incidents and alerts.
- Alert creation is severity-aware: weapon and fire detections map to
  `critical`, smoke maps to `high`, and person detections remain incident-only
  until richer alert rules are added.

## Delivered API surface

- AI service: `GET /health`
- AI service: `POST /v1/inference/run`
- AI service: `POST /v1/inference/dispatch`
- Core API: `POST /api/v1/detections/ingest`

## Development behavior

- The AI service can run without GPU dependencies in `simulated` mode.
- Production-like inference is enabled with `AI_MODEL_BACKEND=ultralytics`,
  `pip install -e ".[model]"`, a YOLO11-compatible `AI_MODEL_WEIGHTS_PATH`,
  and `AI_MODEL_TRACKER_CONFIG=bytetrack.yaml`.
- When the requested model backend is unavailable, the service can fall back to
  `AI_MODEL_FALLBACK_BACKEND` and records that fallback in result metadata.
- Event dispatch is optional and controlled through environment variables.
- Detection payloads preserve model metadata, track IDs, inference FPS, and
  evidence paths so later phases can enrich analytics and review tooling.

## Validation

- Added AI-service tests for simulated fallback and YOLO/ByteTrack result
  parsing, plus backend tests for detection ingestion, alert generation,
  priority mapping, and disabled-camera handling.
- Frontend validation remains `npm run lint` and `npm run build` from
  `apps/web`.
