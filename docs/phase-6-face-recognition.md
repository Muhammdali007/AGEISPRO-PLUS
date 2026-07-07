# Phase 6: Known Person Recognition

Phase 6 adds the first end-to-end known-person workflow without collapsing the
boundary between the business API and the independent AI service.

## Architecture decisions

- Known-person records live in the main FastAPI application so identity
  governance, RBAC, and incident history remain part of the core business
  domain.
- Face matching remains in `apps/ai`; the recognition adapter is intentionally
  pluggable and now supports either a deterministic development-safe hash
  embedding flow or a production-capable `InsightFace` embedding backend.
- Detection ingestion stays centralized at `POST /api/v1/detections/ingest`,
  where recognition outputs are normalized into `known_person` and
  `unknown_person` incidents.
- Recognition events enrich incidents but do not generate alerts in this phase,
  preserving the earlier weapon/fire/smoke alert policy.

## Delivered API surface

- Core API: `GET /api/v1/persons`
- Core API: `POST /api/v1/persons`
- Core API: `GET /api/v1/persons/{person_id}`
- Core API: `PATCH /api/v1/persons/{person_id}`
- Core API: `POST /api/v1/persons/{person_id}/faces`
- Core API: `POST /api/v1/persons/match`
- Core API: extended `POST /api/v1/detections/ingest`
- AI service: extended `POST /v1/inference/run`
- AI service: extended `POST /v1/inference/dispatch`

## Frontend outcomes

- The dashboard now includes a dedicated Persons area with create and detail
  flows.
- Incident detail pages show known-person and unknown-person recognition
  context.
- Operators can inspect enrolled face metadata, recognition counts, and last
  seen timestamps without leaving the dashboard shell.

## Embedding storage and matching

- Every enrolled known-person face now creates a dedicated embedding record in
  the main API database instead of only living inside the person JSON profile.
- Uploaded and incident-promoted face images are embedded from real image bytes.
  When `InsightFace` is enabled in both services, enrollment and live inference
  use the same family of facial embeddings instead of synthetic placeholders.
- The AI service now derives live recognition embeddings from cropped frame
  content when inline frame evidence is present. If evidence or dependencies are
  unavailable, the hash fallback keeps the development workflow operational.
- The matcher can query nearest known-person candidates through
  `POST /api/v1/persons/match`.
- When PostgreSQL has the `vector` extension installed, the matcher uses
  pgvector cosine-distance search automatically. If the extension is not
  available yet, the API falls back to an in-process cosine match so the
  development workflow keeps working.

## Enabling InsightFace

Install the optional recognition extras in both Python services:

```bash
cd apps/api
pip install -e ".[recognition]"

cd ../ai
pip install -e ".[recognition]"
```

Then configure both services consistently:

```bash
AI_RECOGNITION_BACKEND=insightface
API_RECOGNITION_BACKEND=insightface
AI_RECOGNITION_ALLOW_FALLBACK=false
API_RECOGNITION_ALLOW_FALLBACK=false
```

Useful tuning knobs:

- `*_RECOGNITION_INSIGHTFACE_MODEL` defaults to `buffalo_l`.
- `*_RECOGNITION_INSIGHTFACE_PROVIDERS` defaults to `CPUExecutionProvider`.
- `*_RECOGNITION_INSIGHTFACE_CTX_ID` defaults to `-1` for CPU execution.
- `*_RECOGNITION_INSIGHTFACE_DET_SIZE` defaults to `640x640`.

## Validation

- Added API tests for person CRUD, RBAC, known-person ingestion, and
  unknown-person ingestion.
- Added AI service tests for recognition-aware schemas, track deduping, and
  callback payload serialization.
- Frontend validation passes with `npm run lint` and `npm run build`.
