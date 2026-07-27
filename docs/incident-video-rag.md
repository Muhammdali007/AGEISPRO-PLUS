# Incident Video RAG

Incident Video RAG searches retained detector-created incident evidence. It does not record or search
continuous footage, transcribe audio, or replace operator review.

## Local setup

Install Ollama on the host, then download the compact local models:

```powershell
ollama pull gemma3:4b
ollama pull embeddinggemma
```

Set `VIDEO_RAG_ENABLED=true`. When the API and worker run directly on the host, use
`VIDEO_RAG_OLLAMA_URL=http://127.0.0.1:11434`. For the Docker stack on Windows or macOS, use
`http://host.docker.internal:11434`. A Linux deployment must expose its Ollama host to the containers
and set the corresponding reachable URL.

Apply the migration and start the standalone single-concurrency worker:

```powershell
Set-Location apps/api
alembic upgrade head
python -m app.workers.video_rag
```

The Compose stack starts `rag-worker` automatically. It remains idle while Video RAG is disabled.
The migration uses the PostgreSQL 16 pgvector image and enables the `vector` extension. An existing
PostgreSQL 16 data volume remains on the same database major version; take the normal database backup
before replacing the container image.

## Behavior

Authoritative detector metadata is searchable immediately, including while historical visual indexing
is still draining. The worker discovers every visible retained incident with a clip or snapshot,
prioritizes recent incidents, and continuously drains queued work without a delay between incidents.
It samples at most six ordered frames, obtains conservative visual observations from Gemma, embeds
those observations and authoritative incident metadata, and stores searchable chunks in PostgreSQL.
Changed evidence, notes, recognized identity data, or camera metadata causes reindexing.

Queries combine vector similarity and PostgreSQL English full-text ranking. Explicit dashboard camera
and time filters override filters inferred from the question. Answers must cite retrieved incident IDs;
an ungrounded model response is replaced with an abstention and a link to the top evidence record.

Archiving an incident immediately removes its RAG index. Raw questions and chat history are not stored;
the audit log records only a SHA-256 question digest, filters, result IDs, and query duration.

## Operational notes

- The default worker processes one incident at a time for a 16 GB development machine.
- `GET /api/v1/video-rag/status` reports ready, queued, processing, and failed index counts.
- Failed jobs retry with bounded exponential delay up to `VIDEO_RAG_MAX_ATTEMPTS`.
- A 503 query response means Video RAG is disabled or the configured Ollama service/model is unavailable.
- Generated visual observations are not forensic facts. Always verify important results against the
  protected original clip or snapshot.
