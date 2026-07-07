# Development Guide

## Backend

Install dependencies from `apps/api`:

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -e ".[dev]"
```

Run migrations when `AUTO_CREATE_TABLES=false`:

```bash
alembic upgrade head
```

Run the API:

```bash
uvicorn app.main:app --reload
```

Run tests:

```bash
pytest
```

## Frontend

Node.js is required for local frontend development.

```bash
cd apps/web
npm install
npm run dev
```

For VS Code preview or any non-localhost frontend host, keep the frontend talking to
the backend through the built-in Next.js proxy:

```bash
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
API_PROXY_TARGET=http://127.0.0.1:8000
```

The frontend will automatically switch to `/backend/...` requests when it detects
that the page is not being served from `localhost` or `127.0.0.1`, which avoids
preview-origin CORS failures during sign-in and sign-up flows.

Run the Playwright smoke suite after the API is available on `http://127.0.0.1:8000`
and the frontend is available on `http://127.0.0.1:3000`:

```bash
cd apps/web
npm run test:smoke
```

Install the Playwright browser once after dependencies are installed:

```bash
cd apps/web
npx playwright install chromium
```

Useful overrides:

```bash
PLAYWRIGHT_BASE_URL=http://127.0.0.1:3000
PLAYWRIGHT_API_URL=http://127.0.0.1:8000
PLAYWRIGHT_ADMIN_EMAIL=admin@aegispro.local
PLAYWRIGHT_ADMIN_PASSWORD=ChangeMe123!
```

## Infrastructure

```bash
docker compose up -d postgres redis
docker compose --profile web up -d
```

The web profile builds the frontend container and exposes Nginx on
`http://localhost:8080`.
