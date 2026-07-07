# Phase 3 Frontend Dashboard

Phase 3 delivers the first production-shaped frontend application shell on top of
the Phase 2 FastAPI contracts.

## Scope delivered

- Protected dashboard layout with persistent sidebar navigation and system status.
- Public sign-in and sign-up entry points that establish dashboard sessions.
- API-backed overview page with summary cards, incident feed, and alert feed.
- Cameras section with registry view and per-camera detail route.
- Incidents section with queue view and per-incident detail route.
- Users section with RBAC-aware user inventory view.
- Analytics placeholder route using derived data from current APIs.
- React Query provider and typed frontend API client expansion for Phase 2 routes.

## Architectural choices

- The dashboard lives under `app/dashboard/*` so Phase 4 and later phases can add
  deeper route trees without disturbing auth and layout concerns.
- The shell is client-rendered because authentication state depends on browser
  storage in the current Phase 1 foundation.
- TanStack Query manages server state so future streaming, polling, and
  optimistic incident workflows can extend a consistent data layer.
- The UI intentionally uses real backend responses instead of static mocks where
  contracts already exist.

## Phase boundary

Phase 3 is complete when operators can sign in or sign up, navigate the
application, and inspect backend-backed dashboard sections for overview,
cameras, incidents, users, and analytics placeholders.
