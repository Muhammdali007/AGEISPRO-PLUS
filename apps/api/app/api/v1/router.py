from fastapi import APIRouter

from app.api.v1.routes import alerts, auth, cameras, detections, health, incidents, monitoring, persons, users, video_rag, ws

api_router = APIRouter()
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(persons.router, prefix="/persons", tags=["persons"])
api_router.include_router(cameras.router, prefix="/cameras", tags=["cameras"])
api_router.include_router(incidents.router, prefix="/incidents", tags=["incidents"])
api_router.include_router(alerts.router, prefix="/alerts", tags=["alerts"])
api_router.include_router(detections.router, prefix="/detections", tags=["detections"])
api_router.include_router(monitoring.router, prefix="/monitoring", tags=["monitoring"])
api_router.include_router(video_rag.router, prefix="/video-rag", tags=["video-rag"])
api_router.include_router(ws.router, prefix="/ws", tags=["ws"])
