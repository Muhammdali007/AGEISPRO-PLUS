from fastapi import APIRouter

from app.services.system_health import read_readiness

router = APIRouter()


@router.get("")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
async def readiness() -> dict[str, str]:
    return await read_readiness()
