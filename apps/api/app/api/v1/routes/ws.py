import asyncio
from uuid import UUID

from fastapi import APIRouter, Cookie, Depends, Query, WebSocket, WebSocketDisconnect, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import AegisAccessCookie
from app.core.security import TokenType, decode_token, hash_token_identifier
from app.db.session import get_db
from app.models.user import User, UserRole
from app.repositories.user_sessions import UserSessionRepository
from app.repositories.users import UserRepository
from app.services.event_broadcaster import event_broadcaster

router = APIRouter()


async def _authenticate_websocket_user(
    websocket: WebSocket,
    session: AsyncSession,
    access_cookie: str | None,
    access_token: str | None,
) -> User | None:
    credential = access_cookie or access_token
    if not credential:
        await websocket.close(code=4401, reason="Missing credentials")
        return None
    try:
        payload = decode_token(credential, TokenType.access)
        user_id = UUID(payload["sub"])
        access_jti_digest = hash_token_identifier(payload["jti"])
    except (KeyError, ValueError):
        await websocket.close(code=4401, reason="Invalid credentials")
        return None

    user_session = await UserSessionRepository(session).get_active_by_access_jti(access_jti_digest)
    if not user_session or user_session.user_id != user_id or payload.get("sid") != str(user_session.id):
        await websocket.close(code=4401, reason="Invalid session")
        return None

    user = await UserRepository(session).get_by_id(user_id)
    if not user or not user.is_active:
        await websocket.close(code=4401, reason="Inactive user")
        return None
    if user.role not in {
        UserRole.administrator,
        UserRole.supervisor,
        UserRole.operator,
        UserRole.viewer,
    }:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Insufficient role")
        return None
    return user


@router.websocket("/events")
async def event_stream(
    websocket: WebSocket,
    access_cookie: str | None = Cookie(default=None, alias=AegisAccessCookie),
    token: str | None = Query(default=None),
    session: AsyncSession = Depends(get_db),
) -> None:
    user = await _authenticate_websocket_user(websocket, session, access_cookie, token)
    if not user:
        return

    await websocket.accept()
    await websocket.send_json({"type": "system.connected", "role": user.role.value})

    async with event_broadcaster.subscribe() as queue:
        while True:
            receive_task = asyncio.create_task(websocket.receive_text())
            queue_task = asyncio.create_task(queue.get())
            done, pending = await asyncio.wait(
                {receive_task, queue_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in pending:
                task.cancel()

            if receive_task in done:
                try:
                    receive_task.result()
                except WebSocketDisconnect:
                    break
                continue

            try:
                event = queue_task.result()
            except asyncio.CancelledError:
                break

            try:
                await websocket.send_json(event)
            except WebSocketDisconnect:
                break
