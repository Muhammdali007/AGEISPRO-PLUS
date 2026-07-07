import asyncio
from uuid import UUID

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import TokenType, decode_token
from app.db.session import get_db
from app.models.user import User, UserRole
from app.repositories.users import UserRepository
from app.services.event_broadcaster import event_broadcaster

router = APIRouter()


async def _authenticate_websocket_user(
    websocket: WebSocket,
    session: AsyncSession,
    token: str | None,
) -> User | None:
    if not token:
        await websocket.close(code=4401, reason="Missing credentials")
        return None
    try:
        payload = decode_token(token, TokenType.access)
        user_id = UUID(payload["sub"])
    except (KeyError, ValueError):
        await websocket.close(code=4401, reason="Invalid credentials")
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
    token: str | None = Query(default=None),
    session: AsyncSession = Depends(get_db),
) -> None:
    user = await _authenticate_websocket_user(websocket, session, token)
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
