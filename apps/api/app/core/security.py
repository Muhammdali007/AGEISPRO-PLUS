import hashlib
import hmac
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any
from uuid import UUID
from uuid import uuid4

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import settings


class TokenType(StrEnum):
    access = "access"
    refresh = "refresh"


password_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
ALGORITHM = "HS256"


def hash_password(password: str) -> str:
    return password_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return password_context.verify(password, password_hash)


def hash_token_identifier(identifier: str) -> str:
    return hmac.new(
        settings.secret_key.encode("utf-8"),
        identifier.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def create_token(subject: UUID, role: str, token_type: TokenType, session_id: UUID | None = None) -> str:
    expires_delta = (
        timedelta(minutes=settings.access_token_minutes)
        if token_type is TokenType.access
        else timedelta(days=settings.refresh_token_days)
    )
    now = datetime.now(UTC)
    jti = str(uuid4())
    payload: dict[str, Any] = {
        "sub": str(subject),
        "role": role,
        "type": token_type.value,
        "iss": settings.jwt_issuer,
        "aud": settings.jwt_audience,
        "jti": jti,
        "iat": int(now.timestamp()),
        "exp": now + expires_delta,
    }
    if session_id:
        payload["sid"] = str(session_id)
    return jwt.encode(payload, settings.secret_key, algorithm=ALGORITHM)


def decode_token(token: str, expected_type: TokenType) -> dict[str, Any]:
    try:
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[ALGORITHM],
            issuer=settings.jwt_issuer,
            audience=settings.jwt_audience,
        )
    except JWTError as exc:
        raise ValueError("Invalid token") from exc
    if payload.get("type") != expected_type.value:
        raise ValueError("Invalid token type")
    if not payload.get("jti"):
        raise ValueError("Missing token identifier")
    return payload
