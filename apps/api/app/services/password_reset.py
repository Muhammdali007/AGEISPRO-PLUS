from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import secrets
from datetime import UTC, datetime, timedelta
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.security import hash_password
from app.models.password_reset_token import PasswordResetToken
from app.models.user import UserRole
from app.repositories.users import UserRepository

PASSWORD_RESET_RESPONSE_DETAIL = (
    "If an active administrator account matches that email, password reset instructions have been sent."
)
PASSWORD_RESET_INVALID_DETAIL = "Invalid or expired password reset token"

logger = logging.getLogger(__name__)


class PasswordResetError(ValueError):
    pass


class EmailDeliveryError(RuntimeError):
    pass


class PasswordResetEmailSender(Protocol):
    async def send_password_reset_email(self, *, email: str, reset_url: str) -> None:
        pass


class NoopPasswordResetEmailSender:
    async def send_password_reset_email(self, *, email: str, reset_url: str) -> None:
        return None


class SendGridPasswordResetEmailSender:
    def __init__(
        self,
        *,
        api_key: str,
        from_email: str,
        from_name: str,
    ) -> None:
        self.api_key = api_key
        self.from_email = from_email
        self.from_name = from_name

    async def send_password_reset_email(self, *, email: str, reset_url: str) -> None:
        await asyncio.to_thread(self._send, email=email, reset_url=reset_url)

    def _send(self, *, email: str, reset_url: str) -> None:
        payload = {
            "personalizations": [
                {
                    "to": [{"email": email}],
                    "subject": "Reset your AegisPro administrator password",
                }
            ],
            "from": {"email": self.from_email, "name": self.from_name},
            "content": [
                {
                    "type": "text/plain",
                    "value": (
                        "Use the link below to reset your AegisPro administrator password. "
                        "This link expires soon and can be used only once.\n\n"
                        f"{reset_url}\n\n"
                        "If you did not request this reset, ignore this message."
                    ),
                }
            ],
        }
        request = Request(
            "https://api.sendgrid.com/v3/mail/send",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=10) as response:
                if response.status >= 300:
                    raise EmailDeliveryError(f"SendGrid returned HTTP {response.status}")
        except (HTTPError, URLError, TimeoutError) as exc:
            raise EmailDeliveryError("Unable to send password reset email") from exc


class PasswordResetService:
    def __init__(
        self,
        session: AsyncSession,
        *,
        users: UserRepository,
        email_sender: PasswordResetEmailSender,
    ) -> None:
        self.session = session
        self.users = users
        self.email_sender = email_sender

    async def request_admin_password_reset(self, email: str) -> None:
        user = await self.users.get_by_email(email)
        if not user or not user.is_active or user.role is not UserRole.administrator:
            return

        now = datetime.now(UTC)
        raw_token = secrets.token_urlsafe(32)
        token = PasswordResetToken(
            user_id=user.id,
            token_digest=digest_password_reset_token(raw_token),
            expires_at=now + timedelta(minutes=settings.password_reset_token_minutes),
            created_at=now,
        )
        await self.session.execute(
            update(PasswordResetToken)
            .where(PasswordResetToken.user_id == user.id, PasswordResetToken.used_at.is_(None))
            .values(used_at=now)
        )
        self.session.add(token)
        await self.session.flush()

        reset_url = build_password_reset_url(raw_token)
        try:
            await self.email_sender.send_password_reset_email(email=user.email, reset_url=reset_url)
        except EmailDeliveryError:
            logger.exception("Password reset email delivery failed for administrator %s", user.id)

    async def reset_admin_password(self, raw_token: str, password: str) -> None:
        now = datetime.now(UTC)
        token = await self.session.scalar(
            select(PasswordResetToken).where(
                PasswordResetToken.token_digest == digest_password_reset_token(raw_token),
                PasswordResetToken.used_at.is_(None),
                PasswordResetToken.expires_at > now,
            )
        )
        if token is None:
            raise PasswordResetError(PASSWORD_RESET_INVALID_DETAIL)

        user = await self.users.get_by_id(token.user_id)
        if not user or not user.is_active or user.role is not UserRole.administrator:
            raise PasswordResetError(PASSWORD_RESET_INVALID_DETAIL)

        user.password_hash = hash_password(password)
        token.used_at = now
        await self.session.execute(
            update(PasswordResetToken)
            .where(
                PasswordResetToken.user_id == user.id,
                PasswordResetToken.id != token.id,
                PasswordResetToken.used_at.is_(None),
            )
            .values(used_at=now)
        )
        await self.session.flush()


def digest_password_reset_token(raw_token: str) -> str:
    return hmac.new(
        settings.secret_key.encode("utf-8"),
        raw_token.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def build_password_reset_url(raw_token: str) -> str:
    return (
        f"{settings.web_app_url.rstrip('/')}/reset-password?"
        f"token={quote(raw_token, safe='')}"
    )


def get_password_reset_email_sender() -> PasswordResetEmailSender:
    if settings.sendgrid_api_key and settings.password_reset_from_email:
        return SendGridPasswordResetEmailSender(
            api_key=settings.sendgrid_api_key,
            from_email=settings.password_reset_from_email,
            from_name=settings.password_reset_from_name,
        )
    return NoopPasswordResetEmailSender()
