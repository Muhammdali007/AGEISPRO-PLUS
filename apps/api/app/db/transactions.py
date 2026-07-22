from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession

_TRANSACTION_OWNER_KEY = "_aegispro_transaction_owner"


class TransactionScope:
    def __init__(self, *, owns_transaction: bool) -> None:
        self.owns_transaction = owns_transaction


@asynccontextmanager
async def transaction_scope(session: AsyncSession) -> AsyncIterator[TransactionScope]:
    if session.info.get(_TRANSACTION_OWNER_KEY):
        yield TransactionScope(owns_transaction=False)
        return

    session.info[_TRANSACTION_OWNER_KEY] = True
    try:
        if session.in_transaction():
            try:
                yield TransactionScope(owns_transaction=True)
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            return

        async with session.begin():
            yield TransactionScope(owns_transaction=True)
    finally:
        session.info.pop(_TRANSACTION_OWNER_KEY, None)
