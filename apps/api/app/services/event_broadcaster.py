import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4


class EventBroadcaster:
    def __init__(self) -> None:
        self._subscribers: dict[str, asyncio.Queue[dict[str, Any]]] = {}
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def subscribe(self) -> AsyncIterator[asyncio.Queue[dict[str, Any]]]:
        subscriber_id = str(uuid4())
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        async with self._lock:
            self._subscribers[subscriber_id] = queue
        try:
            yield queue
        finally:
            async with self._lock:
                self._subscribers.pop(subscriber_id, None)

    async def publish(self, event: dict[str, Any]) -> None:
        async with self._lock:
            subscribers = list(self._subscribers.values())
        for queue in subscribers:
            await queue.put(event)


event_broadcaster = EventBroadcaster()
