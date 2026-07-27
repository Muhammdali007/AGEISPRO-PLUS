import asyncio
import signal

from app.core.logging import configure_logging
from app.db import metadata as _metadata  # noqa: F401 - register all ORM models for the worker
from app.services.video_rag_indexing import run_video_rag_worker


async def main() -> None:
    worker = asyncio.create_task(run_video_rag_worker())
    loop = asyncio.get_running_loop()
    for signal_name in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signal_name, worker.cancel)
    try:
        await worker
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    configure_logging()
    asyncio.run(main())
