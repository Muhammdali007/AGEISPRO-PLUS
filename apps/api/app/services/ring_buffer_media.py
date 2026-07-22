from __future__ import annotations

import base64
import hashlib
import hmac
import tempfile
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from uuid import UUID

from app.core.config import settings


@dataclass(slots=True)
class BufferedFrame:
    captured_at: datetime
    monotonic_at: float
    content_base64: str
    content_type: str


@dataclass(slots=True)
class EventClipEvidence:
    content_base64: str
    content_type: str
    metadata: dict[str, object]


FrameCapture = Callable[[], Awaitable[tuple[str, str]]]


class RingBufferMediaService:
    def __init__(self) -> None:
        self._frames: dict[str, deque[BufferedFrame]] = defaultdict(deque)

    def add_frame(
        self,
        camera_id: UUID | object,
        *,
        content_base64: str,
        content_type: str,
        captured_at: datetime | None = None,
    ) -> None:
        now = captured_at or datetime.now(UTC)
        frame = BufferedFrame(
            captured_at=now,
            monotonic_at=monotonic(),
            content_base64=content_base64,
            content_type=content_type,
        )
        buffer = self._frames[str(camera_id)]
        buffer.append(frame)
        self._trim_buffer(buffer, frame.monotonic_at)

    async def build_event_clip(
        self,
        camera_id: UUID | object,
        *,
        capture_after_frame: FrameCapture | None = None,
    ) -> EventClipEvidence | None:
        before_frames = self._recent_frames(camera_id)
        after_frames = await self._capture_after_frames(camera_id, capture_after_frame)
        frames = [*before_frames, *after_frames]
        if len(frames) < 2:
            return None

        clip_bytes = self._encode_mp4(frames)
        if clip_bytes is None:
            return None

        checksum = hashlib.sha256(clip_bytes).hexdigest()
        started_at = frames[0].captured_at
        ended_at = frames[-1].captured_at
        signature_payload = f"{camera_id}:{started_at.isoformat()}:{ended_at.isoformat()}:{checksum}"
        signature = hmac.new(
            settings.secret_key.encode("utf-8"),
            signature_payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return EventClipEvidence(
            content_base64=base64.b64encode(clip_bytes).decode("utf-8"),
            content_type="video/mp4",
            metadata={
                "event_clip": {
                    "sha256": checksum,
                    "signature": signature,
                    "signature_algorithm": "hmac-sha256",
                    "signed_payload": signature_payload,
                    "signed_at": datetime.now(UTC).isoformat(),
                    "started_at": started_at.isoformat(),
                    "ended_at": ended_at.isoformat(),
                    "before_seconds": settings.event_clip_before_seconds,
                    "after_seconds": settings.event_clip_after_seconds,
                    "fps": settings.event_clip_fps,
                    "frame_count": len(frames),
                }
            },
        )

    def _recent_frames(self, camera_id: UUID | object) -> list[BufferedFrame]:
        now = monotonic()
        buffer = self._frames[str(camera_id)]
        self._trim_buffer(buffer, now)
        return list(buffer)

    async def _capture_after_frames(
        self,
        camera_id: UUID | object,
        capture_after_frame: FrameCapture | None,
    ) -> list[BufferedFrame]:
        if capture_after_frame is None:
            return []

        frames: list[BufferedFrame] = []
        count = max(1, settings.event_clip_after_seconds * settings.event_clip_fps)
        for _ in range(count):
            try:
                content_base64, content_type = await capture_after_frame()
            except Exception:
                break
            frame = BufferedFrame(
                captured_at=datetime.now(UTC),
                monotonic_at=monotonic(),
                content_base64=content_base64,
                content_type=content_type,
            )
            frames.append(frame)
            self.add_frame(
                camera_id,
                content_base64=content_base64,
                content_type=content_type,
                captured_at=frame.captured_at,
            )
        return frames

    def _trim_buffer(self, buffer: deque[BufferedFrame], now: float) -> None:
        window = settings.event_clip_before_seconds + 1
        max_frames = max(2, window * settings.event_clip_fps * 2)
        while buffer and now - buffer[0].monotonic_at > window:
            buffer.popleft()
        while len(buffer) > max_frames:
            buffer.popleft()

    @staticmethod
    def _encode_mp4(frames: list[BufferedFrame]) -> bytes | None:
        try:
            import cv2
            import numpy as np
        except ImportError:
            return None

        decoded = []
        for frame in frames:
            try:
                raw = base64.b64decode(frame.content_base64, validate=True)
                image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            except ValueError:
                image = None
            if image is not None:
                decoded.append(image)
        if len(decoded) < 2:
            return None

        height, width = decoded[0].shape[:2]
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "event.mp4"
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(settings.event_clip_fps),
                (width, height),
            )
            if not writer.isOpened():
                return None
            try:
                for image in decoded:
                    if image.shape[1] != width or image.shape[0] != height:
                        image = cv2.resize(image, (width, height))
                    writer.write(image)
            finally:
                writer.release()
            if not output_path.is_file():
                return None
            return output_path.read_bytes()


ring_buffer_media_service = RingBufferMediaService()
