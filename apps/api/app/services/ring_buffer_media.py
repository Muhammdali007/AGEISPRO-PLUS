from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import shutil
import subprocess
import tempfile
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from uuid import UUID

from app.core.config import settings

logger = logging.getLogger(__name__)


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
MINIMUM_EVENT_CLIP_SECONDS = 5


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
        if not frames:
            return None

        source_frame_count = len(frames)
        frames, repeated_frame_count = self._resample_clip_frames(
            frames,
            duration_seconds=MINIMUM_EVENT_CLIP_SECONDS,
            fps=settings.event_clip_fps,
        )

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
                    "duration_seconds": len(frames) / settings.event_clip_fps,
                    "minimum_duration_seconds": MINIMUM_EVENT_CLIP_SECONDS,
                    "source_frame_count": source_frame_count,
                    "padded_frame_count": repeated_frame_count,
                }
            },
        )

    @staticmethod
    def _resample_clip_frames(
        frames: list[BufferedFrame],
        *,
        duration_seconds: int,
        fps: int,
    ) -> tuple[list[BufferedFrame], int]:
        ordered = sorted(frames, key=lambda frame: frame.monotonic_at)
        target_count = max(2, duration_seconds * fps)
        end_at = ordered[-1].monotonic_at
        start_at = end_at - ((target_count - 1) / fps)
        sampled: list[BufferedFrame] = []
        source_index = 0
        for position in range(target_count):
            target_at = start_at + (position / fps)
            while (
                source_index + 1 < len(ordered)
                and ordered[source_index + 1].monotonic_at <= target_at
            ):
                source_index += 1
            sampled.append(ordered[source_index])
        distinct_source_frames = len({id(frame) for frame in sampled})
        return sampled, target_count - distinct_source_frames

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
        max_frames = max(2, window * max(settings.event_clip_fps * 2, 10))
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
            normalized = [
                image
                if image.shape[1] == width and image.shape[0] == height
                else cv2.resize(image, (width, height))
                for image in decoded
            ]
            ffmpeg = shutil.which("ffmpeg")
            if not ffmpeg:
                logger.warning(
                    "event_clip_external_encoder_unavailable fallback=opencv_video_writer"
                )
                return RingBufferMediaService._encode_mp4_with_opencv(
                    normalized,
                    output_path,
                )
            command = [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pixel_format",
                "bgr24",
                "-video_size",
                f"{width}x{height}",
                "-framerate",
                str(settings.event_clip_fps),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-profile:v",
                "baseline",
                "-level",
                "3.0",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output_path),
            ]
            try:
                encoded = subprocess.run(
                    command,
                    input=b"".join(image.tobytes() for image in normalized),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    timeout=30,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                logger.exception("event_clip_encoding_failed reason=ffmpeg_execution")
                return RingBufferMediaService._encode_mp4_with_opencv(
                    normalized,
                    output_path,
                )
            if encoded.returncode:
                logger.error(
                    "event_clip_external_encoder_failed fallback=opencv_video_writer "
                    "code=%s stderr=%s",
                    encoded.returncode,
                    encoded.stderr.decode("utf-8", errors="replace")[-1000:],
                )
                return RingBufferMediaService._encode_mp4_with_opencv(
                    normalized,
                    output_path,
                )
            if not output_path.is_file():
                logger.error(
                    "event_clip_external_encoder_failed fallback=opencv_video_writer "
                    "reason=missing_output"
                )
                return RingBufferMediaService._encode_mp4_with_opencv(
                    normalized,
                    output_path,
                )
            return output_path.read_bytes()

    @staticmethod
    def _encode_mp4_with_opencv(images: list[object], output_path: Path) -> bytes | None:
        import cv2

        height, width = images[0].shape[:2]
        writer = None
        selected_codec = None
        for codec in ("avc1", "mp4v"):
            candidate = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*codec),
                settings.event_clip_fps,
                (width, height),
            )
            if candidate.isOpened():
                writer = candidate
                selected_codec = codec
                break
            candidate.release()

        if writer is None:
            logger.error("event_clip_encoding_failed reason=opencv_video_writer_unavailable")
            return None

        try:
            for image in images:
                writer.write(image)
        finally:
            writer.release()

        if not output_path.is_file() or output_path.stat().st_size == 0:
            logger.error("event_clip_encoding_failed reason=opencv_video_writer_empty_output")
            return None
        logger.info("event_clip_encoded encoder=opencv codec=%s", selected_codec)
        return output_path.read_bytes()


ring_buffer_media_service = RingBufferMediaService()
