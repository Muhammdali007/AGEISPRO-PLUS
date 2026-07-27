from __future__ import annotations

import base64
import json
import os
import ssl
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from fastapi import HTTPException, status

from app.core.config import PROJECT_ROOT, settings
from app.services.camera_network_policy import CameraNetworkPolicy

VIDEO_FRAME_EXTENSIONS = {".m3u8", ".mp4", ".webm", ".mov", ".m4v", ".ogv"}
MJPEG_CONTENT_TYPES = {"multipart/x-mixed-replace", "application/octet-stream"}


class MediaAgentError(Exception):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass(slots=True)
class MediaFrame:
    content_base64: str
    content_type: str
    source_position_seconds: float | None = None
    source_duration_seconds: float | None = None


class LocalSubprocessMediaAgent:
    def __init__(self, network_policy: CameraNetworkPolicy | None = None) -> None:
        self.network_policy = network_policy or CameraNetworkPolicy()

    def capture_opencv_frame(
        self,
        source: str | int,
        *,
        display_source: str | int | None = None,
        position_seconds: float | None = None,
    ) -> MediaFrame:
        return self._invoke(
            {
                "action": "capture_opencv_frame",
                "source": source,
                "display_source": display_source,
                "position_seconds": position_seconds,
            }
        )

    def capture_http_frame(
        self,
        *,
        source: str,
        source_descriptor: str,
        skip_tls_verification: bool,
    ) -> MediaFrame:
        return self._invoke(
            {
                "action": "capture_http_frame",
                "source": source,
                "source_descriptor": source_descriptor,
                "skip_tls_verification": skip_tls_verification,
            }
        )

    def _invoke(self, payload: dict[str, object]) -> MediaFrame:
        env = os.environ.copy()
        env["NO_PROXY"] = "*"
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            env.pop(key, None)

        try:
            result = subprocess.run(
                [settings.camera_media_agent_python, "-m", "app.services.media_agent_worker"],
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                cwd=str(_api_runtime_root()),
                env=env,
                timeout=settings.camera_media_agent_timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="The isolated media agent timed out while reading the camera source.",
            ) from exc

        body = (result.stdout or "").strip()
        try:
            parsed = json.loads(body) if body else {}
        except json.JSONDecodeError as exc:
            stderr = (result.stderr or "").strip()
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Isolated media agent returned an invalid response. {stderr}".strip(),
            ) from exc

        if result.returncode != 0:
            detail = str(parsed.get("error") or (result.stderr or "").strip() or "Media capture failed.")
            status_code = int(parsed.get("status_code") or status.HTTP_502_BAD_GATEWAY)
            raise HTTPException(status_code=status_code, detail=detail)

        content_base64 = parsed.get("frame_content_base64")
        content_type = parsed.get("frame_content_type")
        if not isinstance(content_base64, str) or not isinstance(content_type, str):
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Isolated media agent did not return a valid frame payload.",
            )

        return MediaFrame(
            content_base64=content_base64,
            content_type=content_type,
            source_position_seconds=_optional_float(parsed.get("source_position_seconds")),
            source_duration_seconds=_optional_float(parsed.get("source_duration_seconds")),
        )


def _api_runtime_root() -> Path:
    source_checkout_root = PROJECT_ROOT / "apps" / "api"
    if (source_checkout_root / "app").is_dir():
        return source_checkout_root
    return PROJECT_ROOT


def run_media_request(payload: dict[str, object]) -> dict[str, object]:
    action = str(payload.get("action") or "")
    network_policy = CameraNetworkPolicy()

    if action == "capture_opencv_frame":
        source = payload.get("source")
        display_source = payload.get("display_source")
        position_seconds = payload.get("position_seconds")
        if not isinstance(source, (str, int)):
            raise MediaAgentError("Media agent expected a string or integer camera source.", 400)
        if position_seconds is not None and not isinstance(position_seconds, (int, float)):
            raise MediaAgentError("Media frame position must be a number of seconds.", 400)
        frame_content_base64, frame_content_type, frame_position, duration = _capture_opencv_frame(
            source,
            display_source=display_source if isinstance(display_source, (str, int)) else None,
            position_seconds=float(position_seconds) if position_seconds is not None else None,
            network_policy=network_policy,
        )
        return {
            "frame_content_base64": frame_content_base64,
            "frame_content_type": frame_content_type,
            "source_position_seconds": frame_position,
            "source_duration_seconds": duration,
        }

    if action == "capture_http_frame":
        source = payload.get("source")
        source_descriptor = payload.get("source_descriptor")
        skip_tls_verification = bool(payload.get("skip_tls_verification"))
        if not isinstance(source, str) or not isinstance(source_descriptor, str):
            raise MediaAgentError("Media agent expected an HTTP source and descriptor.", 400)
        frame_content_base64, frame_content_type = _capture_http_frame(
            source=source,
            source_descriptor=source_descriptor,
            skip_tls_verification=skip_tls_verification,
            network_policy=network_policy,
        )
        return {
            "frame_content_base64": frame_content_base64,
            "frame_content_type": frame_content_type,
        }

    raise MediaAgentError(f"Unsupported media agent action '{action}'.", 400)


def _capture_opencv_frame(
    source: str | int,
    *,
    display_source: str | int | None,
    position_seconds: float | None = None,
    network_policy: CameraNetworkPolicy,
) -> tuple[str, str, float | None, float | None]:
    if isinstance(source, str):
        parsed = urlparse(source)
        scheme = parsed.scheme.lower()
        if scheme in {"http", "https"}:
            network_policy.validate_url(source, allowed_protocols={"http", "https"})
        elif scheme == "rtsp":
            network_policy.validate_rtsp_url(source)

    try:
        import cv2
    except ImportError as exc:
        raise MediaAgentError("OpenCV is required to read USB, RTSP, and video camera frames.", 503) from exc

    capture = cv2.VideoCapture(source)
    try:
        if position_seconds is not None:
            if position_seconds < 0:
                raise MediaAgentError("Media frame position cannot be negative.", 400)
            capture.set(cv2.CAP_PROP_POS_MSEC, position_seconds * 1000.0)
        ok, frame = capture.read()
        if not ok or frame is None:
            rendered_source = display_source if display_source is not None else source
            raise MediaAgentError(f"Unable to decode a frame from camera source {rendered_source}.", 502)
        encoded, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not encoded:
            raise MediaAgentError("Unable to encode the captured camera frame.", 500)
        frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        duration = frame_count / fps if frame_count > 0 and fps > 0 else None
        frame_position = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0) / 1000.0
        return (
            base64.b64encode(buffer.tobytes()).decode("utf-8"),
            "image/jpeg",
            frame_position,
            duration,
        )
    finally:
        capture.release()


def _capture_http_frame(
    *,
    source: str,
    source_descriptor: str,
    skip_tls_verification: bool,
    network_policy: CameraNetworkPolicy,
) -> tuple[str, str]:
    if _looks_like_video_frame_source(source):
        content, content_type, _, _ = _capture_opencv_frame(
            source,
            display_source=source_descriptor,
            network_policy=network_policy,
        )
        return content, content_type

    context = None
    if skip_tls_verification and source.lower().startswith("https://"):
        context = ssl._create_unverified_context()

    response, final_url = network_policy.open_http_url(
        source,
        method="GET",
        timeout=5,
        headers={"User-Agent": "AegisPro/1.0"},
        context=context,
    )
    with response:
        content_type = (
            response.headers.get("Content-Type")
            or getattr(response.headers, "get_content_type", lambda: "")()
            or ""
        ).lower()
        if not content_type.startswith("image/"):
            if _looks_like_mjpeg_stream(final_url, content_type):
                frame = _extract_jpeg_frame(response)
                if not frame:
                    raise MediaAgentError(
                        f"Unable to extract a frame from the MJPEG stream at {source_descriptor}.",
                        502,
                    )
                return base64.b64encode(frame).decode("utf-8"), "image/jpeg"

            if content_type.startswith("video/"):
                content, frame_content_type, _, _ = _capture_opencv_frame(
                    source,
                    display_source=source_descriptor,
                    network_policy=network_policy,
                )
                return content, frame_content_type

            raise MediaAgentError(
                "This HTTP source is not serving a readable camera frame. Open the camera page and run the AI scan from the live preview.",
                400,
            )
        return base64.b64encode(response.read()).decode("utf-8"), content_type.split(";", 1)[0]


def _looks_like_video_frame_source(source: str) -> bool:
    return Path(urlparse(source).path).suffix.lower() in VIDEO_FRAME_EXTENSIONS


def _looks_like_mjpeg_stream(source: str, content_type: str) -> bool:
    normalized_content_type = content_type.split(";", 1)[0].strip().lower()
    normalized_path = urlparse(source).path.rstrip("/").lower()
    return (
        normalized_content_type in MJPEG_CONTENT_TYPES
        or normalized_path in {"/video", "/mjpeg", "/mjpg"}
        or normalized_path.endswith((".mjpeg", ".mjpg"))
    )


def _extract_jpeg_frame(response) -> bytes | None:  # type: ignore[no-untyped-def]
    buffer = bytearray()
    max_bytes = 5 * 1024 * 1024

    while len(buffer) < max_bytes:
        chunk = response.read(64 * 1024)
        if not chunk:
            break
        buffer.extend(chunk)

        start = buffer.find(b"\xff\xd8")
        if start == -1:
            if len(buffer) > 1024:
                del buffer[:-1024]
            continue

        end = buffer.find(b"\xff\xd9", start + 2)
        if end == -1:
            continue
        return bytes(buffer[start : end + 2])

    return None


def _optional_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None
