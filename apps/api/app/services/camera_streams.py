import asyncio
import ipaddress
import mimetypes
import ssl
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request

from app.core.config import settings
from app.models.camera import Camera, CameraSourceType, CameraStatus
from app.repositories.cameras import CameraRepository
from app.schemas.cameras import (
    CameraConnectionBatch,
    CameraConnectionTest,
    CameraLiveMonitorEntry,
    CameraLiveMonitorResponse,
    CameraLiveMonitorSummary,
    CameraStreamDescriptor,
)
from app.services.camera_network_policy import CameraNetworkPolicy
from app.services.camera_secrets import CameraSecretManager

VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".m4v", ".ogv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}


@dataclass(slots=True)
class HealthProbeResult:
    status: CameraStatus
    message: str
    checked_at: datetime
    latency_ms: int | None = None
    last_seen_at: datetime | None = None


@dataclass(slots=True)
class HttpProbeResult:
    source: str
    status_code: int
    content_type: str | None = None


@dataclass(slots=True)
class HttpStreamProxy:
    source: str
    content_type: str | None
    iterator: Iterator[bytes]


class CameraStreamingService:
    def __init__(self, cameras: CameraRepository) -> None:
        self.cameras = cameras
        self.network_policy = CameraNetworkPolicy()

    async def test_connection(self, camera: Camera) -> CameraConnectionTest:
        result = await self._probe(camera)
        await self.cameras.apply_health(
            camera,
            status=result.status,
            checked_at=result.checked_at,
            last_seen_at=result.last_seen_at,
        )
        return CameraConnectionTest(
            camera_id=camera.id,
            status=result.status,
            message=result.message,
            checked_at=result.checked_at,
            latency_ms=result.latency_ms,
        )

    async def test_connections(self, cameras: list[Camera]) -> CameraConnectionBatch:
        results: list[CameraConnectionTest] = []
        for camera in cameras:
            results.append(await self.test_connection(camera))
        return CameraConnectionBatch(results=results)

    async def describe_stream(self, camera: Camera) -> CameraStreamDescriptor:
        health_message = "Camera health has not been checked yet."
        if camera.health_checked_at:
            health_message = f"Last health check recorded {camera.health_checked_at.isoformat()}."

        descriptor = CameraStreamDescriptor(
            camera_id=camera.id,
            stream_kind="unavailable",
            stream_url=None,
            browser_supported=False,
            requires_relay=False,
            is_live=camera.source_type != CameraSourceType.file,
            health_status=camera.status,
            health_message=health_message,
            checked_at=camera.health_checked_at,
            controls=["refresh-health"],
            notes=[],
        )

        if camera.source_type == CameraSourceType.usb:
            device_id = self._metadata_value(camera, "browser_device_id")
            descriptor.stream_kind = "browser-camera"
            descriptor.browser_supported = True
            descriptor.controls.extend(["play", "mute"])
            descriptor.notes = [
                "Uses the operator browser's local camera access.",
                "Grant camera permission to preview USB devices in the dashboard.",
            ]
            descriptor.browser_device_id = device_id
            return descriptor

        if camera.source_type == CameraSourceType.file:
            source_path = self.resolve_file_source(camera)
            extension = source_path.suffix.lower()
            descriptor.stream_url = f"/api/v1/cameras/{camera.id}/stream/file"
            descriptor.browser_supported = extension in VIDEO_EXTENSIONS or extension in IMAGE_EXTENSIONS
            descriptor.stream_kind = self._stream_kind_from_extension(extension)
            descriptor.controls.extend(["download"])
            descriptor.notes = [f"Resolved file source: {source_path}"]
            return descriptor

        if camera.source_type == CameraSourceType.http:
            source = await self.cameras.get_runtime_source(camera)
            stream_url, stream_format, notes = self._resolve_http_stream(camera, source)
            descriptor.browser_supported = True
            descriptor.stream_kind = self._stream_kind_from_url(stream_url, stream_format=stream_format)
            descriptor.controls.extend(["open-external"])
            descriptor.notes = [
                "HTTP and HLS sources are played directly by the browser when supported.",
                *notes,
            ]
            if self._should_proxy_http_stream(camera, stream_url, descriptor.stream_kind):
                descriptor.stream_url = f"/api/v1/cameras/{camera.id}/stream/http"
                descriptor.notes.append(
                    "This feed is proxied through the API so private-network and self-signed phone cameras can open in the dashboard."
                )
                descriptor.notes.append(
                    "Upstream source: "
                    + CameraSecretManager.build_candidate_descriptor(CameraSourceType.http, stream_url)
                )
            else:
                descriptor.stream_url = stream_url
            if descriptor.stream_kind == "hls":
                descriptor.notes.append("Some browsers require native HLS support or a future HLS.js integration.")
            return descriptor

        if camera.source_type == CameraSourceType.rtsp:
            relay_url = self._metadata_value(camera, "relay_url")
            descriptor.requires_relay = True
            descriptor.controls.extend(["configure-relay"])
            if relay_url:
                descriptor.stream_kind = self._stream_kind_from_url(relay_url, stream_format="hls")
                descriptor.stream_url = relay_url
                descriptor.browser_supported = True
                descriptor.notes = [
                    "RTSP is being delivered through the configured relay URL.",
                    "Point relay_url at an HLS, WebRTC, MJPEG, or MP4-compatible endpoint.",
                ]
            else:
                descriptor.notes = [
                    "Browsers cannot play raw RTSP streams directly.",
                    "Set metadata.relay_url to a browser-compatible proxy output in a later AI/media service.",
                ]
            return descriptor

        return descriptor

    async def describe_live_monitor(self, cameras: list[Camera]) -> CameraLiveMonitorResponse:
        entries: list[CameraLiveMonitorEntry] = []
        groups: dict[str, int] = {}

        for camera in cameras:
            stream = await self.describe_stream(camera)
            entries.append(
                CameraLiveMonitorEntry(
                    camera=CameraSecretManager.serialize_camera(camera),
                    stream=stream,
                )
            )
            group_name = camera.group or "Ungrouped"
            groups[group_name] = groups.get(group_name, 0) + 1

        summary = CameraLiveMonitorSummary(
            total=len(cameras),
            online=sum(camera.status == CameraStatus.online for camera in cameras),
            offline=sum(camera.status == CameraStatus.offline for camera in cameras),
            degraded=sum(camera.status == CameraStatus.degraded for camera in cameras),
            disabled=sum(camera.status == CameraStatus.disabled for camera in cameras),
            unknown=sum(camera.status == CameraStatus.unknown for camera in cameras),
            live=sum(camera.source_type != CameraSourceType.file for camera in cameras),
            browser_ready=sum(entry.stream.browser_supported for entry in entries),
            relay_required=sum(entry.stream.requires_relay for entry in entries),
            detection_enabled=sum(camera.detection_enabled for camera in cameras),
            groups=dict(sorted(groups.items())),
        )
        return CameraLiveMonitorResponse(summary=summary, entries=entries)

    def resolve_file_source(self, camera: Camera) -> Path:
        source_path = Path(camera.source)
        resolved = source_path if source_path.is_absolute() else (settings.storage_root / source_path)
        resolved = resolved.resolve()
        storage_root = settings.storage_root.resolve()
        if resolved != storage_root and storage_root not in resolved.parents:
            raise ValueError("File camera sources must remain inside the configured storage root.")
        return resolved

    def prepare_http_stream_proxy(self, camera: Camera, source: str) -> HttpStreamProxy:
        if camera.source_type != CameraSourceType.http:
            raise ValueError("Camera is not HTTP-backed.")

        probe_result = self._probe_http_camera_source(camera, source)
        stream_url = probe_result.source
        skip_tls_verification = self._should_skip_tls_verification(camera, stream_url)
        if probe_result.status_code >= 400:
            descriptor = CameraSecretManager.build_candidate_descriptor(
                CameraSourceType.http,
                stream_url,
            )
            raise OSError(f"HTTP endpoint {descriptor} responded with status {probe_result.status_code}.")

        return HttpStreamProxy(
            source=stream_url,
            content_type=probe_result.content_type,
            iterator=self._iter_http_stream(stream_url, skip_tls_verification),
        )

    async def _probe(self, camera: Camera) -> HealthProbeResult:
        checked_at = datetime.now(UTC)
        start = time.perf_counter()

        try:
            if camera.source_type == CameraSourceType.usb:
                index = int(camera.source)
                if index < 0:
                    raise ValueError("USB camera index must be zero or a positive integer.")
                message = f"USB source index {index} is valid for browser-based preview."
                latency_ms = int((time.perf_counter() - start) * 1000)
                return HealthProbeResult(
                    status=CameraStatus.online,
                    message=message,
                    checked_at=checked_at,
                    latency_ms=latency_ms,
                    last_seen_at=checked_at,
                )

            if camera.source_type == CameraSourceType.file:
                source_path = self.resolve_file_source(camera)
                if not source_path.is_file():
                    raise FileNotFoundError(f"Video file not found: {source_path}")
                latency_ms = int((time.perf_counter() - start) * 1000)
                return HealthProbeResult(
                    status=CameraStatus.online,
                    message=f"Readable media file found at {source_path}.",
                    checked_at=checked_at,
                    latency_ms=latency_ms,
                    last_seen_at=checked_at,
                )

            if camera.source_type == CameraSourceType.http:
                source = await self.cameras.get_runtime_source(camera)
                probe_result = await asyncio.to_thread(self._probe_http_camera_source, camera, source)
                latency_ms = int((time.perf_counter() - start) * 1000)
                descriptor = CameraSecretManager.build_candidate_descriptor(
                    CameraSourceType.http,
                    probe_result.source,
                )
                return HealthProbeResult(
                    status=CameraStatus.online if probe_result.status_code < 400 else CameraStatus.degraded,
                    message=f"HTTP endpoint {descriptor} responded with status {probe_result.status_code}.",
                    checked_at=checked_at,
                    latency_ms=latency_ms,
                    last_seen_at=checked_at if probe_result.status_code < 400 else None,
                )

            if camera.source_type == CameraSourceType.rtsp:
                source = await self.cameras.get_runtime_source(camera)
                await self._probe_rtsp_source(source)
                latency_ms = int((time.perf_counter() - start) * 1000)
                parsed = urlparse(source)
                port = parsed.port or 554
                return HealthProbeResult(
                    status=CameraStatus.online,
                    message=f"RTSP host {parsed.hostname}:{port} accepted a TCP connection.",
                    checked_at=checked_at,
                    latency_ms=latency_ms,
                    last_seen_at=checked_at,
                )
        except (FileNotFoundError, OSError, URLError, ValueError) as exc:
            latency_ms = int((time.perf_counter() - start) * 1000)
            return HealthProbeResult(
                status=CameraStatus.offline,
                message=str(exc),
                checked_at=checked_at,
                latency_ms=latency_ms,
            )

        return HealthProbeResult(
            status=CameraStatus.unknown,
            message="Unsupported camera source type.",
            checked_at=checked_at,
        )

    @staticmethod
    def _metadata_value(camera: Camera, key: str) -> str | None:
        value = camera.metadata_.get(key)
        return value if isinstance(value, str) and value else None

    def _probe_http_source(self, source: str, skip_tls_verification: bool = False) -> int:
        context = None
        if skip_tls_verification and source.lower().startswith("https://"):
            context = ssl._create_unverified_context()

        request = Request(source, method="HEAD")
        try:
            with self.network_policy.open_http_url(
                source,
                method="HEAD",
                timeout=3,
                headers=dict(request.header_items()),
                context=context,
            )[0] as response:
                return response.status
        except HTTPError as exc:
            if exc.code in {405, 501}:
                request = Request(source, method="GET")
                with self.network_policy.open_http_url(
                    source,
                    method="GET",
                    timeout=3,
                    headers=dict(request.header_items()),
                    context=context,
                )[0] as response:
                    return response.status
            return exc.code
        except OSError:
            request = Request(source, method="GET")
            with self.network_policy.open_http_url(
                source,
                method="GET",
                timeout=3,
                headers=dict(request.header_items()),
                context=context,
            )[0] as response:
                return response.status

    def _probe_http_camera_source(self, camera: Camera, source: str) -> HttpProbeResult:
        errors: list[str] = []
        first_http_failure: HttpProbeResult | None = None
        for candidate in self._http_source_candidates(camera, source):
            try:
                status_code, content_type = self._probe_http_source_with_headers(
                    candidate,
                    self._should_skip_tls_verification(camera, candidate),
                )
            except OSError as exc:
                descriptor = CameraSecretManager.build_candidate_descriptor(
                    CameraSourceType.http,
                    candidate,
                )
                errors.append(f"{descriptor}: {exc}")
                continue

            if status_code >= 400:
                if first_http_failure is None:
                    first_http_failure = HttpProbeResult(
                        source=candidate,
                        status_code=status_code,
                        content_type=content_type,
                    )
                continue
            if content_type and "text/html" in content_type.lower():
                descriptor = CameraSecretManager.build_candidate_descriptor(
                    CameraSourceType.http,
                    candidate,
                )
                errors.append(f"{descriptor}: responded with HTML instead of a media stream")
                continue
            return HttpProbeResult(source=candidate, status_code=status_code, content_type=content_type)

        if first_http_failure is not None:
            return first_http_failure
        if errors:
            raise OSError("; ".join(errors))
        raise OSError("Unable to connect to the configured HTTP camera source.")

    @classmethod
    def _resolve_http_stream(cls, camera: Camera, source: str) -> tuple[str, str, list[str]]:
        explicit_stream_url = cls._metadata_value(camera, "stream_url")
        configured_stream_format = (cls._metadata_value(camera, "stream_format") or "").lower()
        if explicit_stream_url:
            stream_format = cls._infer_http_stream_format(explicit_stream_url, configured_stream_format)
            notes = cls._build_http_stream_inference_notes(explicit_stream_url, configured_stream_format, stream_format)
            return explicit_stream_url, stream_format, notes

        if cls._looks_like_private_http_root(source):
            return (
                cls._join_camera_path(source, "video"),
                configured_stream_format or "mjpeg",
                [
                    "Guessed an IP Webcam-compatible /video endpoint from the configured device base URL.",
                    "Set metadata.stream_url if your phone camera uses a different feed path.",
                ],
            )

        stream_format = cls._infer_http_stream_format(source, configured_stream_format)
        notes = cls._build_http_stream_inference_notes(source, configured_stream_format, stream_format)
        return source, stream_format, notes

    @classmethod
    def _http_source_candidates(cls, camera: Camera, source: str) -> list[str]:
        explicit_stream_url = cls._metadata_value(camera, "stream_url")
        if explicit_stream_url:
            return [explicit_stream_url]

        if cls._looks_like_private_http_root(source):
            return [
                cls._join_camera_path(source, "video"),
                cls._join_camera_path(source, "shot.jpg"),
                source,
            ]

        return [source]

    @staticmethod
    def _join_camera_path(source: str, path: str) -> str:
        normalized = source if source.endswith("/") else f"{source}/"
        return urljoin(normalized, path)

    def _probe_http_source_with_headers(self, source: str, skip_tls_verification: bool = False) -> tuple[int, str | None]:
        context = None
        if skip_tls_verification and source.lower().startswith("https://"):
            context = ssl._create_unverified_context()

        request = Request(source, method="HEAD", headers={"User-Agent": "AegisPro/1.0"})
        try:
            with self.network_policy.open_http_url(
                source,
                method="HEAD",
                timeout=3,
                headers=dict(request.header_items()),
                context=context,
            )[0] as response:
                return response.status, response.headers.get("Content-Type")
        except HTTPError as exc:
            if exc.code in {405, 501}:
                request = Request(source, method="GET", headers={"User-Agent": "AegisPro/1.0"})
                with self.network_policy.open_http_url(
                    source,
                    method="GET",
                    timeout=3,
                    headers=dict(request.header_items()),
                    context=context,
                )[0] as response:
                    return response.status, response.headers.get("Content-Type")
            return exc.code, exc.headers.get("Content-Type") if exc.headers else None
        except OSError:
            request = Request(source, method="GET", headers={"User-Agent": "AegisPro/1.0"})
            with self.network_policy.open_http_url(
                source,
                method="GET",
                timeout=3,
                headers=dict(request.header_items()),
                context=context,
            )[0] as response:
                return response.status, response.headers.get("Content-Type")

    @classmethod
    def _looks_like_private_http_root(cls, source: str) -> bool:
        parsed = urlparse(source)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return False
        if parsed.path not in {"", "/"}:
            return False
        return cls._is_private_camera_host(parsed.hostname)

    @staticmethod
    def _is_private_camera_host(hostname: str) -> bool:
        if hostname in {"localhost", "127.0.0.1", "::1"}:
            return True
        try:
            return ipaddress.ip_address(hostname).is_private
        except ValueError:
            return False

    @classmethod
    def _looks_like_private_http_source(cls, source: str) -> bool:
        parsed = urlparse(source)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return False
        return cls._is_private_camera_host(parsed.hostname)

    @classmethod
    def _should_skip_tls_verification(cls, camera: Camera, source: str) -> bool:
        insecure_tls = camera.metadata_.get("insecure_tls")
        if isinstance(insecure_tls, bool):
            return insecure_tls

        parsed = urlparse(source)
        if parsed.scheme != "https" or not parsed.hostname:
            return False
        return cls._is_private_camera_host(parsed.hostname)

    @classmethod
    def _should_proxy_http_stream(cls, camera: Camera, source: str, stream_kind: str) -> bool:
        if stream_kind == "hls":
            return False

        proxy_stream = camera.metadata_.get("proxy_stream")
        if isinstance(proxy_stream, bool):
            return proxy_stream

        return cls._looks_like_private_http_source(source)

    @classmethod
    def _infer_http_stream_format(cls, source: str, configured_stream_format: str) -> str:
        if configured_stream_format:
            return configured_stream_format
        if cls._looks_like_private_phone_mjpeg_source(source):
            return "mjpeg"
        return ""

    @classmethod
    def _build_http_stream_inference_notes(
        cls,
        source: str,
        configured_stream_format: str,
        resolved_stream_format: str,
    ) -> list[str]:
        if configured_stream_format or resolved_stream_format != "mjpeg":
            return []
        if cls._looks_like_private_phone_mjpeg_source(source):
            return ["Inferred an MJPEG phone-camera feed from the configured HTTP path."]
        return []

    @classmethod
    def _looks_like_private_phone_mjpeg_source(cls, source: str) -> bool:
        parsed = urlparse(source)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return False
        if not cls._is_private_camera_host(parsed.hostname):
            return False

        normalized_path = parsed.path.rstrip("/").lower()
        return normalized_path in {"/video", "/mjpeg", "/mjpg"} or normalized_path.endswith((".mjpeg", ".mjpg"))

    def _iter_http_stream(self, source: str, skip_tls_verification: bool = False) -> Iterator[bytes]:
        context = None
        if skip_tls_verification and source.lower().startswith("https://"):
            context = ssl._create_unverified_context()

        response, _ = self.network_policy.open_http_url(
            source,
            method="GET",
            timeout=10,
            headers={"User-Agent": "AegisPro/1.0"},
            context=context,
        )
        with response:
            while True:
                chunk = response.read(64 * 1024)
                if not chunk:
                    break
                yield chunk

    async def _probe_rtsp_source(self, source: str) -> None:
        validated = self.network_policy.validate_rtsp_url(source)
        connection = await asyncio.wait_for(asyncio.open_connection(validated.hostname, validated.port), timeout=3)
        reader, writer = connection
        del reader
        writer.close()
        await writer.wait_closed()

    @staticmethod
    def _stream_kind_from_url(source: str, *, stream_format: str = "") -> str:
        if stream_format == "mjpeg":
            return "image"
        if stream_format == "hls" or source.lower().endswith(".m3u8"):
            return "hls"
        return CameraStreamingService._stream_kind_from_extension(Path(urlparse(source).path).suffix.lower())

    @staticmethod
    def _stream_kind_from_extension(extension: str) -> str:
        if extension in IMAGE_EXTENSIONS:
            return "image"
        if extension == ".m3u8":
            return "hls"
        if extension in VIDEO_EXTENSIONS:
            return "video"
        mime_type, _ = mimetypes.guess_type(f"camera{extension}")
        if mime_type and mime_type.startswith("image/"):
            return "image"
        return "video"
