from __future__ import annotations

import fnmatch
import ipaddress
import socket
import ssl
from dataclasses import dataclass
from urllib.error import HTTPError
from urllib.parse import urljoin, urlparse
from urllib.request import HTTPRedirectHandler, HTTPSHandler, Request, build_opener

from app.core.config import settings

REDIRECT_STATUS_CODES = {301, 302, 303, 307, 308}
DEFAULT_PORTS = {
    "http": 80,
    "https": 443,
    "rtsp": 554,
}


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


@dataclass(frozen=True, slots=True)
class ValidatedCameraUrl:
    url: str
    scheme: str
    hostname: str
    port: int
    addresses: tuple[str, ...]


class CameraNetworkPolicy:
    def __init__(self) -> None:
        self.allowed_protocols = {item.lower() for item in settings.camera_allowed_protocols}
        self.allowed_ports = set(settings.camera_allowed_ports)
        self.allowed_networks = tuple(
            ipaddress.ip_network(item, strict=False) for item in settings.camera_allowed_networks
        )
        self.blocked_networks = tuple(
            ipaddress.ip_network(item, strict=False) for item in settings.camera_blocked_networks
        )
        self.allowed_hostnames = tuple(item.lower() for item in settings.camera_allowed_hostnames)
        self.max_redirects = settings.camera_max_redirects

    def validate_url(self, source: str, *, allowed_protocols: set[str] | None = None) -> ValidatedCameraUrl:
        parsed = urlparse(source)
        scheme = parsed.scheme.lower()
        protocols = allowed_protocols or self.allowed_protocols
        if scheme not in protocols:
            raise ValueError(f"Camera source scheme '{scheme or 'unknown'}' is not allowed.")
        if not parsed.hostname:
            raise ValueError("Camera source must include a resolvable host.")

        hostname = parsed.hostname.rstrip(".").lower()
        port = parsed.port or DEFAULT_PORTS.get(scheme)
        if port is None:
            raise ValueError(f"Camera source scheme '{scheme}' is missing a supported default port.")
        if port not in self.allowed_ports:
            raise ValueError(f"Camera source port {port} is not allowlisted.")

        addresses = self._resolve_host(hostname)
        if not addresses:
            raise ValueError(f"Camera source host '{hostname}' did not resolve to any IP addresses.")

        rendered_addresses = tuple(str(address) for address in addresses)
        if self._hostname_is_allowlisted(hostname):
            for address in addresses:
                self._ensure_not_blocked(address, hostname)
            return ValidatedCameraUrl(source, scheme, hostname, port, rendered_addresses)

        for address in addresses:
            self._ensure_allowed(address, hostname)

        return ValidatedCameraUrl(source, scheme, hostname, port, rendered_addresses)

    def open_http_url(
        self,
        source: str,
        *,
        method: str,
        timeout: int,
        headers: dict[str, str] | None = None,
        context: ssl.SSLContext | None = None,
    ):
        current_url = source
        redirects_followed = 0

        while True:
            self.validate_url(current_url, allowed_protocols={"http", "https"})
            opener = build_opener(_NoRedirectHandler, HTTPSHandler(context=context))
            request = Request(current_url, method=method, headers=headers or {})

            try:
                response = opener.open(request, timeout=timeout)
            except HTTPError as exc:
                if exc.code not in REDIRECT_STATUS_CODES:
                    raise
                current_url = self._follow_redirect(current_url, exc.headers.get("Location"), redirects_followed)
                redirects_followed += 1
                exc.close()
                continue

            status_code = getattr(response, "status", response.getcode())
            if status_code not in REDIRECT_STATUS_CODES:
                return response, current_url

            location = response.headers.get("Location")
            response.close()
            current_url = self._follow_redirect(current_url, location, redirects_followed)
            redirects_followed += 1

    def validate_rtsp_url(self, source: str) -> ValidatedCameraUrl:
        return self.validate_url(source, allowed_protocols={"rtsp"})

    def _follow_redirect(self, current_url: str, location: str | None, redirects_followed: int) -> str:
        if not location:
            raise OSError("Camera source returned a redirect without a Location header.")
        if redirects_followed >= self.max_redirects:
            raise OSError("Camera source exceeded the configured redirect limit.")
        redirected_url = urljoin(current_url, location)
        self.validate_url(redirected_url, allowed_protocols={"http", "https"})
        return redirected_url

    @staticmethod
    def _resolve_host(hostname: str) -> tuple[ipaddress._BaseAddress, ...]:
        try:
            infos = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
        except socket.gaierror as exc:
            raise ValueError(f"Unable to resolve camera source host '{hostname}'.") from exc

        addresses: list[ipaddress._BaseAddress] = []
        for family, _, _, _, sockaddr in infos:
            if family not in {socket.AF_INET, socket.AF_INET6}:
                continue
            address_text = sockaddr[0]
            address = ipaddress.ip_address(address_text)
            if address not in addresses:
                addresses.append(address)
        return tuple(addresses)

    def _ensure_allowed(self, address: ipaddress._BaseAddress, hostname: str) -> None:
        self._ensure_not_blocked(address, hostname)
        if self.allowed_networks and any(address in network for network in self.allowed_networks):
            return
        raise ValueError(
            f"Camera source host '{hostname}' resolved to non-allowlisted address {address}."
        )

    def _ensure_not_blocked(self, address: ipaddress._BaseAddress, hostname: str) -> None:
        if any(address in network for network in self.blocked_networks):
            raise ValueError(f"Camera source host '{hostname}' resolved to blocked address {address}.")
        if (
            address.is_link_local
            or address.is_loopback
            or address.is_multicast
            or address.is_unspecified
            or address.is_reserved
        ):
            raise ValueError(f"Camera source host '{hostname}' resolved to blocked address {address}.")

    def _hostname_is_allowlisted(self, hostname: str) -> bool:
        return any(fnmatch.fnmatch(hostname, pattern) for pattern in self.allowed_hostnames)
