from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import time


class Handler(BaseHTTPRequestHandler):
    def _send(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send({"status": "ok", "service": "aegispro-ai-stub"})
            return
        if self.path == "/health/runtime":
            self._send(
                {
                    "status": "ok",
                    "inference_backend": "simulated",
                    "recognition_backend": "hash",
                    "gpu_available": False,
                    "detail": "local stub runtime",
                    "capacity": {},
                    "validation_gates": {"status": "local"},
                }
            )
            return
        self._send({"detail": "not found"}, 404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(raw.decode() or "{}")
        except json.JSONDecodeError:
            payload = {}

        result = {
            "camera_id": payload.get("camera_id"),
            "occurred_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model_name": "simulated",
            "model_version": "local-stub",
            "detections": [],
            "metadata": {"stub": True},
            "inference_fps": 1.0,
        }

        if self.path == "/v1/inference/run":
            self._send(result)
            return
        if self.path == "/v1/inference/run-batch":
            items = payload.get("items") or payload.get("requests") or []
            self._send({"results": [dict(result, camera_id=item.get("camera_id")) for item in items]})
            return
        if self.path == "/v1/inference/dispatch":
            self._send({"accepted": True, "dispatched": 0})
            return
        self._send({"detail": "not found"}, 404)


HTTPServer(("0.0.0.0", 8100), Handler).serve_forever()
