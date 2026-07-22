from __future__ import annotations

import json
import sys

from app.services.media_agent import MediaAgentError, run_media_request


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
        result = run_media_request(payload)
    except MediaAgentError as exc:
        sys.stdout.write(json.dumps({"error": exc.message, "status_code": exc.status_code}))
        return 1
    except Exception as exc:
        sys.stdout.write(json.dumps({"error": str(exc), "status_code": 502}))
        return 1

    sys.stdout.write(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
