from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REQUIRED_RUNTIME_GATES = ("load", "soak_8h", "soak_24h", "soak_72h")


def load_runtime_gate_report(path: str | None) -> dict[str, Any]:
    gates = {
        gate: {
            "name": gate,
            "status": "missing",
            "detail": "No gate report has been configured.",
        }
        for gate in REQUIRED_RUNTIME_GATES
    }
    payload: dict[str, Any] = {
        "status": "missing",
        "path": path,
        "generated_at": None,
        "suite_id": None,
        "gates": gates,
        "detail": "No gate report has been configured.",
    }
    if not path:
        return payload

    report_path = Path(path)
    if not report_path.is_file():
        payload["detail"] = f"Gate report file not found: {report_path}"
        return payload

    try:
        raw = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        payload["status"] = "invalid"
        payload["detail"] = f"Unable to read gate report: {exc}"
        return payload

    gate_payload = raw.get("gates") if isinstance(raw, dict) else None
    if not isinstance(gate_payload, dict):
        payload["status"] = "invalid"
        payload["detail"] = "Gate report must contain a top-level 'gates' object."
        return payload

    statuses: list[str] = []
    for gate in REQUIRED_RUNTIME_GATES:
        gate_entry = gate_payload.get(gate)
        if not isinstance(gate_entry, dict):
            payload["gates"][gate] = {
                "name": gate,
                "status": "missing",
                "detail": "Gate result is missing from the report.",
            }
            statuses.append("missing")
            continue

        status = str(gate_entry.get("status") or "missing").strip().lower()
        if status not in {"pass", "fail", "missing"}:
            status = "invalid"

        payload["gates"][gate] = {
            "name": gate,
            "status": status,
            "completed_at": gate_entry.get("completed_at"),
            "duration_hours": gate_entry.get("duration_hours"),
            "detail": gate_entry.get("detail"),
            "metrics": gate_entry.get("metrics") if isinstance(gate_entry.get("metrics"), dict) else {},
        }
        statuses.append(status)

    payload["generated_at"] = raw.get("generated_at") if isinstance(raw, dict) else None
    payload["suite_id"] = raw.get("suite_id") if isinstance(raw, dict) else None
    if any(status in {"fail", "invalid"} for status in statuses):
        payload["status"] = "fail"
        payload["detail"] = "One or more runtime validation gates are failing."
    elif any(status == "missing" for status in statuses):
        payload["status"] = "missing"
        payload["detail"] = "One or more runtime validation gates are missing."
    else:
        payload["status"] = "pass"
        payload["detail"] = "All configured runtime validation gates are passing."

    return payload
