from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from app.core.config import settings
from app.models.incident import IncidentPriority, IncidentRetentionClass


@dataclass(frozen=True)
class IncidentRetentionPolicy:
    retention_hours: int | None
    description: str


def get_documented_incident_retention_policies() -> dict[IncidentRetentionClass, IncidentRetentionPolicy]:
    return {
        IncidentRetentionClass.standard: IncidentRetentionPolicy(
            retention_hours=settings.incident_retention_hours,
            description=(
                "Default operational queue retention. Incidents are archived after the base window,"
                " then evidence cleanup is handled asynchronously."
            ),
        ),
        IncidentRetentionClass.extended: IncidentRetentionPolicy(
            retention_hours=settings.incident_extended_retention_hours,
            description=(
                "Extended operational retention for incidents that need extra analyst review before archival."
            ),
        ),
        IncidentRetentionClass.compliance: IncidentRetentionPolicy(
            retention_hours=settings.incident_compliance_retention_hours,
            description=(
                "Longer retention for escalated or compliance-sensitive incidents before archival."
            ),
        ),
        IncidentRetentionClass.manual: IncidentRetentionPolicy(
            retention_hours=None,
            description=(
                "No automatic archival or evidence deletion. Records in this class require an explicit operator action."
            ),
        ),
    }


def resolve_incident_retention_class(
    *,
    requested: IncidentRetentionClass | None,
    priority: IncidentPriority,
) -> IncidentRetentionClass:
    if requested is not None:
        return requested
    if priority is IncidentPriority.critical:
        return IncidentRetentionClass.compliance
    return IncidentRetentionClass.standard


def compute_incident_retention_expiry(
    *,
    retention_class: IncidentRetentionClass,
    occurred_at,
):
    policy = get_documented_incident_retention_policies()[retention_class]
    if policy.retention_hours is None:
        return None
    return occurred_at + timedelta(hours=policy.retention_hours)
