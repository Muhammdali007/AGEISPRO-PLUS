import type {
  AlertStatus,
  CameraStatus,
  DetectionType,
  IncidentPriority,
  IncidentStatus,
  UserRole
} from "@/lib/api";

export function formatDateTime(value: string | null) {
  if (!value) {
    return "Not available";
  }

  return new Intl.DateTimeFormat("en-US", {
    dateStyle: "medium",
    timeStyle: "short"
  }).format(new Date(value));
}

export function formatPercent(value: number) {
  return `${Math.round(value * 100)}%`;
}

export function labelize(value: string) {
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function statusTone(
  value: CameraStatus | IncidentPriority | IncidentStatus | AlertStatus | UserRole | "ok" | string
) {
  switch (value) {
    case "ok":
    case "online":
    case "resolved":
    case "viewer":
      return "bg-emerald-500/15 text-emerald-200 ring-1 ring-emerald-400/30";
    case "critical":
    case "high":
    case "open":
    case "active":
    case "administrator":
      return "bg-red-500/15 text-red-200 ring-1 ring-red-400/30";
    case "degraded":
    case "warning":
    case "acknowledged":
    case "investigating":
    case "supervisor":
      return "bg-amber-500/15 text-amber-200 ring-1 ring-amber-400/30";
    case "disabled":
    case "dismissed":
    case "operator":
      return "bg-slate-500/20 text-slate-200 ring-1 ring-slate-400/30";
    case "low":
    case "medium":
    case "cleared":
    case "info":
    case "offline":
    case "unknown":
    default:
      return "bg-cyan-500/15 text-cyan-100 ring-1 ring-cyan-400/30";
  }
}

export function detectionTone(value: DetectionType) {
  switch (value) {
    case "weapon":
      return "text-red-300";
    case "fire":
      return "text-orange-300";
    case "smoke":
      return "text-amber-200";
    case "known_person":
      return "text-emerald-200";
    case "unknown_person":
      return "text-fuchsia-200";
    default:
      return "text-cyan-200";
  }
}
