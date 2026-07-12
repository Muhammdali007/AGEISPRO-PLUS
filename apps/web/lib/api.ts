export type UserRole = "administrator" | "supervisor" | "operator" | "viewer";
export type CameraSourceType = "usb" | "rtsp" | "http" | "file";
export type CameraStatus = "online" | "offline" | "degraded" | "disabled" | "unknown";
export type IncidentPriority = "critical" | "high" | "medium" | "low";
export type IncidentStatus =
  | "open"
  | "acknowledged"
  | "investigating"
  | "resolved"
  | "dismissed";
export type DetectionType =
  | "weapon"
  | "fire"
  | "smoke"
  | "person"
  | "known_person"
  | "unknown_person"
  | "system";
export type AlertStatus = "active" | "acknowledged" | "cleared";
export type RecognitionStatus = "known" | "unknown";
export type MonitoringWindow = "24h" | "7d" | "30d";
export type LiveEventType =
  | "system.connected"
  | "incident.created"
  | "incident.updated"
  | "alert.created"
  | "alert.acknowledged"
  | "alert.cleared";

export type CurrentUser = {
  id: string;
  email: string;
  full_name: string;
  role: UserRole;
  is_active: boolean;
};

export type UserCreateInput = {
  email: string;
  full_name: string;
  role: UserRole;
  password: string;
  is_active: boolean;
};

export type TokenPair = {
  access_token: string;
  refresh_token: string;
  token_type: "bearer";
};

export type SignupInput = {
  email: string;
  full_name: string;
  role: UserRole;
  password: string;
};

export type Readiness = {
  status: string;
  database: string;
  redis: string;
};

export type Camera = {
  id: string;
  name: string;
  source_type: CameraSourceType;
  source: string;
  status: CameraStatus;
  location: string | null;
  group: string | null;
  tags: string[];
  detection_enabled: boolean;
  inference_fps: number;
  metadata: Record<string, unknown>;
  last_seen_at: string | null;
  health_checked_at: string | null;
  created_at: string;
  updated_at: string;
};

export type CameraCreateInput = {
  name: string;
  source_type: CameraSourceType;
  source: string;
  status?: CameraStatus;
  location?: string | null;
  group?: string | null;
  tags?: string[];
  detection_enabled: boolean;
  inference_fps: number;
  metadata?: Record<string, unknown>;
};

export type CameraUpdateInput = Partial<CameraCreateInput> & {
  status?: CameraStatus;
};

export type CameraConnectionTest = {
  camera_id: string;
  status: CameraStatus;
  message: string;
  checked_at: string;
  latency_ms: number | null;
};

export type CameraStreamDescriptor = {
  camera_id: string;
  stream_kind: string;
  stream_url: string | null;
  browser_supported: boolean;
  requires_relay: boolean;
  is_live: boolean;
  health_status: CameraStatus;
  health_message: string;
  checked_at: string | null;
  controls: string[];
  notes: string[];
  browser_device_id: string | null;
};

export type CameraLiveMonitorEntry = {
  camera: Camera;
  stream: CameraStreamDescriptor;
};

export type CameraLiveMonitorSummary = {
  total: number;
  online: number;
  offline: number;
  degraded: number;
  disabled: number;
  unknown: number;
  live: number;
  browser_ready: number;
  relay_required: number;
  detection_enabled: number;
  groups: Record<string, number>;
};

export type CameraLiveMonitorResponse = {
  summary: CameraLiveMonitorSummary;
  entries: CameraLiveMonitorEntry[];
};

export type CameraConnectionBatch = {
  results: CameraConnectionTest[];
};

export type CameraDetectionScanRequest = {
  frame_content_base64?: string;
  frame_content_type?: string;
  include_evidence?: boolean;
  requested_detectors?: string[];
  recognition_enabled?: boolean;
  occurrence_hint?: string;
};

export type CameraDetectionScanSummary = {
  detection_type: string;
  confidence: number;
  track_id: string | null;
  recognition_status: string | null;
  identity_label: string | null;
  bounding_box: DetectionOverlayBox | null;
  face_bounding_box: DetectionOverlayBox | null;
  metadata: Record<string, unknown>;
};

export type DetectionOverlayBox = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  label?: string | null;
};

export type CameraDetectionScanResponse = {
  camera_id: string;
  model_name: string;
  model_version: string;
  detection_count: number;
  incident_count: number;
  alert_count: number;
  ignored_count: number;
  detections: CameraDetectionScanSummary[];
  ignored_reasons: string[];
  backend: string | null;
  callback_delivered: boolean;
};

export type Incident = {
  id: string;
  camera_id: string;
  detection_type: DetectionType;
  priority: IncidentPriority;
  status: IncidentStatus;
  confidence: number;
  occurred_at: string;
  bounding_boxes: Array<Record<string, unknown>>;
  snapshot_path: string | null;
  clip_path: string | null;
  recognized_identity: RecognizedIdentity | null;
  operator_notes: string | null;
  assigned_user_id: string | null;
  metadata: Record<string, unknown>;
  created_at: string;
  updated_at: string;
};

export type RecognizedIdentity = {
  status: RecognitionStatus;
  identity_id: string | null;
  identity_label: string | null;
  match_confidence: number | null;
  face_image_path: string | null;
  face_bounding_box: Record<string, unknown> | null;
};

export type PersonFaceProfile = {
  id: string;
  label: string;
  image_path: string;
  embedding_vector: number[];
  embedding_model: string | null;
  embedding_dimensions: number;
  is_primary: boolean;
  captured_at: string;
  metadata: Record<string, unknown>;
};

export type Person = {
  id: string;
  full_name: string;
  person_type: "employee" | "student" | "visitor" | "contractor" | "other";
  department: string | null;
  reference_id: string;
  title: string | null;
  is_active: boolean;
  face_profiles: PersonFaceProfile[];
  face_image_count: number;
  embedding_count: number;
  visit_count: number;
  recognition_count: number;
  last_seen_at: string | null;
  last_recognized_at: string | null;
  metadata: Record<string, unknown>;
  created_at: string;
  updated_at: string;
};

export type PersonCreateInput = {
  full_name: string;
  person_type: "employee" | "student" | "visitor" | "contractor" | "other";
  department?: string | null;
  reference_id?: string | null;
  title?: string | null;
  is_active: boolean;
  metadata?: Record<string, unknown>;
};

export type IncidentSavePersonInput = PersonCreateInput & {
  is_primary?: boolean;
};

export type PersonUpdateInput = Partial<PersonCreateInput>;
export type IncidentUpdateInput = Partial<{
  priority: IncidentPriority;
  status: IncidentStatus;
  operator_notes: string | null;
  assigned_user_id: string | null;
  metadata: Record<string, unknown>;
}>;

export type PersonFaceEnrollmentInput = {
  image_path: string;
  label?: string | null;
  embedding_vector?: number[];
  embedding_model?: string | null;
  is_primary?: boolean;
  metadata?: Record<string, unknown>;
};

export type PersonFaceUploadInput = {
  files: File[];
  is_primary?: boolean;
};

export type Alert = {
  id: string;
  incident_id: string;
  priority: IncidentPriority;
  status: AlertStatus;
  title: string;
  message: string;
  acknowledged: boolean;
  acknowledged_by_id: string | null;
  acknowledged_at: string | null;
  created_at: string;
  updated_at: string;
};

export type LiveEvent = {
  type: LiveEventType;
  incident_id?: string;
  alert_id?: string;
  camera_id?: string;
  detection_type?: DetectionType;
  priority?: IncidentPriority;
  status?: IncidentStatus;
  role?: UserRole;
};

export type MonitoringKpis = {
  incident_volume: number;
  active_alerts: number;
  online_camera_ratio: number;
  average_confidence: number;
};

export type MonitoringSeriesPoint = {
  bucket: string;
  label: string;
  value: number;
};

export type DetectionMixPoint = {
  detection_type: string;
  count: number;
};

export type CameraHealthSummary = {
  total: number;
  online: number;
  offline: number;
  degraded: number;
  disabled: number;
  unknown: number;
  stale: number;
  detection_enabled: number;
  groups: Record<string, number>;
};

export type CameraHealthEntry = {
  camera_id: string;
  name: string;
  status: CameraStatus;
  group: string | null;
  last_seen_at: string | null;
  health_checked_at: string | null;
  stale: boolean;
  detection_enabled: boolean;
};

export type CameraHealthReport = {
  stale_threshold_minutes: number;
  generated_at: string;
  summary: CameraHealthSummary;
  entries: CameraHealthEntry[];
};

export type SystemDependencyStatus = {
  status: string;
  detail: string | null;
};

export type AiRuntimeHealth = {
  status: string;
  inference_backend: string | null;
  fallback_backend: string | null;
  recognition_backend: string | null;
  recognition_providers: string[];
  model_device: string | null;
  gpu_available: boolean;
  gpu_name: string | null;
  gpu_memory_total_mb: number | null;
  gpu_memory_used_mb: number | null;
  gpu_utilization_percent: number | null;
  telemetry_supported: boolean;
  detail: string | null;
};

export type SystemHealthReport = {
  generated_at: string;
  api: SystemDependencyStatus;
  database: SystemDependencyStatus;
  redis: SystemDependencyStatus;
  ai: AiRuntimeHealth;
};

export type OptimizationResourceSnapshot = {
  incidents_total: number;
  incidents_last_24h: number;
  active_alerts_total: number;
  alerts_last_24h: number;
  audit_logs_total: number;
  audit_logs_last_24h: number;
};

export type DatabaseOptimizationReport = {
  status: string;
  pool_size: number;
  max_overflow: number;
  pool_recycle_seconds: number;
  indexed_paths: string[];
  resources: OptimizationResourceSnapshot;
  detail: string | null;
};

export type RedisOptimizationReport = {
  status: string;
  ping_ms: number | null;
  used_memory_human: string | null;
  connected_clients: number | null;
  pubsub_channels: number | null;
  detail: string | null;
};

export type RuntimeOptimizationReport = {
  status: string;
  inference_backend: string | null;
  recognition_backend: string | null;
  gpu_available: boolean;
  gpu_utilization_percent: number | null;
  gpu_memory_used_mb: number | null;
  gpu_memory_total_mb: number | null;
  detail: string | null;
};

export type OptimizationRecommendation = {
  title: string;
  detail: string;
  severity: "info" | "warning" | "critical";
};

export type OptimizationReport = {
  generated_at: string;
  database: DatabaseOptimizationReport;
  redis: RedisOptimizationReport;
  runtime: RuntimeOptimizationReport;
  recommendations: OptimizationRecommendation[];
};

export type MonitoringOverview = {
  window: MonitoringWindow;
  generated_at: string;
  kpis: MonitoringKpis;
  incidents_over_time: MonitoringSeriesPoint[];
  detection_mix: DetectionMixPoint[];
  camera_health: CameraHealthSummary;
  system_health: SystemHealthReport;
};

export type AuditLogEntry = {
  id: string;
  actor_user_id: string | null;
  actor_email: string | null;
  actor_role: string | null;
  action: string;
  resource_type: string;
  resource_id: string | null;
  metadata: Record<string, unknown>;
  created_at: string;
};

export type AuditLogPage = {
  items: AuditLogEntry[];
  total: number;
  limit: number;
  offset: number;
};

const configuredApiUrl = process.env.NEXT_PUBLIC_API_URL;
const defaultApiUrl = "http://127.0.0.1:8000";
const proxyApiUrl = "/backend";
const apiUnavailableMessage = "API is unavailable. Start the backend on http://127.0.0.1:8000 and try again.";

function resolveApiUrl() {
  if (typeof window === "undefined") {
    return configuredApiUrl ?? defaultApiUrl;
  }

  return proxyApiUrl;
}

function resolveWebSocketUrl(accessToken: string) {
  if (typeof window === "undefined") {
    const baseUrl = configuredApiUrl ?? defaultApiUrl;
    return `${baseUrl.replace(/^http/, "ws")}/api/v1/ws/events?token=${encodeURIComponent(accessToken)}`;
  }

  const url = new URL(
    `${configuredApiUrl ?? `${window.location.origin}${proxyApiUrl}`}/api/v1/ws/events`
  );
  url.protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  url.searchParams.set("token", accessToken);
  return url.toString();
}

async function apiFetch<T>(path: string, init?: RequestInit & { token?: string }): Promise<T> {
  const headers = new Headers(init?.headers);

  if (init?.token) {
    headers.set("Authorization", `Bearer ${init.token}`);
  }

  if (!headers.has("Content-Type") && init?.body) {
    headers.set("Content-Type", "application/json");
  }

  let response: Response;
  try {
    response = await fetch(`${resolveApiUrl()}${path}`, {
      ...init,
      headers,
      cache: "no-store"
    });
  } catch {
    throw new Error(apiUnavailableMessage);
  }

  if (!response.ok) {
    const fallback = response.status === 401 ? "Session expired" : "Request failed";
    let detail = fallback;
    try {
      const payload = (await response.json()) as { detail?: string };
      detail = payload.detail ?? fallback;
    } catch {}
    if (detail === fallback && response.status >= 500) {
      throw new Error(apiUnavailableMessage);
    }
    throw new Error(detail);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return response.json() as Promise<T>;
}

function withQuery(path: string, params?: Record<string, string | undefined>) {
  if (!params) {
    return path;
  }

  const search = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value) {
      search.set(key, value);
    }
  });

  const query = search.toString();
  return query ? `${path}?${query}` : path;
}

export async function login(email: string, password: string): Promise<TokenPair> {
  return apiFetch<TokenPair>("/api/v1/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password })
  });
}

export async function signup(payload: SignupInput): Promise<TokenPair> {
  return apiFetch<TokenPair>("/api/v1/auth/signup", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export async function fetchCurrentUser(accessToken: string): Promise<CurrentUser> {
  return apiFetch<CurrentUser>("/api/v1/auth/me", { token: accessToken });
}

export async function fetchReadiness(): Promise<Readiness> {
  return apiFetch<Readiness>("/api/v1/health/ready");
}

export async function listCameras(accessToken: string) {
  return apiFetch<Camera[]>("/api/v1/cameras", { token: accessToken });
}

export async function getLiveMonitor(
  accessToken: string,
  filters?: Partial<{
    status_filter: CameraStatus;
    group: string;
  }>
) {
  return apiFetch<CameraLiveMonitorResponse>(withQuery("/api/v1/cameras/live-monitor", filters), {
    token: accessToken
  });
}

export async function createCamera(accessToken: string, payload: CameraCreateInput) {
  return apiFetch<Camera>("/api/v1/cameras", {
    method: "POST",
    token: accessToken,
    body: JSON.stringify({
      status: "unknown",
      location: null,
      group: null,
      tags: [],
      metadata: {},
      ...payload
    })
  });
}

export async function getCamera(accessToken: string, cameraId: string) {
  return apiFetch<Camera>(`/api/v1/cameras/${cameraId}`, { token: accessToken });
}

export async function updateCamera(accessToken: string, cameraId: string, payload: CameraUpdateInput) {
  return apiFetch<Camera>(`/api/v1/cameras/${cameraId}`, {
    method: "PATCH",
    token: accessToken,
    body: JSON.stringify(payload)
  });
}

export async function deleteCamera(accessToken: string, cameraId: string) {
  return apiFetch<void>(`/api/v1/cameras/${cameraId}`, {
    method: "DELETE",
    token: accessToken
  });
}

export async function getCameraStream(accessToken: string, cameraId: string) {
  return apiFetch<CameraStreamDescriptor>(`/api/v1/cameras/${cameraId}/stream`, { token: accessToken });
}

export async function testCameraConnection(accessToken: string, cameraId: string) {
  return apiFetch<CameraConnectionTest>(`/api/v1/cameras/${cameraId}/test-connection`, {
    method: "POST",
    token: accessToken
  });
}

export async function runCameraDetectionScan(
  accessToken: string,
  cameraId: string,
  payload: CameraDetectionScanRequest = {}
) {
  return apiFetch<CameraDetectionScanResponse>(`/api/v1/cameras/${cameraId}/scan`, {
    method: "POST",
    token: accessToken,
    body: JSON.stringify({
      include_evidence: true,
      recognition_enabled: true,
      requested_detectors: ["weapon", "person", "fire", "smoke"],
      ...payload
    })
  });
}

export async function testLiveMonitorConnections(
  accessToken: string,
  filters?: Partial<{
    status_filter: CameraStatus;
    group: string;
  }>
) {
  return apiFetch<CameraConnectionBatch>(
    withQuery("/api/v1/cameras/live-monitor/test-connections", filters),
    {
      method: "POST",
      token: accessToken
    }
  );
}

export async function fetchProtectedMedia(accessToken: string, path: string) {
  const response = await fetch(`${resolveApiUrl()}${path}`, {
    headers: {
      Authorization: `Bearer ${accessToken}`
    },
    cache: "no-store"
  });

  if (!response.ok) {
    throw new Error("Unable to load protected media");
  }

  return response.blob();
}

export async function fetchIncidentSnapshot(accessToken: string, incidentId: string) {
  return fetchProtectedMedia(accessToken, `/api/v1/incidents/${incidentId}/snapshot`);
}

export async function fetchIncidentClip(accessToken: string, incidentId: string) {
  return fetchProtectedMedia(accessToken, `/api/v1/incidents/${incidentId}/clip`);
}

export async function listIncidents(
  accessToken: string,
  filters?: Partial<{
    camera_id: string;
    status_filter: IncidentStatus;
    detection_type: DetectionType;
    priority: IncidentPriority;
    assigned_user_id: string;
  }>
) {
  return apiFetch<Incident[]>(withQuery("/api/v1/incidents", filters), { token: accessToken });
}

export async function getIncident(accessToken: string, incidentId: string) {
  return apiFetch<Incident>(`/api/v1/incidents/${incidentId}`, { token: accessToken });
}

export async function updateIncident(accessToken: string, incidentId: string, payload: IncidentUpdateInput) {
  return apiFetch<Incident>(`/api/v1/incidents/${incidentId}`, {
    method: "PATCH",
    token: accessToken,
    body: JSON.stringify(payload)
  });
}

export async function listIncidentAlerts(accessToken: string, incidentId: string) {
  return apiFetch<Alert[]>(`/api/v1/incidents/${incidentId}/alerts`, { token: accessToken });
}

export async function listAlerts(
  accessToken: string,
  filters?: Partial<{
    status_filter: AlertStatus;
    priority: IncidentPriority;
  }>
) {
  return apiFetch<Alert[]>(withQuery("/api/v1/alerts", filters), { token: accessToken });
}

export async function getMonitoringOverview(
  accessToken: string,
  window: MonitoringWindow = "24h"
) {
  return apiFetch<MonitoringOverview>(withQuery("/api/v1/monitoring/overview", { window }), {
    token: accessToken
  });
}

export async function getCameraHealthReport(accessToken: string) {
  return apiFetch<CameraHealthReport>("/api/v1/monitoring/camera-health", {
    token: accessToken
  });
}

export async function getSystemHealthReport(accessToken: string) {
  return apiFetch<SystemHealthReport>("/api/v1/monitoring/system-health", {
    token: accessToken
  });
}

export async function getOptimizationReport(accessToken: string) {
  return apiFetch<OptimizationReport>("/api/v1/monitoring/optimization", {
    token: accessToken
  });
}

export async function listAuditLogs(
  accessToken: string,
  filters?: Partial<{
    action: string;
    actor_email: string;
    resource_type: string;
    limit: string;
    offset: string;
  }>
) {
  return apiFetch<AuditLogPage>(withQuery("/api/v1/monitoring/audit-logs", filters), {
    token: accessToken
  });
}

export async function acknowledgeAlert(accessToken: string, alertId: string) {
  return apiFetch<Alert>(`/api/v1/alerts/${alertId}/acknowledge`, {
    method: "POST",
    token: accessToken
  });
}

export async function clearAlert(accessToken: string, alertId: string) {
  return apiFetch<Alert>(`/api/v1/alerts/${alertId}/clear`, {
    method: "POST",
    token: accessToken
  });
}

export async function listUsers(accessToken: string) {
  return apiFetch<CurrentUser[]>("/api/v1/users", { token: accessToken });
}

export async function createUser(accessToken: string, payload: UserCreateInput) {
  return apiFetch<CurrentUser>("/api/v1/users", {
    method: "POST",
    token: accessToken,
    body: JSON.stringify(payload)
  });
}

export async function listPersons(accessToken: string) {
  return apiFetch<Person[]>("/api/v1/persons", { token: accessToken });
}

export async function createPerson(accessToken: string, payload: PersonCreateInput) {
  return apiFetch<Person>("/api/v1/persons", {
    method: "POST",
    token: accessToken,
    body: JSON.stringify(payload)
  });
}

export async function saveIncidentAsPerson(
  accessToken: string,
  incidentId: string,
  payload: IncidentSavePersonInput
) {
  return apiFetch<Person>(`/api/v1/incidents/${incidentId}/save-person`, {
    method: "POST",
    token: accessToken,
    body: JSON.stringify({
      ...payload,
      is_primary: payload.is_primary ?? true
    })
  });
}

export async function getPerson(accessToken: string, personId: string) {
  return apiFetch<Person>(`/api/v1/persons/${personId}`, { token: accessToken });
}

export async function updatePerson(accessToken: string, personId: string, payload: PersonUpdateInput) {
  return apiFetch<Person>(`/api/v1/persons/${personId}`, {
    method: "PATCH",
    token: accessToken,
    body: JSON.stringify(payload)
  });
}

export async function enrollPersonFace(accessToken: string, personId: string, payload: PersonFaceEnrollmentInput) {
  return apiFetch<Person>(`/api/v1/persons/${personId}/faces`, {
    method: "POST",
    token: accessToken,
    body: JSON.stringify({
      ...payload,
      embedding_vector: payload.embedding_vector ?? [],
      metadata: payload.metadata ?? {}
    })
  });
}

export async function uploadPersonFaceImages(accessToken: string, personId: string, payload: PersonFaceUploadInput) {
  const formData = new FormData();
  payload.files.forEach((file) => {
    formData.append("files", file);
  });
  formData.append("is_primary", String(payload.is_primary ?? false));

  const response = await fetch(`${resolveApiUrl()}/api/v1/persons/${personId}/faces/upload`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${accessToken}`
    },
    body: formData,
    cache: "no-store"
  });

  if (!response.ok) {
    const fallback = response.status === 401 ? "Session expired" : "Unable to upload face images";
    let detail = fallback;
    try {
      const payload = (await response.json()) as { detail?: string };
      detail = payload.detail ?? fallback;
    } catch {}
    if (detail === fallback && response.status >= 500) {
      throw new Error(apiUnavailableMessage);
    }
    throw new Error(detail);
  }

  return response.json() as Promise<Person>;
}

export function subscribeToLiveEvents(
  accessToken: string,
  onEvent: (event: LiveEvent) => void,
  onError?: () => void
) {
  const socket = new WebSocket(resolveWebSocketUrl(accessToken));

  socket.onmessage = (message) => {
    try {
      onEvent(JSON.parse(message.data) as LiveEvent);
    } catch {}
  };
  socket.onerror = () => onError?.();

  return () => {
    socket.close();
  };
}
