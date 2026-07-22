"use client";

import { useQuery } from "@tanstack/react-query";
import { Cpu, HardDriveDownload, ShieldAlert } from "lucide-react";
import { useState } from "react";
import { EmptyState, MetricCard, SectionCard } from "@/components/dashboard-ui";
import {
  getCameraHealthReport,
  getMonitoringOverview,
  getOptimizationReport,
  getSystemHealthReport,
  listAuditLogs,
  type CameraHealthEntry,
  type MonitoringWindow
} from "@/lib/api";
import { useAuthStore } from "@/lib/auth-store";
import { cn } from "@/lib/cn";
import { formatDateTime, formatPercent, labelize, statusTone } from "@/lib/format";

const windows: MonitoringWindow[] = ["24h", "7d", "30d"];

export default function AnalyticsPage() {
  const { accessToken, user } = useAuthStore();
  const [window, setWindow] = useState<MonitoringWindow>("24h");
  const [actionFilter, setActionFilter] = useState("");
  const [resourceTypeFilter, setResourceTypeFilter] = useState("");
  const canViewAuditLogs = user?.role === "administrator" || user?.role === "supervisor";

  const overviewQuery = useQuery({
    queryKey: ["monitoring", "overview", accessToken, window],
    queryFn: () => getMonitoringOverview(accessToken!, window),
    enabled: Boolean(accessToken)
  });
  const cameraHealthQuery = useQuery({
    queryKey: ["monitoring", "camera-health", accessToken],
    queryFn: () => getCameraHealthReport(accessToken!),
    enabled: Boolean(accessToken)
  });
  const systemHealthQuery = useQuery({
    queryKey: ["monitoring", "system-health", accessToken],
    queryFn: () => getSystemHealthReport(accessToken!),
    enabled: Boolean(accessToken)
  });
  const optimizationQuery = useQuery({
    queryKey: ["monitoring", "optimization", accessToken],
    queryFn: () => getOptimizationReport(accessToken!),
    enabled: Boolean(accessToken)
  });
  const auditLogsQuery = useQuery({
    queryKey: ["monitoring", "audit-logs", accessToken, actionFilter, resourceTypeFilter],
    queryFn: () =>
      listAuditLogs(accessToken!, {
        action: actionFilter || undefined,
        resource_type: resourceTypeFilter || undefined,
        limit: "12"
      }),
    enabled: Boolean(accessToken && canViewAuditLogs)
  });

  const overview = overviewQuery.data;
  const cameraHealth = cameraHealthQuery.data;
  const systemHealth = systemHealthQuery.data ?? overview?.system_health;
  const optimization = optimizationQuery.data;
  const auditPage = auditLogsQuery.data;

  if (!accessToken) {
    return (
      <EmptyState
        title="Monitoring unavailable"
        description="Sign in again to load operational analytics, health telemetry, and audit activity."
      />
    );
  }

  if (
    overviewQuery.error instanceof Error ||
    cameraHealthQuery.error instanceof Error ||
    systemHealthQuery.error instanceof Error ||
    optimizationQuery.error instanceof Error ||
    (canViewAuditLogs && auditLogsQuery.error instanceof Error)
  ) {
    return (
      <EmptyState
        title="Monitoring unavailable"
        description={
          overviewQuery.error instanceof Error
            ? overviewQuery.error.message
            : cameraHealthQuery.error instanceof Error
              ? cameraHealthQuery.error.message
              : systemHealthQuery.error instanceof Error
                ? systemHealthQuery.error.message
                : optimizationQuery.error instanceof Error
                  ? optimizationQuery.error.message
                : canViewAuditLogs && auditLogsQuery.error instanceof Error
                  ? auditLogsQuery.error.message
                  : "Unable to load monitoring data."
        }
      />
    );
  }

  if (!overview || !cameraHealth || !systemHealth || !optimization || (canViewAuditLogs && !auditPage)) {
    return (
      <EmptyState
        title="Preparing optimization telemetry"
        description="Collecting incident trends, camera telemetry, runtime health, cache performance, and audit history."
      />
    );
  }

  const kpis = overview.kpis;
  const topCameraEntries = cameraHealth.entries.slice(0, 5);

  return (
    <div className="space-y-6">
      <section className="flex flex-wrap items-center justify-between gap-3 rounded-[24px] border border-white/10 bg-panel/60 px-4 py-4">
        <div>
          <p className="text-xs uppercase tracking-[0.22em] text-slate-500">Phase 9</p>
          <h2 className="mt-1 text-xl font-semibold">Operational monitoring and optimization</h2>
          <p className="mt-1 text-sm text-slate-400">
            Incident trends, camera health, service readiness, cache telemetry, and production hardening signals.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          {windows.map((option) => (
            <button
              key={option}
              type="button"
              onClick={() => setWindow(option)}
              className={cn(
                "rounded-full border px-4 py-2 text-sm transition",
                option === window
                  ? "border-accent/40 bg-accent/15 text-emerald-100"
                  : "border-white/10 bg-black/20 text-slate-300 hover:border-white/20 hover:text-white"
              )}
            >
              {option}
            </button>
          ))}
        </div>
      </section>

      <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          label="DB pool"
          value={`${optimization.database.pool_size}+${optimization.database.max_overflow}`}
          detail={`Recycle every ${optimization.database.pool_recycle_seconds}s.`}
        />
        <MetricCard
          label="Redis latency"
          value={optimization.redis.ping_ms !== null ? `${optimization.redis.ping_ms} ms` : "Unavailable"}
          detail={optimization.redis.used_memory_human ? `Memory in use: ${optimization.redis.used_memory_human}` : "Redis memory not reported."}
          tone={optimization.redis.status === "ok" ? "success" : "alert"}
        />
        <MetricCard
          label="24h incident load"
          value={`${optimization.database.resources.incidents_last_24h}`}
          detail={`${optimization.database.resources.incidents_total} incidents persisted overall.`}
        />
        <MetricCard
          label="Audit trail volume"
          value={`${optimization.database.resources.audit_logs_last_24h}`}
          detail={`${optimization.database.resources.audit_logs_total} audit events stored overall.`}
        />
      </section>

      <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          label="Incident volume"
          value={`${kpis.incident_volume}`}
          detail={`Detected over the selected ${window} window.`}
        />
        <MetricCard
          label="Active alerts"
          value={`${kpis.active_alerts}`}
          detail="Alerts still waiting for operator acknowledgement or clearing."
          tone={kpis.active_alerts > 0 ? "alert" : "default"}
        />
        <MetricCard
          label="Online camera ratio"
          value={formatPercent(kpis.online_camera_ratio)}
          detail={`${cameraHealth.summary.online} of ${cameraHealth.summary.total} cameras are online.`}
          tone="success"
        />
        <MetricCard
          label="Average confidence"
          value={formatPercent(kpis.average_confidence)}
          detail="Average confidence across incidents in the selected window."
        />
      </section>

      <div className="grid gap-6 xl:grid-cols-[1.5fr_1fr]">
        <SectionCard
          title="Incidents over time"
          description="Chart-ready volume buckets derived from persisted incident records."
        >
          <BarSeries
            items={overview.incidents_over_time.map((point) => ({
              label: point.label,
              value: point.value
            }))}
            emptyLabel="No incidents detected in this window."
          />
        </SectionCard>

        <SectionCard
          title="Detection mix"
          description="Current detection composition across the selected reporting window."
        >
          {overview.detection_mix.length === 0 ? (
            <EmptyState
              title="No incident mix available"
              description="The mix view will populate as incidents are recorded inside the selected window."
            />
          ) : (
            <div className="space-y-3">
              {overview.detection_mix.map((item) => (
                <div key={item.detection_type} className="rounded-[20px] border border-white/10 bg-black/15 px-4 py-3">
                  <div className="flex items-center justify-between gap-3">
                    <span className="text-sm text-slate-300">{labelize(item.detection_type)}</span>
                    <span className="text-lg font-semibold">{item.count}</span>
                  </div>
                  <div className="mt-3 h-2 rounded-full bg-white/5">
                    <div
                      className="h-2 rounded-full bg-gradient-to-r from-emerald-300 via-cyan-300 to-orange-300"
                      style={{
                        width: `${Math.max(
                          10,
                          Math.round((item.count / Math.max(...overview.detection_mix.map((entry) => entry.count), 1)) * 100)
                        )}%`
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          )}
        </SectionCard>
      </div>

      <div className="grid gap-6 xl:grid-cols-[1.15fr_0.85fr]">
        <SectionCard
          title="Camera health"
          description={`Stale cameras are those not seen within ${cameraHealth.stale_threshold_minutes} minutes.`}
        >
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            <HealthStat label="Online" value={`${cameraHealth.summary.online}`} tone="success" />
            <HealthStat label="Degraded" value={`${cameraHealth.summary.degraded}`} tone="warning" />
            <HealthStat label="Offline" value={`${cameraHealth.summary.offline}`} tone="alert" />
            <HealthStat label="Stale" value={`${cameraHealth.summary.stale}`} tone="muted" />
          </div>
          <div className="mt-4 space-y-3">
            {topCameraEntries.map((camera) => (
              <CameraHealthRow key={camera.camera_id} camera={camera} />
            ))}
          </div>
        </SectionCard>

        <SectionCard
          title="System health"
          description="Aggregated service readiness from the API monitoring layer."
        >
          <div className="space-y-3">
            <SystemStatusRow label="API" value={systemHealth.api.status} detail={systemHealth.api.detail} />
            <SystemStatusRow label="PostgreSQL" value={systemHealth.database.status} detail={systemHealth.database.detail} />
            <SystemStatusRow label="Redis" value={systemHealth.redis.status} detail={systemHealth.redis.detail} />
            <SystemStatusRow label="AI service" value={systemHealth.ai.status} detail={systemHealth.ai.detail} />
          </div>

          <div className="mt-5 rounded-[22px] border border-white/10 bg-black/15 p-4">
            <div className="flex items-center gap-2 text-slate-200">
              <Cpu size={16} aria-hidden="true" className="text-accent" />
              Runtime telemetry
            </div>
            <div className="mt-4 grid gap-3 sm:grid-cols-2">
              <TelemetryCell label="Inference backend" value={labelize(systemHealth.ai.inference_backend ?? "unavailable")} />
              <TelemetryCell label="Recognition backend" value={labelize(systemHealth.ai.recognition_backend ?? "unavailable")} />
              <TelemetryCell label="GPU available" value={systemHealth.ai.gpu_available ? "Yes" : "No"} />
              <TelemetryCell label="GPU name" value={systemHealth.ai.gpu_name ?? "Not reported"} />
              <TelemetryCell
                label="GPU memory"
                value={
                  systemHealth.ai.gpu_memory_total_mb
                    ? `${systemHealth.ai.gpu_memory_used_mb ?? 0}/${systemHealth.ai.gpu_memory_total_mb} MB`
                    : "Not reported"
                }
              />
              <TelemetryCell
                label="Providers"
                value={systemHealth.ai.recognition_providers.length ? systemHealth.ai.recognition_providers.join(", ") : "Not reported"}
              />
            </div>
          </div>
        </SectionCard>
      </div>

      <div className="grid gap-6 xl:grid-cols-[1.1fr_0.9fr]">
        <SectionCard
          title="Optimization report"
          description="Phase 9 hardening telemetry for database, Redis, and AI runtime behavior."
        >
          <div className="grid gap-4 md:grid-cols-2">
            <TelemetryCell label="Indexed paths" value={`${optimization.database.indexed_paths.length}`} />
            <TelemetryCell label="Redis clients" value={`${optimization.redis.connected_clients ?? 0}`} />
            <TelemetryCell label="Pub/Sub channels" value={`${optimization.redis.pubsub_channels ?? 0}`} />
            <TelemetryCell
              label="GPU utilization"
              value={
                optimization.runtime.gpu_utilization_percent !== null
                  ? `${optimization.runtime.gpu_utilization_percent}%`
                  : "Not reported"
              }
            />
          </div>
          <p className="mt-4 text-sm text-slate-400">{optimization.database.detail}</p>
        </SectionCard>

        <SectionCard
          title="Optimization recommendations"
          description="Actionable follow-ups surfaced from current production-readiness telemetry."
        >
          <div className="space-y-3">
            {optimization.recommendations.map((recommendation) => (
              <div key={recommendation.title} className="rounded-[20px] border border-white/10 bg-black/15 px-4 py-3">
                <div className="flex items-center justify-between gap-3">
                  <p className="font-medium text-slate-100">{recommendation.title}</p>
                  <span className={cn("rounded-full px-2.5 py-1 text-xs font-medium", statusTone(recommendation.severity))}>
                    {labelize(recommendation.severity)}
                  </span>
                </div>
                <p className="mt-2 text-sm text-slate-400">{recommendation.detail}</p>
              </div>
            ))}
          </div>
        </SectionCard>
      </div>

      <SectionCard
        title="Recent audit log"
        description="Operator and administrator actions persisted for operational review."
      >
        {!canViewAuditLogs ? (
          <EmptyState
            title="Audit log access restricted"
            description="Only administrators and supervisors can review persisted audit activity."
          />
        ) : (
          <>
            <div className="mb-4 grid gap-3 md:grid-cols-3">
              <label className="block">
                <span className="mb-2 block text-xs uppercase tracking-[0.18em] text-slate-500">Action</span>
                <input
                  className="h-11 w-full rounded-md border border-white/10 bg-black/20 px-3 text-sm outline-none transition focus:border-accent"
                  value={actionFilter}
                  onChange={(event) => setActionFilter(event.target.value)}
                  placeholder="alerts.clear"
                />
              </label>
              <label className="block">
                <span className="mb-2 block text-xs uppercase tracking-[0.18em] text-slate-500">Resource</span>
                <input
                  className="h-11 w-full rounded-md border border-white/10 bg-black/20 px-3 text-sm outline-none transition focus:border-accent"
                  value={resourceTypeFilter}
                  onChange={(event) => setResourceTypeFilter(event.target.value)}
                  placeholder="camera"
                />
              </label>
              <div className="rounded-[20px] border border-white/10 bg-black/15 px-4 py-3 text-sm text-slate-300">
                <div className="flex items-center gap-2">
                  <HardDriveDownload size={16} aria-hidden="true" className="text-accent" />
                  {auditPage?.total ?? 0} audit events matched
                </div>
                <p className="mt-2 text-slate-400">Filters update the persisted log feed without exporting data outside the platform.</p>
              </div>
            </div>

            {auditPage && auditPage.items.length === 0 ? (
              <EmptyState
                title="No audit events matched"
                description="Try a broader filter or generate a new workflow action to populate the audit trail."
              />
            ) : (
              <div className="space-y-3">
                {auditPage?.items.map((entry) => (
                  <div key={entry.id} className="rounded-[20px] border border-white/10 bg-black/15 px-4 py-4">
                    <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                      <div>
                        <div className="flex flex-wrap items-center gap-2">
                          <span className={cn("rounded-full px-2.5 py-1 text-xs font-medium", statusTone(entry.actor_role ?? "viewer"))}>
                            {labelize(entry.actor_role ?? "system")}
                          </span>
                          <span className="rounded-full bg-black/20 px-2.5 py-1 text-xs font-medium text-slate-200 ring-1 ring-white/10">
                            {entry.action}
                          </span>
                          <span className="rounded-full bg-black/20 px-2.5 py-1 text-xs font-medium text-slate-200 ring-1 ring-white/10">
                            {labelize(entry.resource_type)}
                          </span>
                        </div>
                        <p className="mt-2 text-sm text-slate-300">{entry.actor_email ?? "System action"}</p>
                      </div>
                      <div className="text-sm text-slate-400">{formatDateTime(entry.created_at)}</div>
                    </div>
                    {entry.resource_id ? (
                      <p className="mt-3 text-sm text-slate-400">Resource ID: {entry.resource_id}</p>
                    ) : null}
                  </div>
                ))}
              </div>
            )}
          </>
        )}
      </SectionCard>
    </div>
  );
}

function BarSeries({
  items,
  emptyLabel
}: {
  items: Array<{ label: string; value: number }>;
  emptyLabel: string;
}) {
  const peak = Math.max(...items.map((item) => item.value), 0);
  if (peak === 0) {
    return <EmptyState title="No trend data available" description={emptyLabel} />;
  }

  return (
    <div className="grid grid-cols-6 gap-3 md:grid-cols-12 lg:grid-cols-24">
      {items.map((item) => (
        <div key={item.label} className="flex flex-col items-center gap-3">
          <div className="flex h-48 w-full items-end rounded-[22px] border border-white/10 bg-black/15 p-2">
            <div
              className="w-full rounded-[16px] bg-gradient-to-t from-emerald-400 via-cyan-300 to-orange-300"
              style={{ height: `${Math.max(8, Math.round((item.value / peak) * 100))}%` }}
            />
          </div>
          <div className="text-center">
            <p className="text-sm font-medium text-slate-200">{item.value}</p>
            <p className="text-xs text-slate-500">{item.label}</p>
          </div>
        </div>
      ))}
    </div>
  );
}

function HealthStat({
  label,
  value,
  tone
}: {
  label: string;
  value: string;
  tone: "success" | "warning" | "alert" | "muted";
}) {
  const toneClass =
    tone === "success"
      ? "bg-emerald-500/10 text-emerald-100"
      : tone === "warning"
        ? "bg-amber-500/10 text-amber-100"
        : tone === "alert"
          ? "bg-red-500/10 text-red-100"
          : "bg-black/15 text-slate-100";

  return (
    <div className={cn("rounded-[22px] border border-white/10 p-4", toneClass)}>
      <p className="text-xs uppercase tracking-[0.18em] text-slate-400">{label}</p>
      <p className="mt-3 text-2xl font-semibold">{value}</p>
    </div>
  );
}

function CameraHealthRow({ camera }: { camera: CameraHealthEntry }) {
  return (
    <div className="flex flex-col gap-3 rounded-[20px] border border-white/10 bg-black/15 px-4 py-3 md:flex-row md:items-center md:justify-between">
      <div>
        <div className="flex flex-wrap items-center gap-2">
          <p className="font-medium">{camera.name}</p>
          {camera.stale ? (
            <span className="inline-flex items-center gap-1 rounded-full bg-red-500/15 px-2.5 py-1 text-xs font-medium text-red-100 ring-1 ring-red-400/30">
              <ShieldAlert size={12} aria-hidden="true" />
              Stale
            </span>
          ) : null}
        </div>
        <p className="mt-1 text-sm text-slate-400">
          {camera.group ?? "Ungrouped"} | Last seen {formatDateTime(camera.last_seen_at)}
        </p>
      </div>
      <span className={cn("rounded-full px-2.5 py-1 text-xs font-medium", statusTone(camera.status))}>
        {labelize(camera.status)}
      </span>
    </div>
  );
}

function SystemStatusRow({
  label,
  value,
  detail
}: {
  label: string;
  value: string;
  detail: string | null;
}) {
  return (
    <div className="flex items-start justify-between gap-3 rounded-[20px] border border-white/10 bg-black/15 px-4 py-3">
      <div>
        <p className="text-sm font-medium">{label}</p>
        <p className="mt-1 text-sm text-slate-400">{detail ?? "Healthy"}</p>
      </div>
      <span className={cn("rounded-full px-2.5 py-1 text-xs font-medium", statusTone(value))}>{labelize(value)}</span>
    </div>
  );
}

function TelemetryCell({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-[18px] border border-white/10 bg-black/20 px-3 py-3">
      <p className="text-xs uppercase tracking-[0.18em] text-slate-500">{label}</p>
      <p className="mt-2 text-sm text-slate-200">{value}</p>
    </div>
  );
}
