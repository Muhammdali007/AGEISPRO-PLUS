"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ShieldAlert } from "lucide-react";
import Link from "next/link";
import { EmptyState, InlineLink, MetricCard, SectionCard } from "@/components/dashboard-ui";
import { LiveMonitorGrid } from "@/components/live-monitor-grid";
import {
  getLiveMonitor,
  listAlerts,
  listCameras,
  listIncidents,
  listUsers,
  testLiveMonitorConnections,
  type CameraStatus
} from "@/lib/api";
import { useAuthStore } from "@/lib/auth-store";
import { detectionTone, formatDateTime, formatPercent, labelize, statusTone } from "@/lib/format";
import { cn } from "@/lib/cn";

export default function DashboardOverviewPage() {
  const { accessToken } = useAuthStore();
  const queryClient = useQueryClient();

  const camerasQuery = useQuery({
    queryKey: ["cameras", "list", accessToken],
    queryFn: () => listCameras(accessToken!),
    enabled: Boolean(accessToken)
  });
  const incidentsQuery = useQuery({
    queryKey: ["incidents", "list", accessToken],
    queryFn: () => listIncidents(accessToken!),
    enabled: Boolean(accessToken)
  });
  const alertsQuery = useQuery({
    queryKey: ["alerts", "list", accessToken],
    queryFn: () => listAlerts(accessToken!),
    enabled: Boolean(accessToken)
  });
  const usersQuery = useQuery({
    queryKey: ["users", "list", accessToken],
    queryFn: () => listUsers(accessToken!),
    enabled: Boolean(accessToken)
  });
  const liveMonitorQuery = useQuery({
    queryKey: ["cameras", "live-monitor", accessToken],
    queryFn: () => getLiveMonitor(accessToken!),
    enabled: Boolean(accessToken),
    refetchInterval: 30_000
  });
  const refreshHealthMutation = useMutation({
    mutationFn: async (filters?: { status_filter?: CameraStatus; group?: string }) =>
      testLiveMonitorConnections(accessToken!, filters),
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["cameras", "list", accessToken] }),
        queryClient.invalidateQueries({ queryKey: ["cameras", "live-monitor", accessToken] })
      ]);
    }
  });

  const cameras = camerasQuery.data ?? [];
  const incidents = incidentsQuery.data ?? [];
  const alerts = alertsQuery.data ?? [];
  const users = usersQuery.data ?? [];
  const liveMonitor = liveMonitorQuery.data;

  const activeAlerts = alerts.filter((alert) => alert.status === "active").length;
  const onlineCameras = cameras.filter((camera) => camera.status === "online").length;
  const detectionEnabled = cameras.filter((camera) => camera.detection_enabled).length;
  const openIncidents = incidents.filter((incident) => incident.status === "open").length;
  const recentIncidents = incidents.slice(0, 5);
  const hotAlerts = alerts.slice(0, 4);

  return (
    <div className="space-y-6">
      <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          label="Camera estate"
          value={`${onlineCameras}/${cameras.length || 0}`}
          detail={`${detectionEnabled} cameras have AI detection enabled`}
          tone="success"
        />
        <MetricCard
          label="Active alerts"
          value={`${activeAlerts}`}
          detail="Critical events surfaced for operator attention"
          tone={activeAlerts > 0 ? "alert" : "default"}
        />
        <MetricCard
          label="Open incidents"
          value={`${openIncidents}`}
          detail={`${incidents.length} incidents are available in the workflow queue`}
        />
        <MetricCard
          label="User coverage"
          value={`${users.length}`}
          detail="Role-based access is wired into the frontend shell"
        />
      </section>

      <SectionCard
        title="Live camera grid"
        description="Multi-feed monitoring is now organized for operators with batched stream descriptors, group filtering, and health refreshes across visible feeds."
        action={<InlineLink href="/dashboard/cameras" label="Open full camera registry" />}
      >
        {!accessToken ? (
          <EmptyState
            title="Camera monitoring unavailable"
            description="Sign in again to load live feed descriptors and streaming health data."
          />
        ) : liveMonitorQuery.error instanceof Error ? (
          <EmptyState title="Live monitor unavailable" description={liveMonitorQuery.error.message} />
        ) : liveMonitor ? (
          <LiveMonitorGrid
            accessToken={accessToken}
            entries={liveMonitor.entries}
            summary={liveMonitor.summary}
            refreshing={refreshHealthMutation.isPending}
            onRefreshHealth={(filters) => refreshHealthMutation.mutate(filters)}
          />
        ) : (
          <EmptyState
            title="Preparing multi-feed monitor"
            description="Collecting stream descriptors and health states for the live camera wall."
          />
        )}
      </SectionCard>

      <div className="grid gap-6 xl:grid-cols-[1.55fr_1fr]">
        <SectionCard
          title="Recent incidents"
          description="Newest detections entering the investigation workflow."
          action={<InlineLink href="/dashboard/incidents" label="Open incident queue" />}
        >
          {recentIncidents.length === 0 ? (
            <EmptyState
              title="No incidents yet"
              description="The incident workflow is ready to populate as detections are ingested into the platform."
            />
          ) : (
            <div className="space-y-3">
              {recentIncidents.map((incident) => (
                <Link
                  key={incident.id}
                  href={`/dashboard/incidents/${incident.id}`}
                  className="flex flex-col gap-3 rounded-[22px] border border-white/10 bg-black/15 p-4 transition hover:border-accent/30 hover:bg-black/25 md:flex-row md:items-center md:justify-between"
                >
                  <div>
                    <div className="flex items-center gap-3">
                      <span className={cn("text-sm font-medium", detectionTone(incident.detection_type))}>
                        {labelize(incident.detection_type)}
                      </span>
                      <span className={cn("rounded-full px-2.5 py-1 text-xs font-medium", statusTone(incident.priority))}>
                        {labelize(incident.priority)}
                      </span>
                    </div>
                    <p className="mt-2 text-sm text-slate-400">
                      Camera {incident.camera_id.slice(0, 8)} • {formatDateTime(incident.occurred_at)}
                    </p>
                  </div>
                  <div className="flex items-center gap-4">
                    <p className="text-sm text-slate-300">{formatPercent(incident.confidence)}</p>
                    <span className={cn("rounded-full px-2.5 py-1 text-xs font-medium", statusTone(incident.status))}>
                      {labelize(incident.status)}
                    </span>
                  </div>
                </Link>
              ))}
            </div>
          )}
        </SectionCard>

        <SectionCard
          title="Alert feed"
          description="Active alert volume and escalation status remain visible for operators."
          action={<InlineLink href="/dashboard/analytics" label="Open optimization dashboard" />}
        >
          {hotAlerts.length === 0 ? (
            <EmptyState
              title="Alert channel ready"
              description="The feed is wired to the alerts API and updates as incident workflows create or change alert states."
            />
          ) : (
            <div className="space-y-3">
              {hotAlerts.map((alert) => (
                <div key={alert.id} className="rounded-[22px] border border-white/10 bg-black/15 p-4">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <p className="font-medium">{alert.title}</p>
                      <p className="mt-1 text-sm text-slate-400">{alert.message}</p>
                    </div>
                    <ShieldAlert className="shrink-0 text-red-300" size={18} aria-hidden="true" />
                  </div>
                  <div className="mt-4 flex items-center justify-between">
                    <span className={cn("rounded-full px-2.5 py-1 text-xs font-medium", statusTone(alert.priority))}>
                      {labelize(alert.priority)}
                    </span>
                    <span className={cn("rounded-full px-2.5 py-1 text-xs font-medium", statusTone(alert.status))}>
                      {labelize(alert.status)}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </SectionCard>
      </div>
    </div>
  );
}
