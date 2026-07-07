"use client";

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { EmptyState, MetricCard, SectionCard } from "@/components/dashboard-ui";
import { listCameras, listIncidents, listUsers } from "@/lib/api";
import { useAuthStore } from "@/lib/auth-store";
import { detectionTone, formatDateTime, formatPercent, labelize, statusTone } from "@/lib/format";
import { cn } from "@/lib/cn";

export default function IncidentsPage() {
  const { accessToken } = useAuthStore();
  const incidentsQuery = useQuery({
    queryKey: ["incidents", "list", accessToken],
    queryFn: () => listIncidents(accessToken!),
    enabled: Boolean(accessToken)
  });
  const camerasQuery = useQuery({
    queryKey: ["cameras", "list", accessToken],
    queryFn: () => listCameras(accessToken!),
    enabled: Boolean(accessToken)
  });
  const usersQuery = useQuery({
    queryKey: ["users", "list", accessToken],
    queryFn: () => listUsers(accessToken!),
    enabled: Boolean(accessToken)
  });

  const incidents = incidentsQuery.data ?? [];
  const cameraMap = new Map((camerasQuery.data ?? []).map((camera) => [camera.id, camera.name]));
  const userMap = new Map((usersQuery.data ?? []).map((user) => [user.id, user.full_name]));

  return (
    <div className="space-y-6">
      <section className="grid gap-4 md:grid-cols-4">
        <MetricCard label="Total incidents" value={`${incidents.length}`} detail="All detections and system events logged so far." />
        <MetricCard label="Open" value={`${incidents.filter((incident) => incident.status === "open").length}`} detail="Awaiting first operator action." tone="alert" />
        <MetricCard label="Investigating" value={`${incidents.filter((incident) => incident.status === "investigating").length}`} detail="Active operational follow-up." />
        <MetricCard label="Resolved" value={`${incidents.filter((incident) => incident.status === "resolved").length}`} detail="Closed successfully after review." tone="success" />
      </section>

      <SectionCard
        title="Incident queue"
        description="Operators can browse the API-backed workflow today, then enrich it with live evidence in later phases."
      >
        {incidents.length === 0 ? (
          <EmptyState
            title="No incidents logged"
            description="The queue is empty right now, but the page is already wired to render confidence, assignments, and status once detections arrive."
          />
        ) : (
          <div className="space-y-3">
            {incidents.map((incident) => (
              <Link
                key={incident.id}
                href={`/dashboard/incidents/${incident.id}`}
                className="grid gap-4 rounded-[24px] border border-white/10 bg-black/15 p-4 transition hover:border-accent/30 hover:bg-black/25 xl:grid-cols-[1.25fr_0.8fr_0.8fr_0.7fr_0.9fr]"
              >
                <div>
                  <p className={cn("font-medium", detectionTone(incident.detection_type))}>
                    {labelize(incident.detection_type)}
                  </p>
                  <p className="mt-2 text-sm text-slate-400">
                    {cameraMap.get(incident.camera_id) ?? `Camera ${incident.camera_id.slice(0, 8)}`}
                  </p>
                  {incident.recognized_identity?.identity_label ? (
                    <p className="mt-2 text-sm text-emerald-200">
                      {incident.recognized_identity.status === "known" ? "Matched" : "Unidentified"}:{" "}
                      {incident.recognized_identity.identity_label}
                    </p>
                  ) : null}
                </div>
                <QueueCell label="Occurred" value={formatDateTime(incident.occurred_at)} />
                <QueueCell label="Assigned" value={incident.assigned_user_id ? (userMap.get(incident.assigned_user_id) ?? "Assigned") : "Unassigned"} />
                <QueueCell label="Confidence" value={formatPercent(incident.confidence)} />
                <div className="flex flex-wrap items-center gap-2">
                  <span className={cn("rounded-full px-2.5 py-1 text-xs font-medium", statusTone(incident.priority))}>
                    {labelize(incident.priority)}
                  </span>
                  <span className={cn("rounded-full px-2.5 py-1 text-xs font-medium", statusTone(incident.status))}>
                    {labelize(incident.status)}
                  </span>
                </div>
              </Link>
            ))}
          </div>
        )}
      </SectionCard>
    </div>
  );
}

function QueueCell({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-xs uppercase tracking-[0.18em] text-slate-500">{label}</p>
      <p className="mt-2 text-sm text-slate-200">{value}</p>
    </div>
  );
}
