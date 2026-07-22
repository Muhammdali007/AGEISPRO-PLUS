"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { useState } from "react";
import { Button } from "@/components/button";
import { EmptyState, MetricCard, SectionCard } from "@/components/dashboard-ui";
import { deleteIncident, listCameras, listIncidents, listUsers } from "@/lib/api";
import { useAuthStore } from "@/lib/auth-store";
import { detectionDisplayLabel, detectionTone, formatDateTime, formatPercent, labelize, statusTone } from "@/lib/format";
import { cn } from "@/lib/cn";

export default function IncidentsPage() {
  const queryClient = useQueryClient();
  const { accessToken, user } = useAuthStore();
  const [deleteError, setDeleteError] = useState<string | null>(null);
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
  const canDeleteIncident = user?.role === "administrator" || user?.role === "supervisor";

  const deleteIncidentMutation = useMutation({
    mutationFn: async (incidentId: string) => {
      if (!accessToken) {
        throw new Error("You need to sign in again before archiving incidents.");
      }
      return deleteIncident(accessToken, incidentId);
    },
    onSuccess: async () => {
      setDeleteError(null);
      await queryClient.invalidateQueries({ queryKey: ["incidents", "list", accessToken] });
      await queryClient.invalidateQueries({ queryKey: ["alerts", "list", accessToken] });
    },
    onError: (cause) => {
      setDeleteError(cause instanceof Error ? cause.message : "Unable to archive incident");
    }
  });

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
        description="Incidents follow retention classes, can be placed on legal hold, and are archived before evidence cleanup runs asynchronously."
      >
        {incidents.length === 0 ? (
          <EmptyState
            title="No incidents logged"
            description="The queue is empty right now, but the page is already wired to render confidence, assignments, and status once detections arrive."
          />
        ) : (
          <div className="space-y-3">
            {deleteError ? (
              <div className="rounded-md border border-danger/50 bg-danger/10 px-3 py-2 text-sm text-red-200">
                {deleteError}
              </div>
            ) : null}
            {incidents.map((incident) => (
              <div
                key={incident.id}
                className="grid gap-4 rounded-[24px] border border-white/10 bg-black/15 p-4 transition hover:border-accent/30 hover:bg-black/25 xl:grid-cols-[1.25fr_0.8fr_0.8fr_0.7fr_0.9fr]"
              >
                <Link href={`/dashboard/incidents/${incident.id}`} className="contents">
                  <div>
                    <p className={cn("font-medium", detectionTone(incident.detection_type))}>
                      {detectionDisplayLabel(incident.detection_type, incident.bounding_boxes)}
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
                  <QueueCell
                    label="Assigned"
                    value={incident.assigned_user_id ? (userMap.get(incident.assigned_user_id) ?? "Assigned") : "Unassigned"}
                  />
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
                {canDeleteIncident ? (
                  <div className="xl:col-span-5 xl:flex xl:justify-end">
                    <Button
                      type="button"
                      variant="ghost"
                      className="border-danger/50 text-red-200 hover:bg-danger/10"
                      disabled={deleteIncidentMutation.isPending}
                      onClick={() => {
                        setDeleteError(null);
                        void deleteIncidentMutation.mutateAsync(incident.id);
                      }}
                    >
                      {deleteIncidentMutation.isPending ? "Archiving incident" : "Archive incident"}
                    </Button>
                  </div>
                ) : null}
              </div>
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
