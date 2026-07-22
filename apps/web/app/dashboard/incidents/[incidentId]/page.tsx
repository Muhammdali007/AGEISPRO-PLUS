"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import Image from "next/image";
import { useEffect, useMemo, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { Button } from "@/components/button";
import { EmptyState, InlineLink, SectionCard } from "@/components/dashboard-ui";
import {
  acknowledgeAlert,
  clearAlert,
  deleteIncident,
  fetchIncidentClip,
  fetchIncidentSnapshot,
  getIncident,
  listCameras,
  listIncidentAlerts,
  listUsers,
  saveIncidentAsPerson,
  updateIncident,
  type IncidentStatus
} from "@/lib/api";
import { useAuthStore } from "@/lib/auth-store";
import { detectionDisplayLabel, detectionTone, formatDateTime, formatPercent, labelize, statusTone } from "@/lib/format";
import { cn } from "@/lib/cn";

const incidentStatuses: IncidentStatus[] = [
  "open",
  "acknowledged",
  "investigating",
  "resolved",
  "dismissed"
];

export default function IncidentDetailPage() {
  const params = useParams<{ incidentId: string }>();
  const router = useRouter();
  const queryClient = useQueryClient();
  const { accessToken, user } = useAuthStore();
  const [fullNameDraft, setFullNameDraft] = useState("");
  const [personType, setPersonType] = useState<"employee" | "student" | "visitor" | "contractor" | "other">(
    "visitor"
  );
  const [referenceId, setReferenceId] = useState("");
  const [department, setDepartment] = useState("");
  const [title, setTitle] = useState("");
  const [saveError, setSaveError] = useState<string | null>(null);
  const [workflowError, setWorkflowError] = useState<string | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [workflowDraft, setWorkflowDraft] = useState<Partial<Record<"status" | "operator_notes" | "assigned_user_id", string>>>({});

  const incidentQuery = useQuery({
    queryKey: ["incident", params.incidentId, accessToken],
    queryFn: () => getIncident(accessToken!, params.incidentId),
    enabled: Boolean(accessToken && params.incidentId),
    retry: false
  });
  const camerasQuery = useQuery({
    queryKey: ["cameras", "list", accessToken],
    queryFn: () => listCameras(accessToken!),
    enabled: Boolean(accessToken)
  });
  const alertsQuery = useQuery({
    queryKey: ["incident-alerts", params.incidentId, accessToken],
    queryFn: () => listIncidentAlerts(accessToken!, params.incidentId),
    enabled: Boolean(accessToken && params.incidentId)
  });
  const usersQuery = useQuery({
    queryKey: ["users", "list", accessToken],
    queryFn: () => listUsers(accessToken!),
    enabled: Boolean(accessToken)
  });

  const incident = incidentQuery.data;
  const camera = (camerasQuery.data ?? []).find((item) => item.id === incident?.camera_id);
  const assignedUser = (usersQuery.data ?? []).find((item) => item.id === incident?.assigned_user_id);
  const relatedAlerts = alertsQuery.data ?? [];
  const recognizedIdentity = incident?.recognized_identity;
  const suggestedFullName = recognizedIdentity?.status === "known" ? (recognizedIdentity.identity_label ?? "") : "";
  const fullName = fullNameDraft || suggestedFullName;
  const canSaveDetectedPerson =
    Boolean(accessToken) &&
    (user?.role === "administrator" || user?.role === "supervisor" || user?.role === "operator") &&
    Boolean(incident) &&
    Boolean(recognizedIdentity) &&
    !recognizedIdentity?.identity_id &&
    (incident?.detection_type === "unknown_person" || incident?.detection_type === "person") &&
    Boolean(recognizedIdentity?.face_image_path || incident?.snapshot_path);
  const canDeleteIncident = user?.role === "administrator" || user?.role === "supervisor";

  const snapshotQuery = useQuery({
    queryKey: ["incident-snapshot", params.incidentId, accessToken, incident?.snapshot_path],
    queryFn: () => fetchIncidentSnapshot(accessToken!, params.incidentId),
    enabled: Boolean(accessToken && incident?.snapshot_path)
  });
  const clipQuery = useQuery({
    queryKey: ["incident-clip", params.incidentId, accessToken, incident?.clip_path],
    queryFn: () => fetchIncidentClip(accessToken!, params.incidentId),
    enabled: Boolean(accessToken && incident?.clip_path)
  });
  const snapshotUrl = useObjectUrl(snapshotQuery.data);
  const clipUrl = useObjectUrl(clipQuery.data);

  const savePersonMutation = useMutation({
    mutationFn: async () => {
      if (!accessToken || !incident) {
        throw new Error("You need to sign in again before saving a detected person.");
      }
      if (!fullName.trim()) {
        throw new Error("Full name is required.");
      }
      return saveIncidentAsPerson(accessToken, incident.id, {
        full_name: fullName.trim(),
        person_type: personType,
        reference_id: referenceId.trim() || null,
        department: department.trim() || null,
        title: title.trim() || null,
        is_active: true,
        is_primary: true
      });
    },
    onSuccess: async (person) => {
      setSaveError(null);
      await invalidateIncidentQueries(queryClient, accessToken, params.incidentId);
      setFullNameDraft(person.full_name);
      setReferenceId(person.reference_id);
    },
    onError: (cause) => {
      setSaveError(cause instanceof Error ? cause.message : "Unable to save detected person");
    }
  });

  const workflowMutation = useMutation({
    mutationFn: async () => {
      if (!accessToken || !incident) {
        throw new Error("You need to sign in again before updating the incident.");
      }
      return updateIncident(accessToken, incident.id, {
        status: (workflowDraft.status as IncidentStatus | undefined) ?? incident.status,
        operator_notes: workflowDraft.operator_notes ?? incident.operator_notes,
        assigned_user_id: workflowDraft.assigned_user_id === "" ? null : workflowDraft.assigned_user_id ?? incident.assigned_user_id
      });
    },
    onSuccess: async () => {
      setWorkflowError(null);
      setWorkflowDraft({});
      await invalidateIncidentQueries(queryClient, accessToken, params.incidentId);
    },
    onError: (cause) => {
      setWorkflowError(cause instanceof Error ? cause.message : "Unable to update incident workflow");
    }
  });

  const acknowledgeAlertMutation = useMutation({
    mutationFn: async (alertId: string) => {
      if (!accessToken) {
        throw new Error("You need to sign in again before acknowledging alerts.");
      }
      return acknowledgeAlert(accessToken, alertId);
    },
    onSuccess: async () => {
      await invalidateIncidentQueries(queryClient, accessToken, params.incidentId);
    }
  });

  const clearAlertMutation = useMutation({
    mutationFn: async (alertId: string) => {
      if (!accessToken) {
        throw new Error("You need to sign in again before clearing alerts.");
      }
      return clearAlert(accessToken, alertId);
    },
    onSuccess: async () => {
      await invalidateIncidentQueries(queryClient, accessToken, params.incidentId);
    }
  });

  const deleteIncidentMutation = useMutation({
    mutationFn: async () => {
      if (!accessToken || !incident) {
        throw new Error("You need to sign in again before archiving the incident.");
      }
      return deleteIncident(accessToken, incident.id);
    },
    onSuccess: async () => {
      setDeleteError(null);
      await queryClient.invalidateQueries({ queryKey: ["incidents", "list", accessToken] });
      await queryClient.invalidateQueries({ queryKey: ["alerts", "list", accessToken] });
      queryClient.removeQueries({ queryKey: ["incident", params.incidentId, accessToken] });
      router.push("/dashboard/incidents");
    },
    onError: (cause) => {
      setDeleteError(cause instanceof Error ? cause.message : "Unable to archive incident");
    }
  });

  const selectedStatus = (workflowDraft.status as IncidentStatus | undefined) ?? incident?.status ?? "open";
  const selectedAssignee = workflowDraft.assigned_user_id ?? incident?.assigned_user_id ?? "";
  const notesValue = workflowDraft.operator_notes ?? incident?.operator_notes ?? "";

  return (
    <div className="space-y-6">
      <SectionCard
        title={incident ? `${detectionDisplayLabel(incident.detection_type, incident.bounding_boxes)} incident` : "Incident details"}
        description="Phase 7 turns this route into the main operator workflow for evidence review, notes, assignment, and alert handling."
        action={
          <div className="flex flex-wrap items-center gap-3">
            {canDeleteIncident && incident ? (
              <Button
                type="button"
                variant="ghost"
                className="border-danger/50 text-red-200 hover:bg-danger/10"
                disabled={deleteIncidentMutation.isPending}
                onClick={() => {
                  setDeleteError(null);
                  void deleteIncidentMutation.mutateAsync();
                }}
              >
                {deleteIncidentMutation.isPending ? "Archiving incident" : "Archive incident"}
              </Button>
            ) : null}
            <InlineLink href="/dashboard/incidents" label="Back to incidents" />
          </div>
        }
      >
        {incidentQuery.error instanceof Error ? (
          <EmptyState title="Incident unavailable" description={incidentQuery.error.message} />
        ) : !incident ? (
          <EmptyState title="Loading incident" description="Fetching incident, camera, alert, and evidence context from the API." />
        ) : (
          <>
            {deleteError ? (
              <div className="mb-4 rounded-md border border-danger/50 bg-danger/10 px-3 py-2 text-sm text-red-200">
                {deleteError}
              </div>
            ) : null}
            <div className="grid gap-4 lg:grid-cols-2">
              <DetailTile label="Camera" value={camera?.name ?? incident.camera_id} tone="bg-black/20 text-slate-200" />
              <DetailTile label="Occurred" value={formatDateTime(incident.occurred_at)} tone="bg-black/20 text-slate-200" />
              <DetailTile label="Detection" value={detectionDisplayLabel(incident.detection_type, incident.bounding_boxes)} tone={cn("bg-black/20", detectionTone(incident.detection_type))} />
              <DetailTile label="Confidence" value={formatPercent(incident.confidence)} tone="bg-cyan-500/15 text-cyan-100" />
              <DetailTile label="Priority" value={labelize(incident.priority)} tone={statusTone(incident.priority)} />
              <DetailTile label="Status" value={labelize(incident.status)} tone={statusTone(incident.status)} />
              <DetailTile label="Assigned operator" value={assignedUser?.full_name ?? "Unassigned"} tone="bg-black/20 text-slate-200" />
              <DetailTile label="Alerts attached" value={`${relatedAlerts.length}`} tone="bg-black/20 text-slate-200" />
            </div>
          </>
        )}
      </SectionCard>

      {incident ? (
        <SectionCard
          title="Evidence"
          description="Snapshot and clip evidence captured for this incident are available here when the detection pipeline supplied them."
        >
          <div className="grid gap-4 xl:grid-cols-2">
            <EvidencePanel
              title="Snapshot evidence"
              path={incident.snapshot_path}
              mediaState={snapshotQuery.isPending ? "Loading snapshot..." : snapshotQuery.isError ? "Snapshot unavailable" : null}
            >
              {snapshotUrl ? (
                <Image
                  src={snapshotUrl}
                  alt="Incident snapshot evidence"
                  unoptimized
                  width={1280}
                  height={720}
                  className="max-h-80 w-full rounded-[20px] object-contain"
                />
              ) : (
                <p className="text-sm text-slate-400">No snapshot evidence was stored for this incident.</p>
              )}
            </EvidencePanel>

            <EvidencePanel
              title="Clip evidence"
              path={incident.clip_path}
              mediaState={clipQuery.isPending ? "Loading clip..." : clipQuery.isError ? "Clip unavailable" : null}
            >
              {clipUrl ? (
                <video src={clipUrl} controls className="max-h-80 w-full rounded-[20px]" />
              ) : (
                <p className="text-sm text-slate-400">No clip evidence was stored for this incident.</p>
              )}
            </EvidencePanel>
          </div>
        </SectionCard>
      ) : null}

      {incident && recognizedIdentity ? (
        <SectionCard
          title="Recognition"
          description="Known-person and unknown-person context generated during Phase 6 inference."
          action={
            recognizedIdentity.identity_id ? (
              <InlineLink href={`/dashboard/persons/${recognizedIdentity.identity_id}`} label="Open person profile" />
            ) : undefined
          }
        >
          <div className="grid gap-4 lg:grid-cols-2">
            <DetailTile
              label="Recognition status"
              value={labelize(recognizedIdentity.status)}
              tone={recognizedIdentity.status === "known" ? "bg-emerald-500/15 text-emerald-100" : "bg-fuchsia-500/15 text-fuchsia-100"}
            />
            <DetailTile
              label="Identity"
              value={
                recognizedIdentity.identity_label ??
                (recognizedIdentity.status === "known" ? "Matched profile unavailable" : "Not identified")
              }
              tone="bg-black/20 text-slate-200"
            />
            <DetailTile
              label="Match confidence"
              value={
                typeof recognizedIdentity.match_confidence === "number"
                  ? formatPercent(recognizedIdentity.match_confidence)
                  : "Not available"
              }
              tone="bg-cyan-500/15 text-cyan-100"
            />
            <DetailTile
              label="Face evidence"
              value={recognizedIdentity.face_image_path ?? "Not captured"}
              tone="bg-black/20 text-slate-200"
            />
          </div>
        </SectionCard>
      ) : null}

      {incident ? (
        <SectionCard
          title="Operator workflow"
          description="Update the incident lifecycle, assign an operator, and maintain notes for investigation and audit history."
        >
          <form
            className="grid gap-5 lg:grid-cols-2"
            onSubmit={async (event) => {
              event.preventDefault();
              setWorkflowError(null);
              await workflowMutation.mutateAsync();
            }}
          >
            <label className="block">
              <span className="mb-2 block text-sm text-slate-300">Status</span>
              <select
                className="h-11 w-full rounded-md border border-border bg-background px-3 text-sm outline-none transition focus:border-accent"
                value={selectedStatus}
                onChange={(event) => setWorkflowDraft((current) => ({ ...current, status: event.target.value }))}
              >
                {incidentStatuses.map((statusOption) => (
                  <option key={statusOption} value={statusOption}>
                    {labelize(statusOption)}
                  </option>
                ))}
              </select>
            </label>

            <label className="block">
              <span className="mb-2 block text-sm text-slate-300">Assigned operator</span>
              <select
                className="h-11 w-full rounded-md border border-border bg-background px-3 text-sm outline-none transition focus:border-accent"
                value={selectedAssignee}
                onChange={(event) =>
                  setWorkflowDraft((current) => ({
                    ...current,
                    assigned_user_id: event.target.value
                  }))
                }
              >
                <option value="">Unassigned</option>
                {(usersQuery.data ?? []).map((userOption) => (
                  <option key={userOption.id} value={userOption.id}>
                    {userOption.full_name}
                  </option>
                ))}
              </select>
            </label>

            <label className="block lg:col-span-2">
              <span className="mb-2 block text-sm text-slate-300">Operator notes</span>
              <textarea
                className="min-h-32 w-full rounded-md border border-border bg-background px-3 py-3 text-sm outline-none transition focus:border-accent"
                placeholder="Document the investigation, outcome, or follow-up actions."
                value={notesValue}
                onChange={(event) =>
                  setWorkflowDraft((current) => ({
                    ...current,
                    operator_notes: event.target.value
                  }))
                }
              />
            </label>

            {workflowError ? (
              <div className="rounded-md border border-danger/50 bg-danger/10 px-3 py-2 text-sm text-red-200 lg:col-span-2">
                {workflowError}
              </div>
            ) : null}

            <div className="flex flex-wrap items-center gap-3 lg:col-span-2">
              <Button type="submit" disabled={workflowMutation.isPending}>
                {workflowMutation.isPending ? "Saving workflow" : "Save workflow updates"}
              </Button>
            </div>
          </form>
        </SectionCard>
      ) : null}

      {incident ? (
        <SectionCard title="Related alerts" description="Acknowledge and clear alert records attached to this incident.">
          {relatedAlerts.length === 0 ? (
            <EmptyState title="No alerts attached" description="This incident has no linked alert records yet." />
          ) : (
            <div className="space-y-3">
              {relatedAlerts.map((alert) => (
                <div key={alert.id} className="rounded-[22px] border border-white/10 bg-black/15 p-4">
                  <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
                    <div>
                      <p className="font-medium">{alert.title}</p>
                      <p className="mt-1 text-sm text-slate-400">{alert.message}</p>
                      <p className="mt-3 text-xs uppercase tracking-[0.18em] text-slate-500">
                        Created {formatDateTime(alert.created_at)}
                      </p>
                    </div>
                    <div className="flex flex-wrap items-center gap-2">
                      <span className={cn("rounded-full px-2.5 py-1 text-xs font-medium", statusTone(alert.priority))}>
                        {labelize(alert.priority)}
                      </span>
                      <span className={cn("rounded-full px-2.5 py-1 text-xs font-medium", statusTone(alert.status))}>
                        {labelize(alert.status)}
                      </span>
                    </div>
                  </div>
                  <div className="mt-4 flex flex-wrap items-center gap-3">
                    <Button
                      type="button"
                      variant="ghost"
                      disabled={alert.status !== "active" || acknowledgeAlertMutation.isPending}
                      onClick={() => acknowledgeAlertMutation.mutate(alert.id)}
                    >
                      Acknowledge
                    </Button>
                    <Button
                      type="button"
                      variant="ghost"
                      disabled={alert.status === "cleared" || clearAlertMutation.isPending}
                      onClick={() => clearAlertMutation.mutate(alert.id)}
                    >
                      Clear alert
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </SectionCard>
      ) : null}

      {incident && recognizedIdentity && canSaveDetectedPerson ? (
        <SectionCard
          title="Save As Known Person"
          description="When the camera sees an unknown person, an operator or supervisor can save that captured face as a known person and generate an embedding automatically."
        >
          <form
            className="grid gap-5 lg:grid-cols-2"
            onSubmit={async (event) => {
              event.preventDefault();
              setSaveError(null);
              await savePersonMutation.mutateAsync();
            }}
          >
            <label className="block">
              <span className="mb-2 block text-sm text-slate-300">Full name</span>
              <input
                className="h-11 w-full rounded-md border border-border bg-background px-3 text-sm outline-none transition focus:border-accent"
                type="text"
                placeholder="Enter the person name"
                value={fullName}
                onChange={(event) => setFullNameDraft(event.target.value)}
              />
            </label>

            <label className="block">
              <span className="mb-2 block text-sm text-slate-300">Person type</span>
              <select
                className="h-11 w-full rounded-md border border-border bg-background px-3 text-sm outline-none transition focus:border-accent"
                value={personType}
                onChange={(event) => setPersonType(event.target.value as typeof personType)}
              >
                <option value="visitor">Visitor</option>
                <option value="student">Student</option>
                <option value="employee">Employee</option>
                <option value="contractor">Contractor</option>
                <option value="other">Other</option>
              </select>
            </label>

            <label className="block">
              <span className="mb-2 block text-sm text-slate-300">Reference ID</span>
              <input
                className="h-11 w-full rounded-md border border-border bg-background px-3 text-sm outline-none transition focus:border-accent"
                type="text"
                placeholder="Optional badge, employee, or visitor ID"
                value={referenceId}
                onChange={(event) => setReferenceId(event.target.value)}
              />
            </label>

            <label className="block">
              <span className="mb-2 block text-sm text-slate-300">Department</span>
              <input
                className="h-11 w-full rounded-md border border-border bg-background px-3 text-sm outline-none transition focus:border-accent"
                type="text"
                placeholder="Optional department"
                value={department}
                onChange={(event) => setDepartment(event.target.value)}
              />
            </label>

            <label className="block lg:col-span-2">
              <span className="mb-2 block text-sm text-slate-300">Title</span>
              <input
                className="h-11 w-full rounded-md border border-border bg-background px-3 text-sm outline-none transition focus:border-accent"
                type="text"
                placeholder="Optional title"
                value={title}
                onChange={(event) => setTitle(event.target.value)}
              />
            </label>

            <div className="rounded-[22px] border border-white/10 bg-black/15 p-4 text-sm text-slate-300 lg:col-span-2">
              Captured image source: {recognizedIdentity.face_image_path ?? incident.snapshot_path ?? "Not available"}
            </div>

            {saveError ? (
              <div className="rounded-md border border-danger/50 bg-danger/10 px-3 py-2 text-sm text-red-200 lg:col-span-2">
                {saveError}
              </div>
            ) : null}

            <div className="flex flex-wrap items-center gap-3 lg:col-span-2">
              <Button type="submit" disabled={savePersonMutation.isPending}>
                {savePersonMutation.isPending ? "Saving person" : "Save as known person"}
              </Button>
            </div>
          </form>
        </SectionCard>
      ) : null}
    </div>
  );
}

function DetailTile({ label, value, tone }: { label: string; value: string; tone: string }) {
  return (
    <div className="rounded-[22px] border border-white/10 bg-black/15 p-4">
      <p className="text-xs uppercase tracking-[0.18em] text-slate-500">{label}</p>
      <p className={cn("mt-3 inline-flex rounded-full px-2.5 py-1 text-sm font-medium", tone)}>{value}</p>
    </div>
  );
}

function EvidencePanel({
  title,
  path,
  mediaState,
  children
}: {
  title: string;
  path: string | null;
  mediaState: string | null;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-[22px] border border-white/10 bg-black/15 p-4">
      <p className="text-sm font-medium">{title}</p>
      <p className="mt-2 break-all text-xs uppercase tracking-[0.18em] text-slate-500">
        {path ?? "No stored path"}
      </p>
      {mediaState ? <p className="mt-4 text-sm text-slate-400">{mediaState}</p> : null}
      <div className="mt-4">{children}</div>
    </div>
  );
}

function useObjectUrl(blob: Blob | undefined) {
  const url = useMemo(() => (blob ? URL.createObjectURL(blob) : null), [blob]);

  useEffect(
    () => () => {
      if (url) {
        URL.revokeObjectURL(url);
      }
    },
    [url]
  );

  return url;
}

async function invalidateIncidentQueries(
  queryClient: ReturnType<typeof useQueryClient>,
  accessToken: string | null,
  incidentId: string
) {
  await queryClient.invalidateQueries({ queryKey: ["incident", incidentId, accessToken] });
  await queryClient.invalidateQueries({ queryKey: ["incidents", "list", accessToken] });
  await queryClient.invalidateQueries({ queryKey: ["incident-alerts", incidentId, accessToken] });
  await queryClient.invalidateQueries({ queryKey: ["alerts", "list", accessToken] });
  await queryClient.invalidateQueries({ queryKey: ["persons", "list", accessToken] });
}
