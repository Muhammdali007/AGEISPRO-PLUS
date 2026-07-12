"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { CameraIcon, Pencil, Play, Power, Trash2 } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { Button } from "@/components/button";
import { EmptyState, InlineLink, MetricCard, SectionCard } from "@/components/dashboard-ui";
import { deleteCamera, listCameras, updateCamera } from "@/lib/api";
import { useAuthStore } from "@/lib/auth-store";
import { formatDateTime, labelize, statusTone } from "@/lib/format";
import { cn } from "@/lib/cn";

export default function CamerasPage() {
  const router = useRouter();
  const queryClient = useQueryClient();
  const { accessToken, user, logout } = useAuthStore();
  const [pendingCameraId, setPendingCameraId] = useState<string | null>(null);
  const canManageCamera = user?.role === "administrator" || user?.role === "supervisor";
  const camerasQuery = useQuery({
    queryKey: ["cameras", "list", accessToken],
    queryFn: () => listCameras(accessToken!),
    enabled: Boolean(accessToken),
    refetchInterval: 30_000
  });
  const cameraStateMutation = useMutation({
    mutationFn: async ({ cameraId, nextRunning }: { cameraId: string; nextRunning: boolean }) => {
      if (!accessToken) {
        throw new Error("You need to sign in again before changing this camera state.");
      }

      setPendingCameraId(cameraId);
      return updateCamera(accessToken, cameraId, {
        detection_enabled: nextRunning,
        status: nextRunning ? "unknown" : "disabled"
      });
    },
    onSuccess: async (_, variables) => {
      await queryClient.invalidateQueries({ queryKey: ["cameras", "list", accessToken] });
      router.refresh();
      setPendingCameraId(null);

      if (variables.nextRunning) {
        router.push(`/dashboard/cameras/${variables.cameraId}`);
      }
    },
    onError: (cause) => {
      setPendingCameraId(null);
      if (cause instanceof Error && (cause.message === "Invalid credentials" || cause.message === "Session expired")) {
        logout();
        router.push("/login");
      }
    }
  });
  const deleteCameraMutation = useMutation({
    mutationFn: async (cameraId: string) => {
      if (!accessToken) {
        throw new Error("You need to sign in again before deleting this camera.");
      }

      setPendingCameraId(cameraId);
      return deleteCamera(accessToken, cameraId);
    },
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["cameras", "list", accessToken] });
      setPendingCameraId(null);
    },
    onError: (cause) => {
      setPendingCameraId(null);
      if (cause instanceof Error && (cause.message === "Invalid credentials" || cause.message === "Session expired")) {
        logout();
        router.push("/login");
      }
    }
  });

  const cameras = camerasQuery.data ?? [];
  const online = cameras.filter((camera) => camera.status === "online").length;
  const degraded = cameras.filter((camera) => camera.status === "degraded").length;
  const enabled = cameras.filter((camera) => camera.detection_enabled).length;

  async function handleDeleteCamera(cameraId: string) {
    if (!window.confirm("Delete this camera? This action cannot be undone.")) {
      return;
    }

    try {
      await deleteCameraMutation.mutateAsync(cameraId);
    } catch {
      // Mutation state already handles the user-facing error path.
    }
  }

  return (
    <div className="space-y-6">
      <section className="grid gap-4 md:grid-cols-3">
        <MetricCard label="Total cameras" value={`${cameras.length}`} detail="RTSP, HTTP, USB, and file inputs share one management surface." />
        <MetricCard label="Online" value={`${online}`} detail="Streams reporting healthy operational status." tone="success" />
        <MetricCard label="Degraded or disabled" value={`${degraded + (cameras.length - online - degraded)}`} detail={`${enabled} cameras currently have detection enabled.`} tone="alert" />
      </section>

      <SectionCard
        title="Camera registry"
        description="Phase 4 turns the registry into a live operations view with preview-ready source metadata and refreshed health states."
        action={
          <div className="flex flex-wrap items-center gap-3">
            <InlineLink href="/dashboard/cameras/create" label="Add camera" />
            <InlineLink href="/dashboard" label="Back to overview" />
          </div>
        }
      >
        {cameras.length === 0 ? (
          <EmptyState
            title="No cameras configured"
            description="No camera sources are registered yet. Use the add flow to register RTSP, HTTP, USB, or file-backed inputs."
          />
        ) : (
          <div className="space-y-3">
            {cameras.map((camera) => (
              <article
                key={camera.id}
                className="grid gap-4 rounded-[24px] border border-white/10 bg-black/15 p-4 transition hover:border-accent/30 hover:bg-black/25 lg:grid-cols-[1.2fr_0.7fr_0.7fr_0.8fr]"
              >
                <Link href={`/dashboard/cameras/${camera.id}`} className="flex items-start gap-3 rounded-[20px] outline-none focus:ring-2 focus:ring-accent">
                  <div className="mt-1 flex h-10 w-10 items-center justify-center rounded-2xl bg-white/5 text-accent">
                    <CameraIcon size={18} aria-hidden="true" />
                  </div>
                  <div>
                    <p className="font-medium">{camera.name}</p>
                    <p className="mt-1 text-sm text-slate-400">{camera.location ?? camera.source}</p>
                    <p className="mt-2 text-xs uppercase tracking-[0.18em] text-slate-500">
                      {labelize(camera.source_type)}
                    </p>
                  </div>
                </Link>
                <MetaColumn label="Status" value={labelize(camera.status)} tone={statusTone(camera.status)} />
                <MetaColumn
                  label="Inference"
                  value={`${camera.inference_fps} FPS`}
                  tone={camera.detection_enabled ? "bg-emerald-500/15 text-emerald-200" : "bg-slate-500/20 text-slate-300"}
                />
                <div className="rounded-[20px] border border-white/10 bg-black/20 p-3">
                  <p className="text-xs uppercase tracking-[0.18em] text-slate-500">Last seen</p>
                  <p className={cn("mt-3 inline-flex rounded-full px-2.5 py-1 text-sm font-medium", "bg-black/20 text-slate-300")}>
                    {formatDateTime(camera.last_seen_at)}
                  </p>
                  <div className="mt-4 flex flex-wrap gap-2">
                    <Button type="button" variant="ghost" onClick={() => router.push(`/dashboard/cameras/${camera.id}`)}>
                      Open
                    </Button>
                    {canManageCamera ? (
                      <>
                        <Button type="button" variant="ghost" onClick={() => router.push(`/dashboard/cameras/${camera.id}/edit`)}>
                          <Pencil size={15} aria-hidden="true" />
                          Edit
                        </Button>
                        {camera.detection_enabled && camera.status !== "disabled" ? (
                          <Button
                            type="button"
                            variant="ghost"
                            className="border-amber-500/40 text-amber-100 hover:bg-amber-500/10"
                            disabled={cameraStateMutation.isPending || pendingCameraId === camera.id}
                            onClick={() => cameraStateMutation.mutate({ cameraId: camera.id, nextRunning: false })}
                          >
                            <Power size={15} aria-hidden="true" />
                            {pendingCameraId === camera.id && cameraStateMutation.isPending ? "Turning off..." : "Turn off camera"}
                          </Button>
                        ) : (
                          <Button
                            type="button"
                            variant="ghost"
                            className="border-emerald-500/40 text-emerald-100 hover:bg-emerald-500/10"
                            disabled={cameraStateMutation.isPending || pendingCameraId === camera.id}
                            onClick={() => cameraStateMutation.mutate({ cameraId: camera.id, nextRunning: true })}
                          >
                            <Play size={15} aria-hidden="true" />
                            {pendingCameraId === camera.id && cameraStateMutation.isPending ? "Starting..." : "Start camera"}
                          </Button>
                        )}
                        <Button
                          type="button"
                          variant="ghost"
                          className="border-danger/50 text-red-200 hover:bg-danger/10"
                          disabled={deleteCameraMutation.isPending || pendingCameraId === camera.id}
                          onClick={() => handleDeleteCamera(camera.id)}
                        >
                          <Trash2 size={15} aria-hidden="true" />
                          {pendingCameraId === camera.id && deleteCameraMutation.isPending ? "Deleting..." : "Delete"}
                        </Button>
                      </>
                    ) : null}
                  </div>
                </div>
              </article>
            ))}
          </div>
        )}

        {cameraStateMutation.error instanceof Error ? (
          <div className="rounded-md border border-danger/50 bg-danger/10 px-3 py-2 text-sm text-red-200">
            {cameraStateMutation.error.message}
          </div>
        ) : null}

        {deleteCameraMutation.error instanceof Error ? (
          <div className="rounded-md border border-danger/50 bg-danger/10 px-3 py-2 text-sm text-red-200">
            {deleteCameraMutation.error.message}
          </div>
        ) : null}
      </SectionCard>
    </div>
  );
}

function MetaColumn({ label, value, tone }: { label: string; value: string; tone: string }) {
  return (
    <div className="rounded-[20px] border border-white/10 bg-black/20 p-3">
      <p className="text-xs uppercase tracking-[0.18em] text-slate-500">{label}</p>
      <p className={cn("mt-3 inline-flex rounded-full px-2.5 py-1 text-sm font-medium", tone)}>{value}</p>
    </div>
  );
}
