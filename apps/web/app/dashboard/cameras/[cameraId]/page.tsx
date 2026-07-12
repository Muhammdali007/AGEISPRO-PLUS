"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Activity, Pencil, Play, Radar, Radio, RefreshCw, Square, Trash2 } from "lucide-react";
import { useParams, useRouter } from "next/navigation";
import { useEffect, useRef } from "react";
import { Button } from "@/components/button";
import { CameraStreamPanel, type CameraStreamPanelHandle } from "@/components/camera-stream-panel";
import { EmptyState, InlineLink, SectionCard } from "@/components/dashboard-ui";
import { deleteCamera, getCamera, getCameraStream, runCameraDetectionScan, testCameraConnection, updateCamera } from "@/lib/api";
import { useAuthStore } from "@/lib/auth-store";
import { formatDateTime, labelize, statusTone } from "@/lib/format";
import { cn } from "@/lib/cn";

export default function CameraDetailPage() {
  const params = useParams<{ cameraId: string }>();
  const router = useRouter();
  const { accessToken, user, logout } = useAuthStore();
  const queryClient = useQueryClient();
  const canManageCamera = user?.role === "administrator" || user?.role === "supervisor";
  const streamPanelRef = useRef<CameraStreamPanelHandle | null>(null);

  const cameraQuery = useQuery({
    queryKey: ["camera", params.cameraId, accessToken],
    queryFn: async () => getCamera(accessToken!, params.cameraId),
    enabled: Boolean(accessToken && params.cameraId),
    retry: false
  });
  const streamQuery = useQuery({
    queryKey: ["camera-stream", params.cameraId, accessToken],
    queryFn: async () => getCameraStream(accessToken!, params.cameraId),
    enabled: Boolean(accessToken && params.cameraId),
    retry: false
  });
  const testConnection = useMutation({
    mutationFn: async () => testCameraConnection(accessToken!, params.cameraId),
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["camera", params.cameraId, accessToken] }),
        queryClient.invalidateQueries({ queryKey: ["camera-stream", params.cameraId, accessToken] }),
        queryClient.invalidateQueries({ queryKey: ["cameras", "list", accessToken] })
      ]);
    }
  });
  const updateCameraState = useMutation({
    mutationFn: async (nextRunning: boolean) => {
      if (!accessToken) {
        throw new Error("You need to sign in again before changing this camera state.");
      }

      return updateCamera(accessToken, params.cameraId, {
        detection_enabled: nextRunning,
        status: nextRunning ? "unknown" : "disabled"
      });
    },
    onSuccess: async (_, nextRunning) => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["camera", params.cameraId, accessToken] }),
        queryClient.invalidateQueries({ queryKey: ["camera-stream", params.cameraId, accessToken] }),
        queryClient.invalidateQueries({ queryKey: ["cameras", "list", accessToken] })
      ]);

      if (nextRunning) {
        testConnection.mutate();
      }
    },
    onError: (cause) => {
      if (cause instanceof Error && (cause.message === "Invalid credentials" || cause.message === "Session expired")) {
        logout();
        router.push("/login");
      }
    }
  });
  const deleteCameraMutation = useMutation({
    mutationFn: async () => {
      if (!accessToken) {
        throw new Error("You need to sign in again before deleting this camera.");
      }

      return deleteCamera(accessToken, params.cameraId);
    },
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["cameras", "list", accessToken] });
      router.push("/dashboard/cameras");
    },
    onError: (cause) => {
      if (cause instanceof Error && (cause.message === "Invalid credentials" || cause.message === "Session expired")) {
        logout();
        router.push("/login");
      }
    }
  });
  const scanMutation = useMutation({
    mutationFn: async (occurrenceHint?: string) => {
      if (!accessToken) {
        throw new Error("You need to sign in again before running an AI scan.");
      }

      const snapshot = await streamPanelRef.current?.captureFrame();
      return runCameraDetectionScan(accessToken, params.cameraId, {
        frame_content_base64: snapshot?.contentBase64,
        frame_content_type: snapshot?.contentType,
        occurrence_hint: occurrenceHint ?? "dashboard_manual_scan"
      });
    },
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["incidents", accessToken] }),
        queryClient.invalidateQueries({ queryKey: ["alerts", accessToken] }),
        queryClient.invalidateQueries({ queryKey: ["camera", params.cameraId, accessToken] }),
        queryClient.invalidateQueries({ queryKey: ["cameras", "list", accessToken] })
      ]);
    },
    onError: (cause) => {
      if (cause instanceof Error && (cause.message === "Invalid credentials" || cause.message === "Session expired")) {
        logout();
        router.push("/login");
      }
    }
  });

  const camera = cameraQuery.data;
  const stream = streamQuery.data;
  const isRunning = Boolean(camera?.detection_enabled && camera.status !== "disabled");
  const runScan = scanMutation.mutate;
  const scanPending = scanMutation.isPending;

  useEffect(() => {
    if (!accessToken || !stream || !isRunning) {
      return;
    }

    const scanLiveFrame = () => {
      if (document.visibilityState === "visible" && !scanPending) {
        runScan("dashboard_live_scan");
      }
    };
    const initialScan = window.setTimeout(scanLiveFrame, 750);
    const interval = window.setInterval(scanLiveFrame, 2000);

    return () => {
      window.clearTimeout(initialScan);
      window.clearInterval(interval);
    };
  }, [accessToken, isRunning, runScan, scanPending, stream]);

  async function handleDeleteCamera() {
    if (!window.confirm("Delete this camera? This action cannot be undone.")) {
      return;
    }

    try {
      await deleteCameraMutation.mutateAsync();
    } catch {
      // The mutation error is rendered elsewhere if needed.
    }
  }

  return (
    <div className="space-y-6">
      <SectionCard
        title={camera?.name ?? "Camera details"}
        description="Phase 4 adds source-aware preview, health testing, and stream guidance without coupling browser playback to AI inference services."
        action={
          <div className="flex flex-wrap items-center gap-3">
            <Button
              type="button"
              variant="ghost"
              onClick={() => scanMutation.mutate(undefined)}
              disabled={scanMutation.isPending || !accessToken || !camera}
            >
              <Radar size={16} aria-hidden="true" />
              {scanMutation.isPending ? "Scanning..." : "Run AI scan"}
            </Button>
            <Button
              type="button"
              variant="ghost"
              onClick={() => testConnection.mutate()}
              disabled={testConnection.isPending || !accessToken}
            >
              <RefreshCw size={16} className={cn(testConnection.isPending && "animate-spin")} />
              {testConnection.isPending ? "Testing..." : "Test connection"}
            </Button>
            {canManageCamera ? (
              <>
                <Button
                  type="button"
                  variant="ghost"
                  onClick={() => updateCameraState.mutate(!isRunning)}
                  disabled={updateCameraState.isPending || !camera}
                >
                  {isRunning ? <Square size={16} aria-hidden="true" /> : <Play size={16} aria-hidden="true" />}
                  {updateCameraState.isPending ? (isRunning ? "Turning off..." : "Starting...") : isRunning ? "Turn off camera" : "Start camera"}
                </Button>
                <Button
                  type="button"
                  variant="ghost"
                  onClick={() => router.push(`/dashboard/cameras/${params.cameraId}/edit`)}
                  disabled={!camera}
                >
                  <Pencil size={16} aria-hidden="true" />
                  Edit
                </Button>
                <Button
                  type="button"
                  variant="ghost"
                  className="border-danger/50 text-red-200 hover:bg-danger/10"
                  onClick={handleDeleteCamera}
                  disabled={deleteCameraMutation.isPending || !camera}
                >
                  <Trash2 size={16} aria-hidden="true" />
                  {deleteCameraMutation.isPending ? "Deleting..." : "Delete"}
                </Button>
              </>
            ) : null}
            <InlineLink href="/dashboard/cameras" label="Back to cameras" />
          </div>
        }
      >
        {cameraQuery.error instanceof Error ? (
          <EmptyState title="Camera unavailable" description={cameraQuery.error.message} />
        ) : !camera ? (
          <EmptyState
            title="Loading camera"
            description="Fetching the camera configuration and health metadata from the API."
          />
        ) : (
          <div className="space-y-6">
            <div className="grid gap-6 xl:grid-cols-[1.5fr_1fr]">
              <div className="rounded-[26px] border border-white/10 bg-black/20 p-4">
                <div className="mb-4 flex items-center justify-between gap-3">
                  <div>
                    <p className="text-sm font-medium text-slate-200">Live preview</p>
                    <p className="mt-1 text-sm text-slate-400">
                      {stream?.health_message ?? "Waiting for stream configuration details from the API."}
                    </p>
                  </div>
                  <div className="flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-xs uppercase tracking-[0.18em] text-slate-300">
                    <Radio size={14} className="text-accent" />
                    {stream?.is_live ? "Live" : "Recorded"}
                  </div>
                </div>

                {streamQuery.error instanceof Error ? (
                  <EmptyState title="Stream metadata unavailable" description={streamQuery.error.message} />
                ) : stream ? (
                  <CameraStreamPanel
                    ref={streamPanelRef}
                    accessToken={accessToken!}
                    camera={camera}
                    stream={stream}
                    detections={scanMutation.data?.detections ?? []}
                  />
                ) : (
                  <EmptyState
                    title="Preparing stream profile"
                    description="Resolving how this camera source should be previewed in the browser."
                  />
                )}
              </div>

              <div className="space-y-4">
                <HealthCard
                  title="Health"
                  value={labelize(stream?.health_status ?? camera.status)}
                  tone={statusTone(stream?.health_status ?? camera.status)}
                  detail={formatDateTime(stream?.checked_at ?? camera.health_checked_at)}
                />
                <HealthCard
                  title="Detection"
                  value={isRunning ? "Running" : "Stopped"}
                  tone={
                    isRunning
                      ? "bg-emerald-500/15 text-emerald-200"
                      : "bg-slate-500/20 text-slate-200"
                  }
                  detail={isRunning ? `${camera.inference_fps} inference FPS configured` : "Camera is paused for detection and monitoring workflows"}
                />
                <HealthCard
                  title="Playback mode"
                  value={stream ? labelize(stream.stream_kind) : "Loading"}
                  tone="bg-cyan-500/15 text-cyan-100"
                  detail={stream?.requires_relay ? "Relay required for browser playback" : "Direct playback supported"}
                />
              </div>
            </div>

            {testConnection.data ? (
              <div className="rounded-[22px] border border-white/10 bg-black/15 p-4 text-sm text-slate-300">
                <div className="flex items-center gap-2">
                  <Activity size={16} className="text-accent" />
                  Most recent test: {testConnection.data.message}
                </div>
                <p className="mt-2 text-slate-400">
                  Completed {formatDateTime(testConnection.data.checked_at)}
                  {typeof testConnection.data.latency_ms === "number"
                    ? ` | ${testConnection.data.latency_ms} ms`
                    : ""}
                </p>
              </div>
            ) : null}

            {testConnection.error instanceof Error ? (
              <EmptyState title="Connection test failed" description={testConnection.error.message} />
            ) : null}

            {scanMutation.data ? (
              <div className="rounded-[22px] border border-emerald-400/20 bg-emerald-500/10 p-4 text-sm text-emerald-100">
                <div className="flex items-center gap-2">
                  <Radar size={16} aria-hidden="true" />
                  Scan complete: {scanMutation.data.detection_count} detections, {scanMutation.data.incident_count} incidents, {scanMutation.data.alert_count} alerts.
                </div>
                <p className="mt-2 text-emerald-200/80">
                  Backend: {scanMutation.data.backend ?? "unknown"} | Model: {scanMutation.data.model_name}
                </p>
                {scanMutation.data.backend === "simulated" ? (
                  <p className="mt-2 text-amber-200/90">
                    The AI service is running in simulated mode, so it will not perform real face or knife detection from the camera feed.
                  </p>
                ) : null}
                <p className="mt-2 text-emerald-200/80">
                  {scanMutation.data.detections.length > 0
                    ? scanMutation.data.detections
                        .map((detection) =>
                          `${labelize(detection.detection_type)} ${Math.round(detection.confidence * 100)}%${detection.identity_label ? ` (${detection.identity_label})` : ""}`
                        )
                        .join(" | ")
                    : scanMutation.data.ignored_reasons.join(" | ") || "No detections were produced for this frame."}
                </p>
              </div>
            ) : null}

            {scanMutation.error instanceof Error ? (
              <EmptyState title="AI scan failed" description={scanMutation.error.message} />
            ) : null}

            {updateCameraState.error instanceof Error ? (
              <EmptyState title="Unable to update camera state" description={updateCameraState.error.message} />
            ) : null}

            {deleteCameraMutation.error instanceof Error ? (
              <EmptyState title="Unable to delete camera" description={deleteCameraMutation.error.message} />
            ) : null}

            <div className="grid gap-4 lg:grid-cols-2">
              <DetailBlock label="Status" value={labelize(camera.status)} tone={statusTone(camera.status)} />
              <DetailBlock
                label="Source type"
                value={labelize(camera.source_type)}
                tone="bg-cyan-500/15 text-cyan-100"
              />
              <DetailBlock label="Source" value={camera.source} tone="bg-black/20 text-slate-200" />
              <DetailBlock label="Location" value={camera.location ?? "Not set"} tone="bg-black/20 text-slate-200" />
              <DetailBlock label="Group" value={camera.group ?? "Ungrouped"} tone="bg-black/20 text-slate-200" />
              <DetailBlock label="Inference FPS" value={`${camera.inference_fps}`} tone="bg-emerald-500/15 text-emerald-200" />
              <DetailBlock label="Health checked" value={formatDateTime(camera.health_checked_at)} tone="bg-black/20 text-slate-200" />
              <DetailBlock label="Last seen" value={formatDateTime(camera.last_seen_at)} tone="bg-black/20 text-slate-200" />
              <DetailBlock label="Tags" value={camera.tags.length ? camera.tags.join(", ") : "No tags"} tone="bg-black/20 text-slate-200" />
              <DetailBlock
                label="Stream notes"
                value={stream?.notes.join(" | ") || "No stream notes yet"}
                tone="bg-black/20 text-slate-200"
              />
            </div>
          </div>
        )}
      </SectionCard>
    </div>
  );
}

function HealthCard({
  title,
  value,
  detail,
  tone
}: {
  title: string;
  value: string;
  detail: string;
  tone: string;
}) {
  return (
    <div className="rounded-[22px] border border-white/10 bg-black/15 p-4">
      <p className="text-xs uppercase tracking-[0.18em] text-slate-500">{title}</p>
      <p className={cn("mt-3 inline-flex rounded-full px-2.5 py-1 text-sm font-medium", tone)}>{value}</p>
      <p className="mt-3 text-sm text-slate-400">{detail}</p>
    </div>
  );
}

function DetailBlock({ label, value, tone }: { label: string; value: string; tone: string }) {
  return (
    <div className="rounded-[22px] border border-white/10 bg-black/15 p-4">
      <p className="text-xs uppercase tracking-[0.18em] text-slate-500">{label}</p>
      <p className={cn("mt-3 inline-flex max-w-full rounded-full px-2.5 py-1 text-sm font-medium", tone)}>{value}</p>
    </div>
  );
}
