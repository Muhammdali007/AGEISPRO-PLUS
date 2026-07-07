"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Activity, Radio, RefreshCw } from "lucide-react";
import { useParams } from "next/navigation";
import { Button } from "@/components/button";
import { CameraStreamPanel } from "@/components/camera-stream-panel";
import { EmptyState, InlineLink, SectionCard } from "@/components/dashboard-ui";
import { getCamera, getCameraStream, testCameraConnection } from "@/lib/api";
import { useAuthStore } from "@/lib/auth-store";
import { formatDateTime, labelize, statusTone } from "@/lib/format";
import { cn } from "@/lib/cn";

export default function CameraDetailPage() {
  const params = useParams<{ cameraId: string }>();
  const { accessToken } = useAuthStore();
  const queryClient = useQueryClient();

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

  const camera = cameraQuery.data;
  const stream = streamQuery.data;

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
              onClick={() => testConnection.mutate()}
              disabled={testConnection.isPending || !accessToken}
            >
              <RefreshCw size={16} className={cn(testConnection.isPending && "animate-spin")} />
              {testConnection.isPending ? "Testing..." : "Test connection"}
            </Button>
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
                  <CameraStreamPanel accessToken={accessToken!} camera={camera} stream={stream} />
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
                  value={camera.detection_enabled ? "Enabled" : "Disabled"}
                  tone={
                    camera.detection_enabled
                      ? "bg-emerald-500/15 text-emerald-200"
                      : "bg-slate-500/20 text-slate-200"
                  }
                  detail={`${camera.inference_fps} inference FPS configured`}
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
