"use client";

import { useQuery } from "@tanstack/react-query";
import { CameraIcon } from "lucide-react";
import Link from "next/link";
import { EmptyState, InlineLink, MetricCard, SectionCard } from "@/components/dashboard-ui";
import { listCameras } from "@/lib/api";
import { useAuthStore } from "@/lib/auth-store";
import { formatDateTime, labelize, statusTone } from "@/lib/format";
import { cn } from "@/lib/cn";

export default function CamerasPage() {
  const { accessToken } = useAuthStore();
  const camerasQuery = useQuery({
    queryKey: ["cameras", "list", accessToken],
    queryFn: () => listCameras(accessToken!),
    enabled: Boolean(accessToken),
    refetchInterval: 30_000
  });

  const cameras = camerasQuery.data ?? [];
  const online = cameras.filter((camera) => camera.status === "online").length;
  const degraded = cameras.filter((camera) => camera.status === "degraded").length;
  const enabled = cameras.filter((camera) => camera.detection_enabled).length;

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
            <InlineLink href="/dashboard/cameras/create" label="Register camera" />
            <InlineLink href="/dashboard" label="Back to overview" />
          </div>
        }
      >
        {cameras.length === 0 ? (
          <EmptyState
            title="No cameras configured"
            description="No camera sources are registered yet. Use the register flow to add RTSP, HTTP, USB, or file-backed inputs to the registry."
          />
        ) : (
          <div className="space-y-3">
            {cameras.map((camera) => (
              <Link
                key={camera.id}
                href={`/dashboard/cameras/${camera.id}`}
                className="grid gap-4 rounded-[24px] border border-white/10 bg-black/15 p-4 transition hover:border-accent/30 hover:bg-black/25 lg:grid-cols-[1.2fr_0.7fr_0.7fr_0.8fr]"
              >
                <div className="flex items-start gap-3">
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
                </div>
                <MetaColumn label="Status" value={labelize(camera.status)} tone={statusTone(camera.status)} />
                <MetaColumn
                  label="Inference"
                  value={`${camera.inference_fps} FPS`}
                  tone={camera.detection_enabled ? "bg-emerald-500/15 text-emerald-200" : "bg-slate-500/20 text-slate-300"}
                />
                <MetaColumn label="Last seen" value={formatDateTime(camera.last_seen_at)} tone="bg-black/20 text-slate-300" />
              </Link>
            ))}
          </div>
        )}
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
