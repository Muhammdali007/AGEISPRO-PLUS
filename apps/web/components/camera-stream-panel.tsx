"use client";

import { ScanSearch, ShieldAlert, UserRound } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import { EmptyState } from "@/components/dashboard-ui";
import type { Camera, CameraStreamDescriptor } from "@/lib/api";
import { fetchProtectedMedia } from "@/lib/api";
import { cn } from "@/lib/cn";

export function CameraStreamPanel({
  accessToken,
  camera,
  stream,
  variant = "detail"
}: {
  accessToken: string;
  camera: Camera;
  stream: CameraStreamDescriptor;
  variant?: "detail" | "tile";
}) {
  const [protectedUrl, setProtectedUrl] = useState<string | null>(null);
  const [mediaError, setMediaError] = useState<string | null>(null);
  const protectedPath = stream.stream_url?.startsWith("/api/") ? stream.stream_url : null;

  useEffect(() => {
    if (!protectedPath) {
      return;
    }

    let objectUrl: string | null = null;
    let cancelled = false;

    fetchProtectedMedia(accessToken, protectedPath)
      .then((blob) => {
        if (cancelled) {
          return;
        }
        objectUrl = URL.createObjectURL(blob);
        setMediaError(null);
        setProtectedUrl(objectUrl);
      })
      .catch((error: Error) => {
        if (!cancelled) {
          setMediaError(error.message);
        }
      });

    return () => {
      cancelled = true;
      if (objectUrl) {
        URL.revokeObjectURL(objectUrl);
      }
    };
  }, [accessToken, protectedPath]);

  const playbackUrl = protectedPath ? protectedUrl : stream.stream_url;

  if (stream.stream_kind === "browser-camera") {
    return (
      <FeedFrame
        camera={camera}
        stream={stream}
        variant={variant}
        statusMessage="Browser camera preview is connected for live monitoring."
      >
        <BrowserCameraPreview deviceId={stream.browser_device_id} variant={variant} />
      </FeedFrame>
    );
  }

  if (!stream.browser_supported || !playbackUrl) {
    return (
      <FeedFrame
        camera={camera}
        stream={stream}
        variant={variant}
        statusMessage={mediaError ?? stream.health_message ?? "No browser-playable stream is available for this camera right now."}
      >
        <UnavailableStreamState camera={camera} stream={stream} variant={variant} />
      </FeedFrame>
    );
  }

  if (stream.stream_kind === "image") {
    return (
      <FeedFrame
        camera={camera}
        stream={stream}
        variant={variant}
        statusMessage="Image source ready. Detection overlays mark where people and weapon alerts can be visualized."
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={playbackUrl}
          alt={`${camera.name} preview`}
          className={cn(
            "h-full w-full rounded-[24px] object-cover",
            variant === "tile" ? "max-h-[320px]" : "max-h-[480px]"
          )}
        />
      </FeedFrame>
    );
  }

  return (
    <FeedFrame
      camera={camera}
      stream={stream}
      variant={variant}
      statusMessage="Live feed ready for operator monitoring and future inference overlays."
    >
      <video
        src={playbackUrl}
        controls
        autoPlay={stream.is_live}
        muted
        playsInline
        className={cn(
          "h-full w-full rounded-[24px] bg-black object-contain",
          variant === "tile" ? "max-h-[320px]" : "max-h-[480px]"
        )}
      />
    </FeedFrame>
  );
}

function BrowserCameraPreview({
  deviceId,
  variant
}: {
  deviceId: string | null;
  variant: "detail" | "tile";
}) {
  const [error, setError] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  useEffect(() => {
    let activeStream: MediaStream | null = null;
    let mounted = true;

    async function startPreview() {
      try {
        activeStream = await navigator.mediaDevices.getUserMedia({
          video: deviceId ? { deviceId: { exact: deviceId } } : true,
          audio: false
        });
        if (!mounted || !videoRef.current) {
          return;
        }
        videoRef.current.srcObject = activeStream;
        await videoRef.current.play();
      } catch (previewError) {
        if (mounted) {
          setError(
            previewError instanceof Error
              ? previewError.message
              : "Camera permission or local device access is unavailable."
          );
        }
      }
    }

    startPreview();

    return () => {
      mounted = false;
      activeStream?.getTracks().forEach((track) => track.stop());
    };
  }, [deviceId]);

  if (error) {
    return <EmptyState title="USB preview unavailable" description={error} />;
  }

  return (
    <video
      ref={videoRef}
      muted
      playsInline
      className={cn(
        "h-full w-full rounded-[24px] bg-black object-cover",
        variant === "tile" ? "max-h-[320px]" : "max-h-[480px]"
      )}
    />
  );
}

function FeedFrame({
  camera,
  stream,
  variant,
  statusMessage,
  children
}: {
  camera: Camera;
  stream: CameraStreamDescriptor;
  variant: "detail" | "tile";
  statusMessage: string;
  children: ReactNode;
}) {
  return (
    <div className="relative overflow-hidden rounded-[24px] border border-white/10 bg-black">
      <div className="absolute inset-x-0 top-0 z-20 flex items-center justify-between gap-3 border-b border-white/10 bg-slate-950/70 px-4 py-3 backdrop-blur-sm">
        <div>
          <p className="text-xs uppercase tracking-[0.22em] text-emerald-200/80">AI Monitoring Surface</p>
          <p className="mt-1 text-sm text-slate-300">{statusMessage}</p>
        </div>
        <div className="hidden rounded-full border border-emerald-400/20 bg-emerald-500/10 px-3 py-1 text-xs font-medium text-emerald-200 md:block">
          {camera.detection_enabled ? "Detection armed" : "Detection paused"}
        </div>
      </div>

      <div className={cn("relative pt-[76px]", variant === "tile" ? "min-h-[280px]" : "min-h-[420px]")}>{children}</div>

      <DetectionStatusBar
        detectionEnabled={camera.detection_enabled}
        fps={camera.inference_fps}
        streamKind={stream.stream_kind}
        variant={variant}
      />
    </div>
  );
}

function UnavailableStreamState({
  camera,
  stream,
  variant
}: {
  camera: Camera;
  stream: CameraStreamDescriptor;
  variant: "detail" | "tile";
}) {
  return (
    <div
      className={cn(
        "flex w-full flex-col justify-between overflow-hidden rounded-[24px] bg-[radial-gradient(circle_at_top,_rgba(14,165,233,0.14),_transparent_35%),linear-gradient(135deg,_rgba(2,6,23,0.98),_rgba(15,23,42,0.88)_45%,_rgba(3,7,18,0.98))]",
        variant === "tile" ? "h-[280px]" : "h-[420px]"
      )}
    >
      <div className="absolute inset-0 bg-[linear-gradient(rgba(148,163,184,0.08)_1px,transparent_1px),linear-gradient(90deg,rgba(148,163,184,0.08)_1px,transparent_1px)] bg-[size:34px_34px]" />
      <div className="relative z-10 mx-6 mt-8 rounded-[22px] border border-cyan-400/20 bg-slate-950/50 p-4 backdrop-blur-sm">
        <div className="flex items-start gap-3 text-cyan-100">
          <ScanSearch size={18} aria-hidden="true" />
          <div>
            <p className="text-sm font-medium">Preview unavailable</p>
            <p className="mt-1 text-sm text-slate-400">{stream.health_message || "This source does not currently expose a browser-playable feed."}</p>
          </div>
        </div>
      </div>

      <div className="relative z-10 mx-6 mt-6 rounded-[22px] border border-white/10 bg-black/25 p-4 text-sm text-slate-300">
        <p className="font-medium text-slate-100">What the backend reports</p>
        <div className="mt-3 grid gap-3 md:grid-cols-2">
          <StreamFact label="Camera" value={camera.name} />
          <StreamFact label="Source type" value={camera.source_type.toUpperCase()} />
          <StreamFact label="Stream mode" value={stream.stream_kind} />
          <StreamFact label="Browser support" value={stream.browser_supported ? "Available" : "Unavailable"} />
          <StreamFact label="Relay required" value={stream.requires_relay ? "Yes" : "No"} />
          <StreamFact label="Detection" value={camera.detection_enabled ? `${camera.inference_fps} FPS configured` : "Disabled"} />
        </div>
        {stream.notes.length > 0 ? (
          <div className="mt-4 rounded-[18px] border border-white/10 bg-black/20 px-4 py-3 text-sm text-slate-400">
            {stream.notes.join(" ")}
          </div>
        ) : null}
      </div>

      <div className={cn("relative z-10 mx-6 mt-auto grid gap-3 md:grid-cols-3", variant === "tile" ? "mb-4" : "mb-6")}>
        <MonitorStat label="Health" value={stream.health_status} tone="cyan" />
        <MonitorStat label="Playback" value={stream.stream_url ? "Configured" : "Unavailable"} tone="slate" />
        <MonitorStat label="Checked" value={stream.checked_at ? "Recently tested" : "Pending"} tone={camera.detection_enabled ? "emerald" : "slate"} />
      </div>
    </div>
  );
}

function DetectionStatusBar({
  detectionEnabled,
  fps,
  streamKind,
  variant
}: {
  detectionEnabled: boolean;
  fps: number;
  streamKind: string;
  variant: "detail" | "tile";
}) {
  return (
    <div
      className={cn(
        "grid gap-3 border-t border-white/10 bg-gradient-to-t from-slate-950/95 to-slate-950/35 px-4 md:grid-cols-3",
        variant === "tile" ? "py-3" : "py-4"
      )}
    >
      <AlertChip
        icon={<UserRound size={14} aria-hidden="true" />}
        label="Human tracking"
        value={detectionEnabled ? "armed" : "standby"}
        tone={detectionEnabled ? "emerald" : "slate"}
      />
      <AlertChip
        icon={<ShieldAlert size={14} aria-hidden="true" />}
        label="Weapon detection"
        value={detectionEnabled ? "watching" : "paused"}
        tone={detectionEnabled ? "red" : "slate"}
      />
      <AlertChip
        icon={<ScanSearch size={14} aria-hidden="true" />}
        label="Inference"
        value={detectionEnabled ? `${fps} fps | ${streamKind}` : streamKind}
        tone="cyan"
      />
    </div>
  );
}

function AlertChip({
  icon,
  label,
  value,
  tone
}: {
  icon: ReactNode;
  label: string;
  value: string;
  tone: "emerald" | "red" | "cyan" | "slate";
}) {
  return (
    <div className="rounded-[18px] border border-white/10 bg-black/35 px-3 py-3 backdrop-blur-sm">
      <div className="flex items-center gap-2 text-sm text-slate-200">
        <span
          className={cn(
            "inline-flex h-7 w-7 items-center justify-center rounded-full",
            tone === "emerald" && "bg-emerald-500/15 text-emerald-200",
            tone === "red" && "bg-red-500/15 text-red-200",
            tone === "cyan" && "bg-cyan-500/15 text-cyan-100",
            tone === "slate" && "bg-slate-500/20 text-slate-200"
          )}
        >
          {icon}
        </span>
        <span className="font-medium">{label}</span>
      </div>
      <p
        className={cn(
          "mt-2 text-sm",
          tone === "emerald" && "text-emerald-200",
          tone === "red" && "text-red-200",
          tone === "cyan" && "text-cyan-100",
          tone === "slate" && "text-slate-300"
        )}
      >
        {value}
      </p>
    </div>
  );
}

function StreamFact({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-[18px] border border-white/10 bg-black/20 px-4 py-3">
      <p className="text-xs uppercase tracking-[0.18em] text-slate-500">{label}</p>
      <p className="mt-2 text-sm text-slate-200">{value}</p>
    </div>
  );
}

function MonitorStat({
  label,
  value,
  tone
}: {
  label: string;
  value: string;
  tone: "emerald" | "cyan" | "slate";
}) {
  return (
    <div className="rounded-[20px] border border-white/10 bg-black/35 px-4 py-3 backdrop-blur-sm">
      <p className="text-xs uppercase tracking-[0.18em] text-slate-500">{label}</p>
      <p
        className={cn(
          "mt-2 text-sm font-medium",
          tone === "emerald" && "text-emerald-200",
          tone === "cyan" && "text-cyan-100",
          tone === "slate" && "text-slate-300"
        )}
      >
        {value}
      </p>
    </div>
  );
}
