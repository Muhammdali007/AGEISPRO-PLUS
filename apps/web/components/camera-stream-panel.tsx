"use client";

import { ScanSearch, ShieldAlert, UserRound } from "lucide-react";
import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from "react";
import type { ReactNode, RefCallback } from "react";
import { EmptyState } from "@/components/dashboard-ui";
import type { Camera, CameraDetectionScanSummary, CameraStreamDescriptor, DetectionOverlayBox } from "@/lib/api";
import { fetchProtectedMedia } from "@/lib/api";
import { cn } from "@/lib/cn";

export type CameraStreamPanelHandle = {
  captureFrame: () => Promise<{ contentBase64: string; contentType: string } | null>;
};

export const CameraStreamPanel = forwardRef<CameraStreamPanelHandle, {
  accessToken: string;
  camera: Camera;
  stream: CameraStreamDescriptor;
  detections?: CameraDetectionScanSummary[];
  variant?: "detail" | "tile";
}>(function CameraStreamPanel({
  accessToken,
  camera,
  stream,
  detections = [],
  variant = "detail"
}, ref) {
  const [protectedMedia, setProtectedMedia] = useState<{
    sourcePath: string | null;
    url: string | null;
    error: string | null;
  }>({
    sourcePath: null,
    url: null,
    error: null
  });
  const cameraTurnedOff = camera.status === "disabled";
  const protectedPath = stream.stream_url?.startsWith("/api/") ? stream.stream_url : null;
  const imageRef = useRef<HTMLImageElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const surfaceRef = useRef<HTMLDivElement | null>(null);
  const [activeMediaElement, setActiveMediaElement] = useState<HTMLImageElement | HTMLVideoElement | null>(null);
  const [surfaceVersion, setSurfaceVersion] = useState(0);
  const setImageElement = useCallback<RefCallback<HTMLImageElement>>((element) => {
    imageRef.current = element;
    setActiveMediaElement(element);
  }, []);
  const setVideoElement = useCallback<RefCallback<HTMLVideoElement>>((element) => {
    videoRef.current = element;
    setActiveMediaElement(element);
  }, []);

  async function captureFrame() {
    if (cameraTurnedOff) {
      return null;
    }

    try {
      if (videoRef.current && videoRef.current.videoWidth > 0 && videoRef.current.videoHeight > 0) {
        return drawToBase64(videoRef.current, videoRef.current.videoWidth, videoRef.current.videoHeight);
      }

      if (imageRef.current && imageRef.current.naturalWidth > 0 && imageRef.current.naturalHeight > 0) {
        return drawToBase64(imageRef.current, imageRef.current.naturalWidth, imageRef.current.naturalHeight);
      }
    } catch {
      // Cross-origin camera feeds can be displayed but not read through canvas.
      // Returning null lets the API fetch a server-side snapshot instead.
      return null;
    }

    return null;
  }

  useImperativeHandle(ref, () => ({
    captureFrame
  }));

  useEffect(() => {
    if (!protectedPath || cameraTurnedOff) {
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
        setProtectedMedia({
          sourcePath: protectedPath,
          url: objectUrl,
          error: null
        });
      })
      .catch((error: Error) => {
        if (!cancelled) {
          setProtectedMedia({
            sourcePath: protectedPath,
            url: null,
            error: error.message
          });
        }
      });

    return () => {
      cancelled = true;
      if (objectUrl) {
        URL.revokeObjectURL(objectUrl);
      }
    };
  }, [accessToken, cameraTurnedOff, protectedPath]);

  useEffect(() => {
    const mediaElements = [imageRef.current, videoRef.current].filter(Boolean) as Array<
      HTMLImageElement | HTMLVideoElement
    >;
    const resizeObserver = typeof ResizeObserver !== "undefined"
      ? new ResizeObserver(() => setSurfaceVersion((current) => current + 1))
      : null;

    const handleSurfaceChange = () => setSurfaceVersion((current) => current + 1);

    if (surfaceRef.current && resizeObserver) {
      resizeObserver.observe(surfaceRef.current);
    }

    mediaElements.forEach((element) => {
      resizeObserver?.observe(element);
      element.addEventListener("loadeddata", handleSurfaceChange);
      element.addEventListener("loadedmetadata", handleSurfaceChange);
      element.addEventListener("load", handleSurfaceChange);
    });

    return () => {
      resizeObserver?.disconnect();
      mediaElements.forEach((element) => {
        element.removeEventListener("loadeddata", handleSurfaceChange);
        element.removeEventListener("loadedmetadata", handleSurfaceChange);
        element.removeEventListener("load", handleSurfaceChange);
      });
    };
  }, [protectedMedia.url, stream.stream_kind]);

  const protectedMediaReady = protectedMedia.sourcePath === protectedPath;
  const playbackUrl = protectedPath
    ? protectedMediaReady
      ? protectedMedia.url
      : null
    : stream.stream_url;
  const mediaError = protectedPath && protectedMediaReady ? protectedMedia.error : null;

  if (cameraTurnedOff) {
    return (
      <FeedFrame
        camera={camera}
        stream={stream}
        variant={variant}
        statusMessage="Camera playback is turned off until this camera is started again."
      >
        <UnavailableStreamState
          camera={camera}
          stream={stream}
          variant={variant}
          title="Camera turned off"
          description="Playback is disabled because this camera has been turned off in AegisPro."
          playbackValue="Off"
        />
      </FeedFrame>
    );
  }

  if (stream.stream_kind === "browser-camera") {
    return (
      <FeedFrame
        camera={camera}
        stream={stream}
        variant={variant}
        statusMessage="Browser camera preview is connected for live monitoring."
      >
        <div ref={surfaceRef} className="relative h-full w-full">
          <BrowserCameraPreview deviceId={stream.browser_device_id} variant={variant} videoRef={videoRef} setVideoElement={setVideoElement} />
          <DetectionOverlay
            detections={detections}
            mediaElement={activeMediaElement}
            objectFit="cover"
            refreshKey={surfaceVersion}
          />
        </div>
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
        <div ref={surfaceRef} className="relative h-full w-full">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            ref={setImageElement}
            src={playbackUrl}
            alt={`${camera.name} preview`}
            className={cn(
              "h-full w-full rounded-[24px] object-cover",
              variant === "tile" ? "max-h-[320px]" : "max-h-[480px]"
            )}
          />
          <DetectionOverlay
            detections={detections}
            mediaElement={activeMediaElement}
            objectFit="cover"
            refreshKey={surfaceVersion}
          />
        </div>
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
      <div ref={surfaceRef} className="relative h-full w-full">
        <video
          ref={setVideoElement}
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
        <DetectionOverlay
          detections={detections}
          mediaElement={activeMediaElement}
          objectFit="contain"
          refreshKey={surfaceVersion}
        />
      </div>
    </FeedFrame>
  );
});

function BrowserCameraPreview({
  deviceId,
  variant,
  videoRef,
  setVideoElement
}: {
  deviceId: string | null;
  variant: "detail" | "tile";
  videoRef: { current: HTMLVideoElement | null };
  setVideoElement: RefCallback<HTMLVideoElement>;
}) {
  const [error, setError] = useState<string | null>(null);

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
  }, [deviceId, videoRef]);

  if (error) {
    return <EmptyState title="USB preview unavailable" description={error} />;
  }

  return (
    <video
      ref={setVideoElement}
      muted
      playsInline
      className={cn(
        "h-full w-full rounded-[24px] bg-black object-cover",
        variant === "tile" ? "max-h-[320px]" : "max-h-[480px]"
      )}
    />
  );
}

function drawToBase64(
  element: HTMLVideoElement | HTMLImageElement,
  width: number,
  height: number
) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context) {
    return Promise.resolve(null);
  }
  context.drawImage(element, 0, 0, width, height);
  const dataUrl = canvas.toDataURL("image/jpeg", 0.9);
  const [, contentBase64 = ""] = dataUrl.split(",", 2);
  return Promise.resolve(
    contentBase64
      ? {
          contentBase64,
          contentType: "image/jpeg"
        }
      : null
  );
}

function DetectionOverlay({
  detections,
  mediaElement,
  objectFit,
  refreshKey
}: {
  detections: CameraDetectionScanSummary[];
  mediaElement: HTMLImageElement | HTMLVideoElement | null;
  objectFit: "contain" | "cover";
  refreshKey: number;
}) {
  void refreshKey;

  const visibleDetections = detections.filter((detection) =>
    ["weapon", "fire", "smoke", "person", "known_person", "unknown_person"].includes(detection.detection_type)
      || Boolean(detection.face_bounding_box)
  );
  const knownPeople = detections.filter(
    (detection) => detection.recognition_status === "known" && detection.identity_label
  );

  if (!mediaElement || visibleDetections.length === 0) {
    return null;
  }

  const fittedBounds = getFittedMediaBounds(mediaElement, objectFit);
  if (!fittedBounds) {
    return null;
  }

  return (
    <div className="pointer-events-none absolute inset-0 z-10 overflow-hidden rounded-[24px]">
      {visibleDetections.flatMap((detection, index) => {
        const boxes: Array<{ key: string; box: DetectionOverlayBox; kind: "primary" | "face" }> = [];
        if (shouldRenderPrimaryBox(detection)) {
          boxes.push({
            key: `${detection.track_id ?? detection.detection_type}-${index}-primary`,
            box: detection.bounding_box!,
            kind: "primary"
          });
        }
        if (detection.face_bounding_box) {
          boxes.push({
            key: `${detection.track_id ?? detection.detection_type}-${index}-face`,
            box: detection.face_bounding_box,
            kind: "face"
          });
        }

        return boxes.map(({ key, box, kind }) => {
          const style = getOverlayBoxStyle(box, fittedBounds);
          if (!style) {
            return null;
          }

          const palette = getDetectionPalette(kind === "face" ? "face" : detection.detection_type);
          const confidence = `${Math.round(detection.confidence * 100)}%`;
          const knownPersonDetails = buildKnownPersonSubtitle(detection);
          const title = kind === "face"
            ? detection.identity_label ?? "Face"
            : formatDetectionLabel(box.label || detection.detection_type);
          const subtitle = kind === "face"
            ? detection.identity_label
              ? knownPersonDetails
              : "Face detected"
            : detection.identity_label
              ? `${detection.identity_label} | ${confidence}`
              : confidence;

          return (
            <div
              key={key}
              className="absolute rounded-[18px] border-2 shadow-[0_0_0_1px_rgba(15,23,42,0.45)]"
              style={{
                ...style,
                borderColor: palette.border,
                backgroundColor: palette.fill
              }}
            >
              <div
                className="absolute left-0 top-0 max-w-[220px] -translate-y-[calc(100%+8px)] rounded-xl border px-2.5 py-1.5 text-[11px] leading-tight backdrop-blur-sm"
                style={{
                  borderColor: palette.border,
                  backgroundColor: palette.badge,
                  color: palette.text
                }}
              >
                <p className="font-semibold">{title}</p>
                <p className="mt-0.5 text-[10px] opacity-90">{subtitle}</p>
              </div>
            </div>
          );
        });
      })}

      {knownPeople.length > 0 ? (
        <div className="absolute bottom-4 right-4 max-w-[280px] rounded-[20px] border border-emerald-300/40 bg-slate-950/82 px-4 py-3 text-xs text-emerald-50 shadow-2xl backdrop-blur-md">
          <p className="text-[10px] uppercase tracking-[0.18em] text-emerald-200/80">Known Person</p>
          {knownPeople.slice(0, 2).map((person, index) => (
            <p key={`${person.track_id ?? person.identity_label}-${index}`} className="mt-2 leading-relaxed">
              {person.identity_label}
              {buildKnownPersonSubtitle(person) ? ` | ${buildKnownPersonSubtitle(person)}` : ""}
            </p>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function shouldRenderPrimaryBox(detection: CameraDetectionScanSummary) {
  return Boolean(
    detection.bounding_box
      && ["weapon", "fire", "smoke", "person", "known_person", "unknown_person"].includes(detection.detection_type)
  );
}

function formatDetectionLabel(label: string) {
  return label.replaceAll("_", " ").replace(/\b\w/g, (match) => match.toUpperCase());
}

function buildKnownPersonSubtitle(detection: CameraDetectionScanSummary) {
  const detailValues = [
    readMetadataString(detection.metadata, "title"),
    readMetadataString(detection.metadata, "department"),
    readMetadataString(detection.metadata, "reference_id"),
    readMetadataString(detection.metadata, "person_type")
  ].filter(Boolean);
  return detailValues.join(" | ");
}

function readMetadataString(metadata: Record<string, unknown>, key: string) {
  const value = metadata[key];
  return typeof value === "string" && value.trim() ? value : "";
}

function getDetectionPalette(detectionType: string) {
  switch (detectionType) {
    case "weapon":
      return {
        border: "rgba(248, 113, 113, 0.95)",
        fill: "rgba(127, 29, 29, 0.12)",
        badge: "rgba(69, 10, 10, 0.86)",
        text: "rgb(254 226 226)"
      };
    case "fire":
      return {
        border: "rgba(251, 146, 60, 0.95)",
        fill: "rgba(154, 52, 18, 0.14)",
        badge: "rgba(124, 45, 18, 0.88)",
        text: "rgb(255 237 213)"
      };
    case "smoke":
      return {
        border: "rgba(148, 163, 184, 0.95)",
        fill: "rgba(51, 65, 85, 0.16)",
        badge: "rgba(15, 23, 42, 0.88)",
        text: "rgb(226 232 240)"
      };
    case "known_person":
      return {
        border: "rgba(74, 222, 128, 0.95)",
        fill: "rgba(20, 83, 45, 0.1)",
        badge: "rgba(6, 78, 59, 0.88)",
        text: "rgb(220 252 231)"
      };
    case "person":
      return {
        border: "rgba(56, 189, 248, 0.95)",
        fill: "rgba(14, 116, 144, 0.08)",
        badge: "rgba(8, 47, 73, 0.88)",
        text: "rgb(224 242 254)"
      };
    case "unknown_person":
    case "face":
    default:
      return {
        border: "rgba(56, 189, 248, 0.95)",
        fill: "rgba(14, 116, 144, 0.08)",
        badge: "rgba(8, 47, 73, 0.88)",
        text: "rgb(224 242 254)"
      };
  }
}

function getOverlayBoxStyle(
  box: DetectionOverlayBox,
  fittedBounds: { left: number; top: number; width: number; height: number; naturalWidth: number; naturalHeight: number; }
) {
  const scaleX = fittedBounds.width / fittedBounds.naturalWidth;
  const scaleY = fittedBounds.height / fittedBounds.naturalHeight;
  const left = fittedBounds.left + (box.x1 * scaleX);
  const top = fittedBounds.top + (box.y1 * scaleY);
  const width = Math.max((box.x2 - box.x1) * scaleX, 2);
  const height = Math.max((box.y2 - box.y1) * scaleY, 2);

  if (![left, top, width, height].every(Number.isFinite)) {
    return null;
  }

  return { left, top, width, height };
}

function getFittedMediaBounds(
  mediaElement: HTMLImageElement | HTMLVideoElement,
  objectFit: "contain" | "cover"
) {
  const rect = mediaElement.getBoundingClientRect();
  const naturalWidth = "videoWidth" in mediaElement ? mediaElement.videoWidth : mediaElement.naturalWidth;
  const naturalHeight = "videoHeight" in mediaElement ? mediaElement.videoHeight : mediaElement.naturalHeight;

  if (!rect.width || !rect.height || !naturalWidth || !naturalHeight) {
    return null;
  }

  const mediaRatio = naturalWidth / naturalHeight;
  const frameRatio = rect.width / rect.height;
  const fitByWidth = objectFit === "contain" ? mediaRatio > frameRatio : mediaRatio < frameRatio;
  const width = fitByWidth ? rect.width : rect.height * mediaRatio;
  const height = fitByWidth ? rect.width / mediaRatio : rect.height;

  return {
    left: (rect.width - width) / 2,
    top: (rect.height - height) / 2,
    width,
    height,
    naturalWidth,
    naturalHeight
  };
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
  variant,
  title = "Preview unavailable",
  description,
  playbackValue = "Unavailable"
}: {
  camera: Camera;
  stream: CameraStreamDescriptor;
  variant: "detail" | "tile";
  title?: string;
  description?: string;
  playbackValue?: string;
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
            <p className="text-sm font-medium">{title}</p>
            <p className="mt-1 text-sm text-slate-400">
              {description ?? stream.health_message ?? "This source does not currently expose a browser-playable feed."}
            </p>
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
        <MonitorStat label="Playback" value={playbackValue === "Unavailable" && stream.stream_url ? "Configured" : playbackValue} tone="slate" />
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
