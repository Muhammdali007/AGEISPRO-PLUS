"use client";

import { Info } from "lucide-react";
import type { ReactNode } from "react";
import type { UseFormReturn } from "react-hook-form";
import { useWatch } from "react-hook-form";
import type { CameraCreateInput, CameraSourceType, UserRole } from "@/lib/api";
import { Button } from "@/components/button";

export const cameraSourceTypes = ["usb", "rtsp", "http", "file"] as const satisfies readonly CameraSourceType[];
export const registrationRoles = ["administrator", "supervisor", "operator", "viewer"] as const satisfies readonly UserRole[];

export type CameraFormValues = {
  name: string;
  registration_role: UserRole;
  source_type: CameraSourceType;
  source: string;
  location: string;
  group: string;
  tags: string;
  inference_fps: number;
  detection_enabled: boolean;
  metadata: string;
};

export function CameraForm({
  form,
  error,
  isSubmitting,
  submitLabel,
  submittingLabel,
  onCancel,
  onSubmit
}: {
  form: UseFormReturn<CameraFormValues>;
  error: string | null;
  isSubmitting: boolean;
  submitLabel: string;
  submittingLabel: string;
  onCancel: () => void;
  onSubmit: (values: CameraFormValues) => void | Promise<void>;
}) {
  const selectedSourceType = useWatch({
    control: form.control,
    name: "source_type"
  });

  return (
    <form className="grid gap-5 lg:grid-cols-2" onSubmit={form.handleSubmit(onSubmit)}>
      <FormField label="Camera name" error={form.formState.errors.name?.message}>
        <input
          className="h-11 w-full rounded-md border border-border bg-background px-3 text-sm outline-none transition focus:border-accent"
          type="text"
          {...form.register("name")}
        />
      </FormField>

      <FormField label="Registering as role" error={form.formState.errors.registration_role?.message}>
        <select
          className="h-11 w-full rounded-md border border-border bg-background px-3 text-sm outline-none transition focus:border-accent"
          {...form.register("registration_role")}
        >
          {registrationRoles.map((roleOption) => (
            <option key={roleOption} value={roleOption}>
              {roleOption}
            </option>
          ))}
        </select>
      </FormField>

      <FormField label="Source type" error={form.formState.errors.source_type?.message}>
        <select
          className="h-11 w-full rounded-md border border-border bg-background px-3 text-sm outline-none transition focus:border-accent"
          {...form.register("source_type")}
        >
          {cameraSourceTypes.map((sourceType) => (
            <option key={sourceType} value={sourceType}>
              {sourceType.toUpperCase()}
            </option>
          ))}
        </select>
      </FormField>

      <FormField label="Source" error={form.formState.errors.source?.message}>
        <input
          className="h-11 w-full rounded-md border border-border bg-background px-3 text-sm outline-none transition focus:border-accent"
          type="text"
          placeholder={sourcePlaceholder(selectedSourceType)}
          {...form.register("source")}
        />
      </FormField>

      <FormField label="Inference FPS" error={form.formState.errors.inference_fps?.message}>
        <input
          className="h-11 w-full rounded-md border border-border bg-background px-3 text-sm outline-none transition focus:border-accent"
          type="number"
          min={1}
          max={30}
          {...form.register("inference_fps", { valueAsNumber: true })}
        />
      </FormField>

      <FormField label="Location" error={form.formState.errors.location?.message}>
        <input
          className="h-11 w-full rounded-md border border-border bg-background px-3 text-sm outline-none transition focus:border-accent"
          type="text"
          {...form.register("location")}
        />
      </FormField>

      <FormField label="Group" error={form.formState.errors.group?.message}>
        <input
          className="h-11 w-full rounded-md border border-border bg-background px-3 text-sm outline-none transition focus:border-accent"
          type="text"
          {...form.register("group")}
        />
      </FormField>

      <FormField label="Tags" error={undefined}>
        <input
          className="h-11 w-full rounded-md border border-border bg-background px-3 text-sm outline-none transition focus:border-accent"
          type="text"
          placeholder="entry-gate, west-wing"
          {...form.register("tags")}
        />
      </FormField>

      <label className="flex items-center gap-3 rounded-[20px] border border-white/10 bg-black/15 px-4 py-3">
        <input className="h-4 w-4 accent-emerald-400" type="checkbox" {...form.register("detection_enabled")} />
        <span>
          <span className="block text-sm font-medium text-slate-100">Detection enabled</span>
          <span className="block text-sm text-slate-400">Allow this source to participate in inference workflows immediately.</span>
        </span>
      </label>

      <label className="rounded-[20px] border border-cyan-500/20 bg-cyan-500/5 p-4 text-sm text-slate-300 lg:col-span-2">
        <span className="flex items-center gap-2 font-medium text-cyan-100">
          <Info size={16} aria-hidden="true" />
          Source hints
        </span>
        <span className="mt-2 block text-slate-400">{sourceHelp(selectedSourceType)}</span>
        <span className="mt-2 block text-slate-500">
          The selected role will be stored with the camera registration metadata for auditing.
        </span>
      </label>

      <FormField label="Metadata JSON" error={form.formState.errors.metadata?.message}>
        <textarea
          className="min-h-36 w-full rounded-md border border-border bg-background px-3 py-3 text-sm outline-none transition focus:border-accent"
          placeholder={metadataPlaceholder(selectedSourceType)}
          {...form.register("metadata")}
        />
      </FormField>

      {error ? (
        <div className="rounded-md border border-danger/50 bg-danger/10 px-3 py-2 text-sm text-red-200 lg:col-span-2">
          {error}
        </div>
      ) : null}

      <div className="flex flex-wrap items-center gap-3 lg:col-span-2">
        <Button type="submit" disabled={isSubmitting}>
          {isSubmitting ? submittingLabel : submitLabel}
        </Button>
        <Button type="button" variant="ghost" onClick={onCancel}>
          Cancel
        </Button>
      </div>
    </form>
  );
}

export function buildCameraPayload(values: CameraFormValues, userEmail: string | null | undefined): CameraCreateInput {
  const normalizedSource = normalizeCameraSource(values.source);
  const metadata = values.metadata.trim() ? (JSON.parse(values.metadata) as Record<string, unknown>) : {};

  if (values.source_type === "http") {
    applyIpWebcamDefaults(normalizedSource, metadata);
  }

  return {
    name: values.name,
    source_type: values.source_type,
    source: normalizedSource,
    location: values.location || null,
    group: values.group || null,
    tags: values.tags
      .split(",")
      .map((tag) => tag.trim())
      .filter(Boolean),
    detection_enabled: values.detection_enabled,
    inference_fps: values.inference_fps,
    metadata: {
      ...metadata,
      registered_by_role: values.registration_role,
      registered_by_email: userEmail ?? null
    }
  };
}

function FormField({
  label,
  error,
  children
}: {
  label: string;
  error?: string;
  children: ReactNode;
}) {
  return (
    <label className="block">
      <span className="mb-2 block text-sm text-slate-300">{label}</span>
      {children}
      {error ? <span className="mt-1 block text-sm text-danger">{error}</span> : null}
    </label>
  );
}

function sourcePlaceholder(sourceType: CameraSourceType) {
  if (sourceType === "usb") {
    return "0";
  }

  if (sourceType === "rtsp") {
    return "rtsp://username:password@camera-host:554/stream";
  }

  if (sourceType === "http") {
    return "https://camera-host/live.m3u8";
  }

  return "storage/uploads/site-entry.mp4";
}

function sourceHelp(sourceType: CameraSourceType) {
  if (sourceType === "usb") {
    return "Use the device index visible to the backend host, such as 0 or 1. Add browser metadata if you want direct in-browser USB preview.";
  }

  if (sourceType === "rtsp") {
    return "Use the full RTSP address. If browser preview needs a relay, include metadata like relay_url.";
  }

  if (sourceType === "http") {
    return "Use an HTTP or HTTPS stream URL. Metadata can include stream_url, stream_format, or insecure_tls for phone/webcam feeds that need extra hints.";
  }

  return "Use a file path that remains inside the configured media storage root on the API server.";
}

function metadataPlaceholder(sourceType: CameraSourceType) {
  if (sourceType === "usb") {
    return '{\n  "browser_device_id": "optional-browser-device-id"\n}';
  }

  if (sourceType === "rtsp") {
    return '{\n  "relay_url": "https://relay.example.com/camera.m3u8"\n}';
  }

  if (sourceType === "http") {
    return '{\n  "stream_format": "mjpeg",\n  "stream_url": "https://phone-ip:port/video",\n  "insecure_tls": true\n}';
  }

  return "{\n}";
}

function normalizeCameraSource(source: string) {
  return source
    .trim()
    .replace(/^(ipv4|ipv6)\s*:\s*/i, "")
    .replace(/^ip(?:v4|v6)\s+/i, "");
}

function applyIpWebcamDefaults(source: string, metadata: Record<string, unknown>) {
  if (!looksLikePrivateHttpRoot(source)) {
    return;
  }

  if (typeof metadata.stream_url !== "string" || !metadata.stream_url.trim()) {
    metadata.stream_url = `${source.replace(/\/+$/, "")}/video`;
  }

  if (typeof metadata.stream_format !== "string" || !metadata.stream_format.trim()) {
    metadata.stream_format = "mjpeg";
  }
}

function looksLikePrivateHttpRoot(source: string) {
  try {
    const url = new URL(source);
    if (!["http:", "https:"].includes(url.protocol)) {
      return false;
    }
    if (url.pathname !== "/" && url.pathname !== "") {
      return false;
    }
    return /^(localhost|127\.0\.0\.1|192\.168\.|10\.|172\.(1[6-9]|2\d|3[0-1])\.)/i.test(url.hostname);
  } catch {
    return false;
  }
}
