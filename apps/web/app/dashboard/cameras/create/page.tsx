"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ArrowLeft, Camera, Info } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import type { ReactNode } from "react";
import { useForm, useWatch } from "react-hook-form";
import { z } from "zod";
import { Button } from "@/components/button";
import { EmptyState, SectionCard } from "@/components/dashboard-ui";
import { createCamera, type CameraSourceType, type UserRole } from "@/lib/api";
import { useAuthStore } from "@/lib/auth-store";

const cameraSourceTypes = ["usb", "rtsp", "http", "file"] as const satisfies readonly CameraSourceType[];
const registrationRoles = ["administrator", "supervisor", "operator", "viewer"] as const satisfies readonly UserRole[];

const createCameraSchema = z.object({
  name: z.string().min(1, "Camera name is required").max(160, "Camera name is too long"),
  registration_role: z.enum(registrationRoles),
  source_type: z.enum(cameraSourceTypes),
  source: z.string().min(1, "Source is required"),
  location: z.string().max(255, "Location is too long").optional().or(z.literal("")),
  group: z.string().max(120, "Group is too long").optional().or(z.literal("")),
  tags: z.string(),
  inference_fps: z.coerce.number().int().min(1, "Inference FPS must be at least 1").max(30, "Inference FPS cannot exceed 30"),
  detection_enabled: z.boolean(),
  metadata: z
    .string()
    .superRefine((value, ctx) => {
      if (!value.trim()) {
        return;
      }

      try {
        const parsed = JSON.parse(value) as unknown;
        if (!parsed || Array.isArray(parsed) || typeof parsed !== "object") {
          ctx.addIssue({
            code: z.ZodIssueCode.custom,
            message: "Metadata must be a JSON object."
          });
        }
      } catch {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: "Metadata must be valid JSON."
        });
      }
    })
});

type CreateCameraForm = z.infer<typeof createCameraSchema>;

export default function CreateCameraPage() {
  const router = useRouter();
  const queryClient = useQueryClient();
  const { accessToken, user, logout } = useAuthStore();
  const [error, setError] = useState<string | null>(null);
  const form = useForm<CreateCameraForm>({
    resolver: zodResolver(createCameraSchema),
    defaultValues: {
      name: "",
      registration_role: user?.role ?? "operator",
      source_type: "usb",
      source: "",
      location: "",
      group: "",
      tags: "",
      inference_fps: 5,
      detection_enabled: true,
      metadata: ""
    }
  });

  const selectedSourceType = useWatch({
    control: form.control,
    name: "source_type"
  });

  const createCameraMutation = useMutation({
    mutationFn: async (values: CreateCameraForm) => {
      if (!accessToken) {
        throw new Error("You need to sign in again before registering a camera.");
      }

      const normalizedSource = normalizeCameraSource(values.source);
      const metadata = values.metadata.trim() ? (JSON.parse(values.metadata) as Record<string, unknown>) : {};

      if (values.source_type === "http") {
        applyIpWebcamDefaults(normalizedSource, metadata);
      }

      return createCamera(accessToken, {
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
          registered_by_email: user?.email ?? null
        }
      });
    },
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["cameras", "list", accessToken] });
      router.push("/dashboard/cameras");
    },
    onError: (cause) => {
      if (cause instanceof Error && (cause.message === "Invalid credentials" || cause.message === "Session expired")) {
        logout();
        router.push("/login");
        return;
      }
      setError(cause instanceof Error ? cause.message : "Unable to register camera");
    }
  });

  async function onSubmit(values: CreateCameraForm) {
    setError(null);
    try {
      await createCameraMutation.mutateAsync(values);
    } catch {
      // The mutation's onError handler already renders the user-facing message.
    }
  }

  if (user?.role && user.role !== "administrator" && user.role !== "supervisor" && user.role !== "operator") {
    return (
      <SectionCard
        title="Register camera"
        description="Only administrators, supervisors, and operators can add camera sources."
        action={<BackLink />}
      >
        <EmptyState
          title="Elevated access required"
          description="Your current account can view camera inventory, but only operators, supervisors, and administrators can register new sources."
        />
      </SectionCard>
    );
  }

  return (
    <div className="space-y-6">
      <SectionCard
        title="Register camera"
        description="Create a new camera source for the live operations registry."
        action={<BackLink />}
      >
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
              {...form.register("inference_fps")}
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
            <Button type="submit" disabled={createCameraMutation.isPending}>
              <Camera size={16} aria-hidden="true" />
              {createCameraMutation.isPending ? "Registering camera" : "Register camera"}
            </Button>
            <Button type="button" variant="ghost" onClick={() => router.push("/dashboard/cameras")}>
              Cancel
            </Button>
          </div>
        </form>
      </SectionCard>
    </div>
  );
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

function BackLink() {
  return (
    <Link
      href="/dashboard/cameras"
      className="inline-flex items-center gap-2 text-sm font-medium text-accent transition hover:text-emerald-300"
    >
      <ArrowLeft size={15} aria-hidden="true" />
      Back to cameras
    </Link>
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
