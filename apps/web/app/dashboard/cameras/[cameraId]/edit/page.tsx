"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowLeft } from "lucide-react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { useForm } from "react-hook-form";
import { CameraForm, buildCameraPayload, type CameraFormValues } from "@/components/camera-form";
import { EmptyState, SectionCard } from "@/components/dashboard-ui";
import { getCamera, updateCamera, type CameraUpdateInput } from "@/lib/api";
import { useAuthStore } from "@/lib/auth-store";
import { buildCameraFormSchema } from "@/lib/camera-schema";

export default function EditCameraPage() {
  const params = useParams<{ cameraId: string }>();
  const router = useRouter();
  const queryClient = useQueryClient();
  const { accessToken, user, logout } = useAuthStore();
  const [error, setError] = useState<string | null>(null);
  const form = useForm<CameraFormValues>({
    resolver: zodResolver(buildCameraFormSchema({ requireSource: false })),
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

  const cameraQuery = useQuery({
    queryKey: ["camera", params.cameraId, accessToken],
    queryFn: async () => getCamera(accessToken!, params.cameraId),
    enabled: Boolean(accessToken && params.cameraId),
    retry: false
  });

  useEffect(() => {
    if (!cameraQuery.data) {
      return;
    }

    form.reset({
      name: cameraQuery.data.name,
      registration_role: user?.role ?? "operator",
      source_type: cameraQuery.data.source_type,
      source: "",
      location: cameraQuery.data.location ?? "",
      group: cameraQuery.data.group ?? "",
      tags: cameraQuery.data.tags.join(", "),
      inference_fps: cameraQuery.data.inference_fps,
      detection_enabled: cameraQuery.data.detection_enabled,
      metadata: ""
    });
  }, [cameraQuery.data, form, user?.role]);

  const updateCameraMutation = useMutation({
    mutationFn: async (values: CameraFormValues) => {
      if (!accessToken) {
        throw new Error("You need to sign in again before editing this camera.");
      }

      const payload: CameraUpdateInput = { ...buildCameraPayload(values, user?.email) };
      if (!values.source.trim()) {
        delete payload.source;
      }
      if (!values.metadata.trim()) {
        delete payload.metadata;
      }
      return updateCamera(accessToken, params.cameraId, payload);
    },
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["camera", params.cameraId, accessToken] }),
        queryClient.invalidateQueries({ queryKey: ["camera-stream", params.cameraId, accessToken] }),
        queryClient.invalidateQueries({ queryKey: ["cameras", "list", accessToken] })
      ]);
      router.push(`/dashboard/cameras/${params.cameraId}`);
    },
    onError: (cause) => {
      if (cause instanceof Error && (cause.message === "Invalid credentials" || cause.message === "Session expired")) {
        logout();
        router.push("/login");
        return;
      }
      setError(cause instanceof Error ? cause.message : "Unable to update camera");
    }
  });

  async function onSubmit(values: CameraFormValues) {
    setError(null);
    try {
      await updateCameraMutation.mutateAsync(values);
    } catch {
      // The mutation's onError handler already renders the user-facing message.
    }
  }

  if (user?.role && user.role !== "administrator" && user.role !== "supervisor") {
    return (
      <SectionCard
        title="Edit camera"
        description="Only administrators and supervisors can edit camera sources."
        action={<BackLink cameraId={params.cameraId} />}
      >
        <EmptyState
          title="Elevated access required"
          description="Your current account can view camera inventory, but only supervisors and administrators can edit sources."
        />
      </SectionCard>
    );
  }

  return (
    <div className="space-y-6">
      <SectionCard
        title={cameraQuery.data ? `Edit ${cameraQuery.data.name}` : "Edit camera"}
        description="Update the camera source, routing, and detection settings for this stream."
        action={<BackLink cameraId={params.cameraId} />}
      >
        {cameraQuery.error instanceof Error ? (
          <EmptyState title="Camera unavailable" description={cameraQuery.error.message} />
        ) : !cameraQuery.data ? (
          <EmptyState title="Loading camera" description="Fetching the current camera configuration from the API." />
        ) : (
          <CameraForm
            form={form}
            error={error}
            isSubmitting={updateCameraMutation.isPending}
            submitLabel="Save changes"
            submittingLabel="Saving changes"
            sourceFieldLabel="Replace source"
            sourceFieldHint={
              `Current protected source: ${cameraQuery.data.source}. Leave this blank to keep the existing encrypted source.`
            }
            onCancel={() => router.push(`/dashboard/cameras/${params.cameraId}`)}
            onSubmit={onSubmit}
          />
        )}
      </SectionCard>
    </div>
  );
}

function BackLink({ cameraId }: { cameraId: string }) {
  return (
    <Link
      href={`/dashboard/cameras/${cameraId}`}
      className="inline-flex items-center gap-2 text-sm font-medium text-accent transition hover:text-emerald-300"
    >
      <ArrowLeft size={15} aria-hidden="true" />
      Back to camera
    </Link>
  );
}
