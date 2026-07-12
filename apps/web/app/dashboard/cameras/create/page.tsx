"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ArrowLeft } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { useForm } from "react-hook-form";
import { CameraForm, buildCameraPayload, type CameraFormValues } from "@/components/camera-form";
import { EmptyState, SectionCard } from "@/components/dashboard-ui";
import { createCamera } from "@/lib/api";
import { useAuthStore } from "@/lib/auth-store";
import { cameraFormSchema } from "@/lib/camera-schema";

export default function CreateCameraPage() {
  const router = useRouter();
  const queryClient = useQueryClient();
  const { accessToken, user, logout } = useAuthStore();
  const [error, setError] = useState<string | null>(null);
  const form = useForm<CameraFormValues>({
    resolver: zodResolver(cameraFormSchema),
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

  const createCameraMutation = useMutation({
    mutationFn: async (values: CameraFormValues) => {
      if (!accessToken) {
        throw new Error("You need to sign in again before registering a camera.");
      }

      return createCamera(accessToken, buildCameraPayload(values, user?.email));
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

  async function onSubmit(values: CameraFormValues) {
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
        title="Add camera"
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
        title="Add camera"
        description="Create a new camera source for the live operations registry."
        action={<BackLink />}
      >
        <CameraForm
          form={form}
          error={error}
          isSubmitting={createCameraMutation.isPending}
          submitLabel="Add camera"
          submittingLabel="Adding camera"
          onCancel={() => router.push("/dashboard/cameras")}
          onSubmit={onSubmit}
        />
      </SectionCard>
    </div>
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
