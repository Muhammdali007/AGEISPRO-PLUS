"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowLeft, Trash2 } from "lucide-react";
import Image from "next/image";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import type { FormEvent, ReactNode } from "react";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { Button } from "@/components/button";
import { EmptyState, SectionCard } from "@/components/dashboard-ui";
import {
  deletePerson,
  fetchProtectedMedia,
  getPerson,
  updatePerson,
  uploadPersonFaceImages,
  validatePersonFaceUpload
} from "@/lib/api";
import { useAuthStore } from "@/lib/auth-store";
import { formatDateTime, statusTone } from "@/lib/format";
import { cn } from "@/lib/cn";

const personTypes = ["employee", "student", "visitor", "contractor", "other"] as const;

const updatePersonSchema = z.object({
  full_name: z.string().min(1, "Full name is required").max(160, "Full name is too long"),
  person_type: z.enum(personTypes),
  department: z.string().max(120, "Department is too long").optional().or(z.literal("")),
  reference_id: z.string().max(64, "Reference ID is too long").optional().or(z.literal("")),
  title: z.string().max(120, "Title is too long").optional().or(z.literal("")),
  is_active: z.boolean()
});

type UpdatePersonForm = z.infer<typeof updatePersonSchema>;

export default function PersonDetailPage() {
  const params = useParams<{ personId: string }>();
  const router = useRouter();
  const queryClient = useQueryClient();
  const { accessToken, user, logout } = useAuthStore();
  const [profileError, setProfileError] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [uploadFiles, setUploadFiles] = useState<File[]>([]);
  const [uploadPrimary, setUploadPrimary] = useState(false);
  const canManage = user?.role === "administrator" || user?.role === "supervisor";

  const personQuery = useQuery({
    queryKey: ["persons", "detail", params.personId, accessToken],
    queryFn: () => getPerson(accessToken!, params.personId),
    enabled: Boolean(accessToken && params.personId),
    retry: false
  });

  const person = personQuery.data;
  const updateForm = useForm<UpdatePersonForm>({
    resolver: zodResolver(updatePersonSchema),
    defaultValues: {
      full_name: "",
      person_type: "visitor",
      department: "",
      reference_id: "",
      title: "",
      is_active: true
    }
  });

  useEffect(() => {
    if (person) {
      updateForm.reset({
        full_name: person.full_name,
        person_type: person.person_type,
        department: person.department ?? "",
        reference_id: person.reference_id,
        title: person.title ?? "",
        is_active: person.is_active
      });
    }
  }, [person, updateForm]);

  const updateMutation = useMutation({
    mutationFn: async (values: UpdatePersonForm) => {
      if (!accessToken) {
        throw new Error("You need to sign in again before updating a person.");
      }
      return updatePerson(accessToken, params.personId, {
        full_name: values.full_name,
        person_type: values.person_type,
        department: values.department || null,
        reference_id: values.reference_id || null,
        title: values.title || null,
        is_active: values.is_active
      });
    },
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["persons", "detail", params.personId, accessToken] });
      await queryClient.invalidateQueries({ queryKey: ["persons", "list", accessToken] });
    },
    onError: (cause) => {
      if (cause instanceof Error && (cause.message === "Invalid credentials" || cause.message === "Session expired")) {
        logout();
        router.push("/login");
        return;
      }
      setProfileError(cause instanceof Error ? cause.message : "Unable to update person");
    }
  });

  const uploadMutation = useMutation({
    mutationFn: async () => {
      if (!accessToken) {
        throw new Error("You need to sign in again before uploading face images.");
      }
      return uploadPersonFaceImages(accessToken, params.personId, {
        files: uploadFiles,
        is_primary: uploadPrimary
      });
    },
    onSuccess: async () => {
      setUploadFiles([]);
      setUploadPrimary(false);
      await queryClient.invalidateQueries({ queryKey: ["persons", "detail", params.personId, accessToken] });
      await queryClient.invalidateQueries({ queryKey: ["persons", "list", accessToken] });
    },
    onError: (cause) => {
      if (cause instanceof Error && (cause.message === "Invalid credentials" || cause.message === "Session expired")) {
        logout();
        router.push("/login");
        return;
      }
      setUploadError(cause instanceof Error ? cause.message : "Unable to upload face images");
    }
  });

  const deleteMutation = useMutation({
    mutationFn: async () => {
      if (!accessToken) {
        throw new Error("You need to sign in again before deleting a person.");
      }
      await deletePerson(accessToken, params.personId);
    },
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["persons", "list", accessToken] });
      router.push("/dashboard/persons");
    },
    onError: (cause) => {
      if (cause instanceof Error && (cause.message === "Invalid credentials" || cause.message === "Session expired")) {
        logout();
        router.push("/login");
        return;
      }
      setDeleteError(cause instanceof Error ? cause.message : "Unable to delete person");
    }
  });

  async function onProfileSubmit(values: UpdatePersonForm) {
    setProfileError(null);
    await updateMutation.mutateAsync(values);
  }

  async function onUploadSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setUploadError(null);
    if (uploadFiles.length === 0) {
      setUploadError("Choose one or more face images first.");
      return;
    }
    try {
      validatePersonFaceUpload({
        files: uploadFiles,
        is_primary: uploadPrimary
      });
    } catch (cause) {
      setUploadError(cause instanceof Error ? cause.message : "Unable to upload face images");
      return;
    }
    await uploadMutation.mutateAsync();
  }

  async function onDelete() {
    if (!person) {
      return;
    }
    setDeleteError(null);
    const confirmed = window.confirm(`Delete ${person.full_name}? This will remove the person profile and enrolled faces.`);
    if (!confirmed) {
      return;
    }
    await deleteMutation.mutateAsync();
  }

  return (
    <div className="space-y-6">
      <SectionCard
        title={person ? person.full_name : "Known person"}
        description="Profile details, recognition history, and enrolled face metadata for Phase 6."
        action={<BackLink />}
      >
        {personQuery.error instanceof Error ? (
          <EmptyState title="Person unavailable" description={personQuery.error.message} />
        ) : !person ? (
          <EmptyState title="Loading person" description="Fetching profile, recognition history, and face metadata from the API." />
        ) : (
          <div className="grid gap-4 lg:grid-cols-2">
            <DetailTile label="Person type" value={person.person_type} tone="bg-black/20 text-slate-200" />
            <DetailTile label="Reference ID" value={person.reference_id} tone="bg-black/20 text-slate-200" />
            <DetailTile label="Department" value={person.department ?? "Not set"} tone="bg-black/20 text-slate-200" />
            <DetailTile label="Title" value={person.title ?? "Not set"} tone="bg-black/20 text-slate-200" />
            <DetailTile label="Profile state" value={person.is_active ? "Active" : "Inactive"} tone={statusTone(person.is_active ? "ok" : "disabled")} />
            <DetailTile label="Recognition count" value={`${person.recognition_count}`} tone="bg-emerald-500/15 text-emerald-100" />
            <DetailTile label="Visit count" value={`${person.visit_count}`} tone="bg-cyan-500/15 text-cyan-100" />
            <DetailTile label="Last seen" value={formatDateTime(person.last_seen_at)} tone="bg-black/20 text-slate-200" />
            <DetailTile label="Last recognized" value={formatDateTime(person.last_recognized_at)} tone="bg-black/20 text-slate-200" />
          </div>
        )}
      </SectionCard>

      {person ? (
        <SectionCard title="Enrolled faces" description="Face references and embedding metadata available to the recognition pipeline.">
          {person.face_profiles.length === 0 ? (
            <EmptyState
              title="No faces enrolled"
              description="Enroll the first face metadata record to give the recognition pipeline a profile to work with."
            />
          ) : (
            <div className="space-y-3">
              {person.face_profiles.map((profile) => (
                <div key={profile.id} className="grid gap-4 rounded-[22px] border border-white/10 bg-black/15 p-4 md:grid-cols-[1.2fr_0.8fr_0.8fr]">
                  <div>
                    <ProtectedFaceImage
                      accessToken={accessToken}
                      personId={person.id}
                      faceId={profile.id}
                      alt={profile.label}
                    />
                    <p className="font-medium">{profile.label}</p>
                    <p className="mt-2 break-all text-sm text-slate-400">{profile.image_path}</p>
                  </div>
                  <div>
                    <p className="text-xs uppercase tracking-[0.18em] text-slate-500">Embedding</p>
                    <p className="mt-2 text-sm text-slate-200">
                      {profile.embedding_model ?? "Not set"} ({profile.embedding_dimensions} dims)
                    </p>
                  </div>
                  <div>
                    <p className="text-xs uppercase tracking-[0.18em] text-slate-500">Captured</p>
                    <p className="mt-2 text-sm text-slate-200">{formatDateTime(profile.captured_at)}</p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </SectionCard>
      ) : null}

      {person && canManage ? (
        <SectionCard title="Upload face images" description="Add 3-5 clear single-person views (front, slight left/right, and the real camera angle). Quality checks reject group photos and faces that are too small or unclear.">
          <form className="grid gap-5" onSubmit={onUploadSubmit}>
            <FormField label="Face images">
              <input
                className="block w-full text-sm text-slate-300 file:mr-4 file:rounded-md file:border-0 file:bg-emerald-500/15 file:px-4 file:py-2 file:text-sm file:font-medium file:text-emerald-100"
                type="file"
                accept="image/*"
                multiple
                onChange={(event) => {
                  const files = Array.from(event.target.files ?? []);
                  try {
                    validatePersonFaceUpload({ files });
                    setUploadFiles(files);
                    setUploadError(null);
                  } catch (cause) {
                    setUploadFiles([]);
                    event.target.value = "";
                    setUploadError(cause instanceof Error ? cause.message : "Unable to select face images");
                  }
                }}
              />
            </FormField>

            {uploadFiles.length > 0 ? (
              <div className="rounded-[20px] border border-white/10 bg-black/15 px-4 py-3 text-sm text-slate-300">
                {uploadFiles.map((file) => file.name).join(", ")}
              </div>
            ) : null}

            <label className="flex items-center gap-3 rounded-[20px] border border-white/10 bg-black/15 px-4 py-3">
              <input
                className="h-4 w-4 accent-emerald-400"
                type="checkbox"
                checked={uploadPrimary}
                onChange={(event) => setUploadPrimary(event.target.checked)}
              />
              <span>
                <span className="block text-sm font-medium text-slate-100">Mark first image as primary</span>
                <span className="block text-sm text-slate-400">The first selected image will be flagged as the preferred enrollment record.</span>
              </span>
            </label>

            {uploadError ? (
              <div className="rounded-md border border-danger/50 bg-danger/10 px-3 py-2 text-sm text-red-200">
                {uploadError}
              </div>
            ) : null}

            <div className="flex flex-wrap items-center gap-3">
              <Button type="submit" disabled={uploadMutation.isPending}>
                {uploadMutation.isPending ? "Uploading images" : "Upload images"}
              </Button>
            </div>
          </form>
        </SectionCard>
      ) : null}

      {person && canManage ? (
        <SectionCard title="Update profile" description="Adjust identity metadata or deactivate the profile without losing recognition history.">
          <form className="grid gap-5 lg:grid-cols-2" onSubmit={updateForm.handleSubmit(onProfileSubmit)}>
            <FormField label="Full name" error={updateForm.formState.errors.full_name?.message}>
              <input
                className="h-11 w-full rounded-md border border-border bg-background px-3 text-sm outline-none transition focus:border-accent"
                type="text"
                {...updateForm.register("full_name")}
              />
            </FormField>

            <FormField label="Person type" error={updateForm.formState.errors.person_type?.message}>
              <select
                className="h-11 w-full rounded-md border border-border bg-background px-3 text-sm outline-none transition focus:border-accent"
                {...updateForm.register("person_type")}
              >
                <option value="visitor">Visitor</option>
                <option value="student">Student</option>
                <option value="employee">Employee</option>
                <option value="contractor">Contractor</option>
                <option value="other">Other</option>
              </select>
            </FormField>

            <FormField label="Reference ID" error={updateForm.formState.errors.reference_id?.message}>
              <input
                className="h-11 w-full rounded-md border border-border bg-background px-3 text-sm outline-none transition focus:border-accent"
                type="text"
                placeholder="Optional: employee ID, student ID, badge ID, visitor code"
                {...updateForm.register("reference_id")}
              />
            </FormField>

            <FormField label="Department" error={updateForm.formState.errors.department?.message}>
              <input
                className="h-11 w-full rounded-md border border-border bg-background px-3 text-sm outline-none transition focus:border-accent"
                type="text"
                {...updateForm.register("department")}
              />
            </FormField>

            <FormField label="Title" error={updateForm.formState.errors.title?.message}>
              <input
                className="h-11 w-full rounded-md border border-border bg-background px-3 text-sm outline-none transition focus:border-accent"
                type="text"
                {...updateForm.register("title")}
              />
            </FormField>

            <label className="flex items-center gap-3 rounded-[20px] border border-white/10 bg-black/15 px-4 py-3 lg:col-span-2">
              <input className="h-4 w-4 accent-emerald-400" type="checkbox" {...updateForm.register("is_active")} />
              <span>
                <span className="block text-sm font-medium text-slate-100">Profile active</span>
                <span className="block text-sm text-slate-400">Inactive profiles remain visible in history but should not be matched going forward.</span>
              </span>
            </label>

            {profileError ? (
              <div className="rounded-md border border-danger/50 bg-danger/10 px-3 py-2 text-sm text-red-200 lg:col-span-2">
                {profileError}
              </div>
            ) : null}

            <div className="flex flex-wrap items-center gap-3 lg:col-span-2">
              <Button type="submit" disabled={updateMutation.isPending}>
                {updateMutation.isPending ? "Saving profile" : "Save profile"}
              </Button>
            </div>
          </form>
        </SectionCard>
      ) : null}

      {person && canManage ? (
        <SectionCard title="Danger zone" description="Delete this person if the profile and enrolled faces should be removed from the registry.">
          {deleteError ? (
            <div className="mb-4 rounded-md border border-danger/50 bg-danger/10 px-3 py-2 text-sm text-red-200">
              {deleteError}
            </div>
          ) : null}

          <div className="flex flex-wrap items-center gap-3">
            <Button
              type="button"
              variant="ghost"
              className="border-red-400/30 text-red-200 hover:bg-red-500/10"
              onClick={onDelete}
              disabled={deleteMutation.isPending}
            >
              <Trash2 size={16} aria-hidden="true" />
              {deleteMutation.isPending ? "Deleting person" : "Delete person"}
            </Button>
          </div>
        </SectionCard>
      ) : null}
    </div>
  );
}

function ProtectedFaceImage({
  accessToken,
  personId,
  faceId,
  alt
}: {
  accessToken: string | null;
  personId: string;
  faceId: string;
  alt: string;
}) {
  const [src, setSrc] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    let objectUrl: string | null = null;

    async function load() {
      if (!accessToken) {
        setSrc(null);
        return;
      }

      try {
        const blob = await fetchProtectedMedia(accessToken, `/api/v1/persons/${personId}/faces/${faceId}/image`);
        if (!active) {
          return;
        }
        objectUrl = URL.createObjectURL(blob);
        setSrc(objectUrl);
      } catch {
        if (active) {
          setSrc(null);
        }
      }
    }

    void load();

    return () => {
      active = false;
      if (objectUrl) {
        URL.revokeObjectURL(objectUrl);
      }
    };
  }, [accessToken, faceId, personId]);

  if (!src) {
    return null;
  }

  return (
    <Image
      src={src}
      alt={alt}
      width={96}
      height={96}
      unoptimized
      className="mb-3 h-24 w-24 rounded-2xl border border-white/10 object-cover"
    />
  );
}

function DetailTile({ label, value, tone }: { label: string; value: string; tone: string }) {
  return (
    <div className="rounded-[22px] border border-white/10 bg-black/15 p-4">
      <p className="text-xs uppercase tracking-[0.18em] text-slate-500">{label}</p>
      <p className={cn("mt-3 inline-flex rounded-full px-2.5 py-1 text-sm font-medium", tone)}>{value}</p>
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
      href="/dashboard/persons"
      className="inline-flex items-center gap-2 text-sm font-medium text-accent transition hover:text-emerald-300"
    >
      <ArrowLeft size={15} aria-hidden="true" />
      Back to persons
    </Link>
  );
}
