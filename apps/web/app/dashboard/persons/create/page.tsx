"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ArrowLeft, UserPlus } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import type { ChangeEvent, ReactNode } from "react";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { Button } from "@/components/button";
import { EmptyState, SectionCard } from "@/components/dashboard-ui";
import { createPerson, uploadPersonFaceImages } from "@/lib/api";
import { useAuthStore } from "@/lib/auth-store";

const personTypes = ["employee", "student", "visitor", "contractor", "other"] as const;

const createPersonSchema = z.object({
  full_name: z.string().min(1, "Full name is required").max(160, "Full name is too long"),
  person_type: z.enum(personTypes),
  department: z.string().max(120, "Department is too long").optional().or(z.literal("")),
  reference_id: z.string().max(64, "Reference ID is too long").optional().or(z.literal("")),
  title: z.string().max(120, "Title is too long").optional().or(z.literal("")),
  is_active: z.boolean()
});

type CreatePersonForm = z.infer<typeof createPersonSchema>;

export default function CreatePersonPage() {
  const router = useRouter();
  const queryClient = useQueryClient();
  const { accessToken, user, logout } = useAuthStore();
  const [error, setError] = useState<string | null>(null);
  const [uploadFiles, setUploadFiles] = useState<File[]>([]);
  const [uploadPrimary, setUploadPrimary] = useState(true);
  const handleFileChange = handleFileChangeFactory(setUploadFiles);
  const form = useForm<CreatePersonForm>({
    resolver: zodResolver(createPersonSchema),
    defaultValues: {
      full_name: "",
      person_type: "visitor",
      department: "",
      reference_id: "",
      title: "",
      is_active: true
    }
  });

  const createPersonMutation = useMutation({
    mutationFn: async (values: CreatePersonForm) => {
      if (!accessToken) {
        throw new Error("You need to sign in again before creating a person.");
      }
      const person = await createPerson(accessToken, {
        full_name: values.full_name,
        person_type: values.person_type,
        department: values.department || null,
        reference_id: values.reference_id || null,
        title: values.title || null,
        is_active: values.is_active
      });
      if (uploadFiles.length > 0) {
        await uploadPersonFaceImages(accessToken, person.id, {
          files: uploadFiles,
          is_primary: uploadPrimary
        });
      }
      return person;
    },
    onSuccess: async (person) => {
      await queryClient.invalidateQueries({ queryKey: ["persons", "list", accessToken] });
      await queryClient.invalidateQueries({ queryKey: ["persons", "detail", person.id, accessToken] });
      router.push(`/dashboard/persons/${person.id}`);
    },
    onError: (cause) => {
      if (cause instanceof Error && (cause.message === "Invalid credentials" || cause.message === "Session expired")) {
        logout();
        router.push("/login");
        return;
      }
      setError(cause instanceof Error ? cause.message : "Unable to create person");
    }
  });

  function onSubmit(values: CreatePersonForm) {
    setError(null);
    createPersonMutation.mutate(values);
  }

  if (user?.role && user.role !== "administrator" && user.role !== "supervisor" && user.role !== "operator") {
    return (
      <SectionCard
        title="Create known person"
        description="Only administrators, supervisors, and operators can add known-person profiles."
        action={<BackLink />}
      >
        <EmptyState
          title="Elevated access required"
          description="Your current account can view known persons, but only operators, supervisors, and administrators can add or modify them."
        />
      </SectionCard>
    );
  }

  return (
    <div className="space-y-6">
      <SectionCard
        title="Create known person"
        description="Add a profile to the Phase 6 recognition registry."
        action={<BackLink />}
      >
        <form className="grid gap-5 lg:grid-cols-2" onSubmit={form.handleSubmit(onSubmit)}>
          <FormField label="Full name" error={form.formState.errors.full_name?.message}>
            <input
              className="h-11 w-full rounded-md border border-border bg-background px-3 text-sm outline-none transition focus:border-accent"
              type="text"
              autoComplete="name"
              {...form.register("full_name")}
            />
          </FormField>

          <FormField label="Person type" error={form.formState.errors.person_type?.message}>
            <select
              className="h-11 w-full rounded-md border border-border bg-background px-3 text-sm outline-none transition focus:border-accent"
              {...form.register("person_type")}
            >
              <option value="visitor">Visitor</option>
              <option value="student">Student</option>
              <option value="employee">Employee</option>
              <option value="contractor">Contractor</option>
              <option value="other">Other</option>
            </select>
          </FormField>

          <FormField label="Reference ID" error={form.formState.errors.reference_id?.message}>
            <input
              className="h-11 w-full rounded-md border border-border bg-background px-3 text-sm outline-none transition focus:border-accent"
              placeholder="Optional: employee ID, student ID, badge ID, visitor code"
              type="text"
              {...form.register("reference_id")}
            />
          </FormField>

          <FormField label="Department" error={form.formState.errors.department?.message}>
            <input
              className="h-11 w-full rounded-md border border-border bg-background px-3 text-sm outline-none transition focus:border-accent"
              type="text"
              {...form.register("department")}
            />
          </FormField>

          <FormField label="Title" error={form.formState.errors.title?.message}>
            <input
              className="h-11 w-full rounded-md border border-border bg-background px-3 text-sm outline-none transition focus:border-accent"
              type="text"
              {...form.register("title")}
            />
          </FormField>

          <FormField label="Upload photos">
            <input
              className="block w-full text-sm text-slate-300 file:mr-4 file:rounded-md file:border-0 file:bg-emerald-500/15 file:px-4 file:py-2 file:text-sm file:font-medium file:text-emerald-100"
              type="file"
              accept="image/*"
              multiple
              onChange={handleFileChange}
            />
            <span className="mt-2 block text-sm text-slate-400">
              Optional. If you attach photos here, the app will create the profile and immediately extract embeddings from each uploaded image.
            </span>
          </FormField>

          <label className="flex items-center gap-3 rounded-[20px] border border-white/10 bg-black/15 px-4 py-3">
            <input
              className="h-4 w-4 accent-emerald-400"
              type="checkbox"
              checked={uploadPrimary}
              onChange={(event) => setUploadPrimary(event.target.checked)}
            />
            <span>
              <span className="block text-sm font-medium text-slate-100">Mark first uploaded image as primary</span>
              <span className="block text-sm text-slate-400">Useful when you select multiple photos and want one preferred face record.</span>
            </span>
          </label>

          {uploadFiles.length > 0 ? (
            <div className="rounded-[20px] border border-white/10 bg-black/15 px-4 py-3 text-sm text-slate-300 lg:col-span-2">
              {uploadFiles.map((file) => file.name).join(", ")}
            </div>
          ) : null}

          <label className="flex items-center gap-3 rounded-[20px] border border-white/10 bg-black/15 px-4 py-3 lg:col-span-2">
            <input className="h-4 w-4 accent-emerald-400" type="checkbox" {...form.register("is_active")} />
            <span>
              <span className="block text-sm font-medium text-slate-100">Profile active</span>
              <span className="block text-sm text-slate-400">Active profiles are eligible for recognition matching.</span>
            </span>
          </label>

          {error ? (
            <div className="rounded-md border border-danger/50 bg-danger/10 px-3 py-2 text-sm text-red-200 lg:col-span-2">
              {error}
            </div>
          ) : null}

          <div className="flex flex-wrap items-center gap-3 lg:col-span-2">
            <Button type="submit" disabled={createPersonMutation.isPending}>
              <UserPlus size={16} aria-hidden="true" />
              {createPersonMutation.isPending ? "Creating profile" : "Create profile"}
            </Button>
            <Button type="button" variant="ghost" onClick={() => router.push("/dashboard/persons")}>
              Cancel
            </Button>
          </div>
        </form>
      </SectionCard>
    </div>
  );
}

function handleFileChangeFactory(setUploadFiles: (files: File[]) => void) {
  return (event: ChangeEvent<HTMLInputElement>) => {
    setUploadFiles(Array.from(event.target.files ?? []));
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
