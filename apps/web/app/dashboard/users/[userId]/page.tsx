"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowLeft, Eye, EyeOff, KeyRound, Trash2 } from "lucide-react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import type { ReactNode } from "react";
import type { UseFormRegisterReturn } from "react-hook-form";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { Button } from "@/components/button";
import { EmptyState, SectionCard } from "@/components/dashboard-ui";
import { deleteUser, getUser, updateUser, type UserRole } from "@/lib/api";
import { useAuthStore } from "@/lib/auth-store";

const roles = ["administrator", "supervisor", "operator", "viewer"] as const satisfies readonly UserRole[];

const updateUserSchema = z
  .object({
    full_name: z.string().min(1, "Full name is required").max(160, "Full name is too long"),
    email: z.string().email("Enter a valid email"),
    role: z.enum(roles),
    password: z.string().max(128, "Password is too long").optional().or(z.literal("")),
    confirmPassword: z.string().max(128, "Confirm the password").optional().or(z.literal("")),
    is_active: z.boolean()
  })
  .superRefine((values, context) => {
    const passwordProvided = Boolean(values.password);
    const confirmProvided = Boolean(values.confirmPassword);

    if ((passwordProvided || confirmProvided) && (values.password?.length ?? 0) < 8) {
      context.addIssue({
        code: z.ZodIssueCode.custom,
        message: "Password must be at least 8 characters",
        path: ["password"]
      });
    }

    if ((passwordProvided || confirmProvided) && (values.confirmPassword?.length ?? 0) < 8) {
      context.addIssue({
        code: z.ZodIssueCode.custom,
        message: "Confirm the password",
        path: ["confirmPassword"]
      });
    }

    if ((passwordProvided || confirmProvided) && values.password !== values.confirmPassword) {
      context.addIssue({
        code: z.ZodIssueCode.custom,
        message: "Passwords do not match",
        path: ["confirmPassword"]
      });
    }
  });

type UpdateUserForm = z.infer<typeof updateUserSchema>;

export default function EditUserPage() {
  const params = useParams<{ userId: string }>();
  const router = useRouter();
  const queryClient = useQueryClient();
  const { accessToken, user, logout } = useAuthStore();
  const [error, setError] = useState<string | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const userQuery = useQuery({
    queryKey: ["users", "detail", params.userId, accessToken],
    queryFn: () => getUser(accessToken!, params.userId),
    enabled: Boolean(accessToken && params.userId),
    retry: false
  });
  const availableRoles = user?.role === "administrator" ? roles : roles.filter((role) => role !== "administrator");
  const managedUser = userQuery.data;
  const blockedByRole = user?.role === "supervisor" && managedUser?.role === "administrator";

  const form = useForm<UpdateUserForm>({
    resolver: zodResolver(updateUserSchema),
    defaultValues: {
      full_name: "",
      email: "",
      role: "viewer",
      password: "",
      confirmPassword: "",
      is_active: true
    }
  });

  useEffect(() => {
    if (!managedUser) {
      return;
    }

    form.reset({
      full_name: managedUser.full_name,
      email: managedUser.email,
      role: managedUser.role,
      password: "",
      confirmPassword: "",
      is_active: managedUser.is_active
    });
  }, [form, managedUser]);

  const updateMutation = useMutation({
    mutationFn: async (values: UpdateUserForm) => {
      if (!accessToken) {
        throw new Error("You need to sign in again before editing this user.");
      }

      return updateUser(accessToken, params.userId, {
        full_name: values.full_name,
        email: values.email,
        role: values.role,
        password: values.password || undefined,
        is_active: values.is_active
      });
    },
    onSuccess: async (updatedUser) => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["users", "detail", params.userId, accessToken] }),
        queryClient.invalidateQueries({ queryKey: ["users", "list", accessToken] }),
        queryClient.invalidateQueries({ queryKey: ["auth", "me", accessToken] })
      ]);

      if (updatedUser.id === user?.id && !updatedUser.is_active) {
        logout();
        router.push("/login");
        return;
      }

      if (updatedUser.id === user?.id && !["administrator", "supervisor"].includes(updatedUser.role)) {
        router.push("/dashboard");
        return;
      }

      router.push("/dashboard/users");
    },
    onError: (cause) => {
      if (cause instanceof Error && (cause.message === "Invalid credentials" || cause.message === "Session expired")) {
        logout();
        router.push("/login");
        return;
      }
      setError(cause instanceof Error ? cause.message : "Unable to update user");
    }
  });

  const deleteMutation = useMutation({
    mutationFn: async () => {
      if (!accessToken) {
        throw new Error("You need to sign in again before deleting this user.");
      }

      await deleteUser(accessToken, params.userId);
    },
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["users", "list", accessToken] }),
        queryClient.invalidateQueries({ queryKey: ["auth", "me", accessToken] })
      ]);

      if (managedUser?.id === user?.id) {
        logout();
        router.push("/login");
        return;
      }

      router.push("/dashboard/users");
    },
    onError: (cause) => {
      if (cause instanceof Error && (cause.message === "Invalid credentials" || cause.message === "Session expired")) {
        logout();
        router.push("/login");
        return;
      }
      setDeleteError(cause instanceof Error ? cause.message : "Unable to delete user");
    }
  });

  async function onSubmit(values: UpdateUserForm) {
    setError(null);
    await updateMutation.mutateAsync(values);
  }

  async function onDelete() {
    if (!managedUser) {
      return;
    }
    setDeleteError(null);
    const confirmed = window.confirm(`Delete ${managedUser.full_name}? This will permanently remove the account.`);
    if (!confirmed) {
      return;
    }
    await deleteMutation.mutateAsync();
  }

  if (user?.role && user.role !== "administrator" && user.role !== "supervisor") {
    return (
      <SectionCard
        title="Edit user"
        description="Only administrators and supervisors can manage user accounts."
        action={<BackLink />}
      >
        <EmptyState
          title="Elevated access required"
          description="Your current account can sign in to the dashboard, but only administrators and supervisors can manage users."
        />
      </SectionCard>
    );
  }

  return (
    <div className="space-y-6">
      <SectionCard
        title={managedUser ? `Edit ${managedUser.full_name}` : "Edit user"}
        description="Update identity, role, password, and access state for this account."
        action={<BackLink />}
      >
        {userQuery.error instanceof Error ? (
          <EmptyState title="User unavailable" description={userQuery.error.message} />
        ) : !managedUser ? (
          <EmptyState title="Loading user" description="Fetching the current account record from the API." />
        ) : blockedByRole ? (
          <EmptyState
            title="Administrator access required"
            description="Supervisors can manage users, but administrator accounts remain reserved for administrators."
          />
        ) : (
          <form className="grid gap-5 lg:grid-cols-2" onSubmit={form.handleSubmit(onSubmit)}>
            <FormField label="Full name" error={form.formState.errors.full_name?.message}>
              <input
                className="h-11 w-full rounded-md border border-border bg-background px-3 text-sm outline-none transition focus:border-accent"
                type="text"
                autoComplete="name"
                {...form.register("full_name")}
              />
            </FormField>

            <FormField label="Email" error={form.formState.errors.email?.message}>
              <input
                className="h-11 w-full rounded-md border border-border bg-background px-3 text-sm outline-none transition focus:border-accent"
                type="email"
                autoComplete="email"
                {...form.register("email")}
              />
            </FormField>

            <FormField label="Role" error={form.formState.errors.role?.message}>
              <select
                className="h-11 w-full rounded-md border border-border bg-background px-3 text-sm outline-none transition focus:border-accent"
                {...form.register("role")}
              >
                {availableRoles.map((role) => (
                  <option key={role} value={role}>
                    {role}
                  </option>
                ))}
              </select>
            </FormField>

            <label className="flex items-center gap-3 rounded-[20px] border border-white/10 bg-black/15 px-4 py-3">
              <input className="h-4 w-4 accent-emerald-400" type="checkbox" {...form.register("is_active")} />
              <span>
                <span className="block text-sm font-medium text-slate-100">Active account</span>
                <span className="block text-sm text-slate-400">Inactive users stay in the audit trail but cannot sign in.</span>
              </span>
            </label>

            <div className="border-t border-white/10 pt-5 lg:col-span-2">
              <div className="mb-5 flex items-start gap-3">
                <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-accent text-slate-950">
                  <KeyRound size={18} aria-hidden="true" />
                </div>
                <div>
                  <h2 className="text-base font-semibold text-slate-100">Change password</h2>
                  <p className="mt-1 text-sm text-slate-400">
                    Enter and confirm a new password, or leave both fields blank to keep the current password.
                  </p>
                </div>
              </div>

              <div className="grid gap-5 lg:grid-cols-2">
                <FormField label="New password" error={form.formState.errors.password?.message}>
                  <PasswordInput
                    autoComplete="new-password"
                    isVisible={showPassword}
                    onToggle={() => setShowPassword((current) => !current)}
                    toggleLabel={showPassword ? "Hide new password" : "Show new password"}
                    registration={form.register("password")}
                  />
                </FormField>

                <FormField label="Confirm new password" error={form.formState.errors.confirmPassword?.message}>
                  <PasswordInput
                    autoComplete="new-password"
                    isVisible={showConfirmPassword}
                    onToggle={() => setShowConfirmPassword((current) => !current)}
                    toggleLabel={showConfirmPassword ? "Hide confirmed password" : "Show confirmed password"}
                    registration={form.register("confirmPassword")}
                  />
                </FormField>
              </div>
            </div>

            {error ? (
              <div className="rounded-md border border-danger/50 bg-danger/10 px-3 py-2 text-sm text-red-200 lg:col-span-2">
                {error}
              </div>
            ) : null}

            <div className="flex flex-wrap items-center gap-3 lg:col-span-2">
              <Button type="submit" disabled={updateMutation.isPending}>
                {updateMutation.isPending ? "Saving user" : "Save user"}
              </Button>
              <Button type="button" variant="ghost" onClick={() => router.push("/dashboard/users")}>
                Cancel
              </Button>
            </div>
          </form>
        )}
      </SectionCard>

      {managedUser && !blockedByRole ? (
        <SectionCard
          title="Danger zone"
          description="Delete this account if it should be removed from the platform."
        >
          {deleteError ? (
            <div className="mb-4 rounded-md border border-danger/50 bg-danger/10 px-3 py-2 text-sm text-red-200">
              {deleteError}
            </div>
          ) : null}

          <div className="flex flex-wrap items-center gap-3">
            <Button type="button" variant="ghost" className="border-red-400/30 text-red-200 hover:bg-red-500/10" onClick={onDelete} disabled={deleteMutation.isPending}>
              <Trash2 size={16} aria-hidden="true" />
              {deleteMutation.isPending ? "Deleting user" : "Delete user"}
            </Button>
          </div>
        </SectionCard>
      ) : null}
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

function PasswordInput({
  autoComplete,
  isVisible,
  onToggle,
  toggleLabel,
  registration
}: {
  autoComplete: string;
  isVisible: boolean;
  onToggle: () => void;
  toggleLabel: string;
  registration: UseFormRegisterReturn;
}) {
  return (
    <div className="relative">
      <input
        className="h-11 w-full rounded-md border border-border bg-background px-3 pr-11 text-sm outline-none transition focus:border-accent"
        type={isVisible ? "text" : "password"}
        autoComplete={autoComplete}
        {...registration}
      />
      <button
        type="button"
        className="absolute right-2 top-1/2 flex h-8 w-8 -translate-y-1/2 items-center justify-center rounded-md text-slate-400 transition hover:bg-panelSoft hover:text-slate-100 focus:outline-none focus:ring-2 focus:ring-accent"
        aria-label={toggleLabel}
        onClick={onToggle}
      >
        {isVisible ? <EyeOff size={16} aria-hidden="true" /> : <Eye size={16} aria-hidden="true" />}
      </button>
    </div>
  );
}

function BackLink() {
  return (
    <Link
      href="/dashboard/users"
      className="inline-flex items-center gap-2 text-sm font-medium text-accent transition hover:text-emerald-300"
    >
      <ArrowLeft size={15} aria-hidden="true" />
      Back to users
    </Link>
  );
}
