"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { ArrowLeft, UserPlus } from "lucide-react";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { Button } from "@/components/button";
import { EmptyState, SectionCard } from "@/components/dashboard-ui";
import { createUser, type UserRole } from "@/lib/api";
import { useAuthStore } from "@/lib/auth-store";

const roles = ["administrator", "supervisor", "operator", "viewer"] as const satisfies readonly UserRole[];

const createUserSchema = z
  .object({
    full_name: z.string().min(1, "Full name is required").max(160, "Full name is too long"),
    email: z.string().email("Enter a valid email"),
    role: z.enum(roles),
    password: z.string().min(8, "Password must be at least 8 characters").max(128, "Password is too long"),
    confirmPassword: z.string().min(8, "Confirm the password"),
    is_active: z.boolean()
  })
  .refine((values) => values.password === values.confirmPassword, {
    message: "Passwords do not match",
    path: ["confirmPassword"]
  });

type CreateUserForm = z.infer<typeof createUserSchema>;

export default function CreateUserPage() {
  const router = useRouter();
  const queryClient = useQueryClient();
  const { accessToken, user, logout } = useAuthStore();
  const [error, setError] = useState<string | null>(null);
  const availableRoles = user?.role === "administrator" ? roles : roles.filter((role) => role !== "administrator");
  const form = useForm<CreateUserForm>({
    resolver: zodResolver(createUserSchema),
    defaultValues: {
      full_name: "",
      email: "",
      role: "viewer",
      password: "",
      confirmPassword: "",
      is_active: true
    }
  });

  const createUserMutation = useMutation({
    mutationFn: async (values: CreateUserForm) => {
      if (!accessToken) {
        throw new Error("You need to sign in again before creating a user.");
      }

      return createUser(accessToken, {
        email: values.email,
        full_name: values.full_name,
        role: values.role,
        password: values.password,
        is_active: values.is_active
      });
    },
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["users", "list", accessToken] });
      router.push("/dashboard/users");
    },
    onError: (cause) => {
      if (cause instanceof Error && (cause.message === "Invalid credentials" || cause.message === "Session expired")) {
        logout();
        router.push("/login");
        return;
      }
      setError(cause instanceof Error ? cause.message : "Unable to create user");
    }
  });

  async function onSubmit(values: CreateUserForm) {
    setError(null);
    await createUserMutation.mutateAsync(values);
  }

  if (user?.role && user.role !== "administrator" && user.role !== "supervisor") {
    return (
      <SectionCard
        title="Create user"
        description="Only administrators and supervisors can create new accounts through the current backend contract."
        action={<BackLink />}
      >
        <EmptyState
          title="Elevated access required"
          description="Your current account can view users, but only an administrator or supervisor can create a new one."
        />
      </SectionCard>
    );
  }

  return (
    <div className="space-y-6">
      <SectionCard
        title="Create user"
        description="Add a new account using the RBAC-aware backend endpoint."
        action={<BackLink />}
      >
        <form className="grid gap-5 lg:grid-cols-2" onSubmit={form.handleSubmit(onSubmit)}>
          <label className="block">
            <span className="mb-2 block text-sm text-slate-300">Full name</span>
            <input
              className="h-11 w-full rounded-md border border-border bg-background px-3 text-sm outline-none transition focus:border-accent"
              type="text"
              autoComplete="name"
              {...form.register("full_name")}
            />
            {form.formState.errors.full_name ? (
              <span className="mt-1 block text-sm text-danger">{form.formState.errors.full_name.message}</span>
            ) : null}
          </label>

          <label className="block">
            <span className="mb-2 block text-sm text-slate-300">Email</span>
            <input
              className="h-11 w-full rounded-md border border-border bg-background px-3 text-sm outline-none transition focus:border-accent"
              type="email"
              autoComplete="email"
              {...form.register("email")}
            />
            {form.formState.errors.email ? (
              <span className="mt-1 block text-sm text-danger">{form.formState.errors.email.message}</span>
            ) : null}
          </label>

          <label className="block">
            <span className="mb-2 block text-sm text-slate-300">Role</span>
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
            {form.formState.errors.role ? (
              <span className="mt-1 block text-sm text-danger">{form.formState.errors.role.message}</span>
            ) : null}
          </label>

          <label className="flex items-center gap-3 rounded-[20px] border border-white/10 bg-black/15 px-4 py-3">
            <input className="h-4 w-4 accent-emerald-400" type="checkbox" {...form.register("is_active")} />
            <span>
              <span className="block text-sm font-medium text-slate-100">Active account</span>
              <span className="block text-sm text-slate-400">The new user can sign in immediately after creation.</span>
            </span>
          </label>

          <label className="block">
            <span className="mb-2 block text-sm text-slate-300">Password</span>
            <input
              className="h-11 w-full rounded-md border border-border bg-background px-3 text-sm outline-none transition focus:border-accent"
              type="password"
              autoComplete="new-password"
              {...form.register("password")}
            />
            {form.formState.errors.password ? (
              <span className="mt-1 block text-sm text-danger">{form.formState.errors.password.message}</span>
            ) : null}
          </label>

          <label className="block">
            <span className="mb-2 block text-sm text-slate-300">Confirm password</span>
            <input
              className="h-11 w-full rounded-md border border-border bg-background px-3 text-sm outline-none transition focus:border-accent"
              type="password"
              autoComplete="new-password"
              {...form.register("confirmPassword")}
            />
            {form.formState.errors.confirmPassword ? (
              <span className="mt-1 block text-sm text-danger">{form.formState.errors.confirmPassword.message}</span>
            ) : null}
          </label>

          {error ? (
            <div className="rounded-md border border-danger/50 bg-danger/10 px-3 py-2 text-sm text-red-200 lg:col-span-2">
              {error}
            </div>
          ) : null}

          <div className="flex flex-wrap items-center gap-3 lg:col-span-2">
            <Button type="submit" disabled={createUserMutation.isPending}>
              <UserPlus size={16} aria-hidden="true" />
              {createUserMutation.isPending ? "Creating user" : "Create user"}
            </Button>
            <Button type="button" variant="ghost" onClick={() => router.push("/dashboard/users")}>
              Cancel
            </Button>
          </div>
        </form>
      </SectionCard>
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
