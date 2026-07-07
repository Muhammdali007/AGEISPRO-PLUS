"use client";

import Link from "next/link";
import { ShieldCheck, UserPlus } from "lucide-react";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { z } from "zod";
import { Button } from "@/components/button";
import { fetchCurrentUser, signup, type UserRole } from "@/lib/api";
import { useAuthStore } from "@/lib/auth-store";

const roles = ["administrator", "supervisor", "operator", "viewer"] as const satisfies readonly UserRole[];

const signupSchema = z
  .object({
    full_name: z.string().min(1, "Full name is required.").max(160, "Full name is too long."),
    email: z.string().email("Enter a valid email address."),
    role: z.enum(roles),
    password: z.string().min(8, "Password must be at least 8 characters.").max(128, "Password is too long."),
    confirmPassword: z.string().min(8, "Confirm your password.")
  })
  .refine((values) => values.password === values.confirmPassword, {
    message: "Passwords do not match.",
    path: ["confirmPassword"]
  });

type SignupField = "full_name" | "email" | "role" | "password" | "confirmPassword";

export default function SignupPage() {
  const router = useRouter();
  const { accessToken, hydrated, hydrate, setTokens, setUser } = useAuthStore();
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [role, setRole] = useState<UserRole>("viewer");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [fieldErrors, setFieldErrors] = useState<Partial<Record<SignupField, string>>>({});
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  useEffect(() => {
    hydrate();
  }, [hydrate]);

  useEffect(() => {
    if (hydrated && accessToken) {
      router.replace("/dashboard");
    }
  }, [accessToken, hydrated, router]);

  async function onSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    setFieldErrors({});

    const parsed = signupSchema.safeParse({
      full_name: fullName,
      email,
      role,
      password,
      confirmPassword
    });

    if (!parsed.success) {
      const nextErrors: Partial<Record<SignupField, string>> = {};
      for (const issue of parsed.error.issues) {
        const field = issue.path[0];
        if (
          field === "full_name" ||
          field === "email" ||
          field === "role" ||
          field === "password" ||
          field === "confirmPassword"
        ) {
          nextErrors[field] = issue.message;
        }
      }
      setFieldErrors(nextErrors);
      return;
    }

    setIsSubmitting(true);

    try {
      const tokens = await signup({
        full_name: parsed.data.full_name,
        email: parsed.data.email,
        role: parsed.data.role,
        password: parsed.data.password
      });
      setTokens(tokens);

      try {
        setUser(await fetchCurrentUser(tokens.access_token));
      } catch {}

      window.location.href = "/dashboard";
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Unable to create account");
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <main className="flex min-h-screen items-center justify-center px-4 py-10">
      <section className="w-full max-w-md rounded-lg border border-border bg-panel p-6 shadow-2xl">
        <div className="mb-8 flex items-center gap-3">
          <div className="flex h-11 w-11 items-center justify-center rounded-md bg-accent text-slate-950">
            <ShieldCheck size={24} aria-hidden="true" />
          </div>
          <div>
            <h1 className="text-2xl font-semibold tracking-normal">Create your AegisPro account</h1>
            <p className="text-sm text-slate-400">Describe your role during signup so the dashboard can grant the right access level.</p>
          </div>
        </div>

        <form className="space-y-5" onSubmit={onSubmit}>
          <label className="block">
            <span className="mb-2 block text-sm text-slate-300">Full name</span>
            <input
              className="h-11 w-full rounded-md border border-border bg-background px-3 text-sm outline-none transition focus:border-accent"
              type="text"
              autoComplete="name"
              value={fullName}
              onChange={(event) => setFullName(event.target.value)}
            />
            {fieldErrors.full_name ? (
              <span className="mt-1 block text-sm text-danger">{fieldErrors.full_name}</span>
            ) : null}
          </label>

          <label className="block">
            <span className="mb-2 block text-sm text-slate-300">Email</span>
            <input
              className="h-11 w-full rounded-md border border-border bg-background px-3 text-sm outline-none transition focus:border-accent"
              type="email"
              autoComplete="email"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
            />
            {fieldErrors.email ? <span className="mt-1 block text-sm text-danger">{fieldErrors.email}</span> : null}
          </label>

          <label className="block">
            <span className="mb-2 block text-sm text-slate-300">Role</span>
            <select
              className="h-11 w-full rounded-md border border-border bg-background px-3 text-sm outline-none transition focus:border-accent"
              value={role}
              onChange={(event) => setRole(event.target.value as UserRole)}
            >
              {roles.map((roleOption) => (
                <option key={roleOption} value={roleOption}>
                  {roleOption}
                </option>
              ))}
            </select>
            {fieldErrors.role ? <span className="mt-1 block text-sm text-danger">{fieldErrors.role}</span> : null}
          </label>

          <label className="block">
            <span className="mb-2 block text-sm text-slate-300">Password</span>
            <input
              className="h-11 w-full rounded-md border border-border bg-background px-3 text-sm outline-none transition focus:border-accent"
              type="password"
              autoComplete="new-password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
            />
            {fieldErrors.password ? (
              <span className="mt-1 block text-sm text-danger">{fieldErrors.password}</span>
            ) : null}
          </label>

          <label className="block">
            <span className="mb-2 block text-sm text-slate-300">Confirm password</span>
            <input
              className="h-11 w-full rounded-md border border-border bg-background px-3 text-sm outline-none transition focus:border-accent"
              type="password"
              autoComplete="new-password"
              value={confirmPassword}
              onChange={(event) => setConfirmPassword(event.target.value)}
            />
            {fieldErrors.confirmPassword ? (
              <span className="mt-1 block text-sm text-danger">{fieldErrors.confirmPassword}</span>
            ) : null}
          </label>

          {error ? (
            <div className="rounded-md border border-danger/50 bg-danger/10 px-3 py-2 text-sm text-red-200">
              {error}
            </div>
          ) : null}

          <Button className="w-full" type="submit" disabled={isSubmitting}>
            <UserPlus size={16} aria-hidden="true" />
            {isSubmitting ? "Creating account..." : "Sign up"}
          </Button>
        </form>

        <p className="mt-5 text-center text-sm text-slate-400">
          Already have an account?{" "}
          <Link href="/login" className="font-medium text-accent transition hover:text-emerald-300">
            Sign in
          </Link>
        </p>
      </section>
    </main>
  );
}
