"use client";

import Link from "next/link";
import { KeyRound, ShieldCheck } from "lucide-react";
import { useState } from "react";
import { z } from "zod";
import { Button } from "@/components/button";
import { resetPassword } from "@/lib/api";

const resetPasswordSchema = z
  .object({
    password: z.string().min(8, "Password must be at least 8 characters.").max(128, "Password is too long."),
    confirmPassword: z.string().min(8, "Confirm the password.")
  })
  .refine((values) => values.password === values.confirmPassword, {
    message: "Passwords do not match.",
    path: ["confirmPassword"]
  });

type FieldErrors = Partial<Record<"password" | "confirmPassword", string>>;

export default function ResetPasswordPage() {
  const [token] = useState(() =>
    typeof window === "undefined" ? "" : (new URLSearchParams(window.location.search).get("token") ?? "")
  );
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [fieldErrors, setFieldErrors] = useState<FieldErrors>({});
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function onSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setFieldErrors({});
    setMessage(null);
    setError(null);

    if (!token) {
      setError("Password reset link is missing or invalid.");
      return;
    }

    const parsed = resetPasswordSchema.safeParse({ password, confirmPassword });
    if (!parsed.success) {
      const nextErrors: FieldErrors = {};
      for (const issue of parsed.error.issues) {
        const field = issue.path[0];
        if (field === "password" || field === "confirmPassword") {
          nextErrors[field] = issue.message;
        }
      }
      setFieldErrors(nextErrors);
      return;
    }

    setIsSubmitting(true);
    try {
      const response = await resetPassword(token, parsed.data.password);
      setPassword("");
      setConfirmPassword("");
      setMessage(response.detail);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Unable to reset password");
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
            <h1 className="text-2xl font-semibold tracking-normal">Choose a new password</h1>
            <p className="text-sm text-slate-400">Reset links work once and expire quickly.</p>
          </div>
        </div>

        <form className="space-y-5" onSubmit={onSubmit}>
          <label className="block">
            <span className="mb-2 block text-sm text-slate-300">New password</span>
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
            <span className="mb-2 block text-sm text-slate-300">Confirm new password</span>
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

          {message ? (
            <div className="rounded-md border border-accent/50 bg-accent/10 px-3 py-2 text-sm text-emerald-100">
              {message}
            </div>
          ) : null}
          {error ? (
            <div className="rounded-md border border-danger/50 bg-danger/10 px-3 py-2 text-sm text-red-200">
              {error}
            </div>
          ) : null}

          <Button className="w-full" type="submit" disabled={isSubmitting}>
            <KeyRound size={16} aria-hidden="true" />
            {isSubmitting ? "Resetting..." : "Reset password"}
          </Button>
        </form>

        <p className="mt-5 text-center text-sm text-slate-400">
          Ready to continue?{" "}
          <Link href="/login" className="font-medium text-accent transition hover:text-emerald-300">
            Sign in
          </Link>
        </p>
      </section>
    </main>
  );
}
