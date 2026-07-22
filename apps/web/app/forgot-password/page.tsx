"use client";

import Link from "next/link";
import { Mail, ShieldCheck } from "lucide-react";
import { useState } from "react";
import { z } from "zod";
import { Button } from "@/components/button";
import { requestPasswordReset } from "@/lib/api";

const forgotPasswordSchema = z.object({
  email: z.string().email("Enter a valid email address.")
});

export default function ForgotPasswordPage() {
  const [email, setEmail] = useState("");
  const [fieldError, setFieldError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function onSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setFieldError(null);
    setMessage(null);
    setError(null);

    const parsed = forgotPasswordSchema.safeParse({ email });
    if (!parsed.success) {
      setFieldError(parsed.error.issues[0]?.message ?? "Enter a valid email address.");
      return;
    }

    setIsSubmitting(true);
    try {
      const response = await requestPasswordReset(parsed.data.email);
      setMessage(response.detail);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Unable to request a password reset");
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
            <h1 className="text-2xl font-semibold tracking-normal">Reset administrator password</h1>
            <p className="text-sm text-slate-400">Password recovery is limited to active administrators.</p>
          </div>
        </div>

        <form className="space-y-5" onSubmit={onSubmit}>
          <label className="block">
            <span className="mb-2 block text-sm text-slate-300">Administrator email</span>
            <input
              className="h-11 w-full rounded-md border border-border bg-background px-3 text-sm outline-none transition focus:border-accent"
              type="email"
              autoComplete="email"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
            />
            {fieldError ? <span className="mt-1 block text-sm text-danger">{fieldError}</span> : null}
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
            <Mail size={16} aria-hidden="true" />
            {isSubmitting ? "Sending..." : "Send reset link"}
          </Button>
        </form>

        <p className="mt-5 text-center text-sm text-slate-400">
          Remembered it?{" "}
          <Link href="/login" className="font-medium text-accent transition hover:text-emerald-300">
            Sign in
          </Link>
        </p>
      </section>
    </main>
  );
}
