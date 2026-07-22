"use client";

import Link from "next/link";
import { Eye, EyeOff, Lock, ShieldCheck } from "lucide-react";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { z } from "zod";
import { Button } from "@/components/button";
import { fetchCurrentUser, login } from "@/lib/api";
import { useAuthStore } from "@/lib/auth-store";

const loginSchema = z.object({
  email: z.string().email("Enter a valid email address."),
  password: z.string().min(8, "Password must be at least 8 characters.")
});

export default function LoginPage() {
  const router = useRouter();
  const { accessToken, hydrated, hydrate, setTokens, setUser } = useAuthStore();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [fieldErrors, setFieldErrors] = useState<Partial<Record<"email" | "password", string>>>({});
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

    const parsed = loginSchema.safeParse({ email, password });
    if (!parsed.success) {
      const nextErrors: Partial<Record<"email" | "password", string>> = {};
      for (const issue of parsed.error.issues) {
        const field = issue.path[0];
        if (field === "email" || field === "password") {
          nextErrors[field] = issue.message;
        }
      }
      setFieldErrors(nextErrors);
      return;
    }

    setIsSubmitting(true);

    try {
      const tokens = await login(parsed.data.email, parsed.data.password);
      setTokens(tokens);
      setUser(await fetchCurrentUser(tokens.access_token));

      router.replace("/dashboard");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Unable to sign in");
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
            <h1 className="text-2xl font-semibold tracking-normal">AegisPro</h1>
            <p className="text-sm text-slate-400">Secure surveillance operations</p>
          </div>
        </div>

        <form className="space-y-5" onSubmit={onSubmit}>
          <label className="block" htmlFor="email">
            <span className="mb-2 block text-sm text-slate-300">Email</span>
            <input
              id="email"
              className="h-11 w-full rounded-md border border-border bg-background px-3 text-sm outline-none transition focus:border-accent"
              type="email"
              autoComplete="email"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
            />
            {fieldErrors.email ? (
              <span className="mt-1 block text-sm text-danger">{fieldErrors.email}</span>
            ) : null}
          </label>

          <div className="block">
            <div className="mb-2 flex items-center justify-between gap-3 text-sm text-slate-300">
              <label htmlFor="password">Password</label>
              <Link href="/forgot-password" className="font-medium text-accent transition hover:text-emerald-300">
                Forgot password?
              </Link>
            </div>
            <div className="relative">
              <input
                id="password"
                className="h-11 w-full rounded-md border border-border bg-background px-3 pr-11 text-sm outline-none transition focus:border-accent"
                type={showPassword ? "text" : "password"}
                autoComplete="current-password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
              />
              <button
                type="button"
                className="absolute right-2 top-1/2 flex h-8 w-8 -translate-y-1/2 items-center justify-center rounded-md text-slate-400 transition hover:bg-panelSoft hover:text-slate-100 focus:outline-none focus:ring-2 focus:ring-accent"
                aria-label={showPassword ? "Hide characters" : "Show characters"}
                onClick={() => setShowPassword((current) => !current)}
              >
                {showPassword ? <EyeOff size={16} aria-hidden="true" /> : <Eye size={16} aria-hidden="true" />}
              </button>
            </div>
            {fieldErrors.password ? (
              <span className="mt-1 block text-sm text-danger">{fieldErrors.password}</span>
            ) : null}
          </div>

          {error ? (
            <div className="rounded-md border border-danger/50 bg-danger/10 px-3 py-2 text-sm text-red-200">
              {error}
            </div>
          ) : null}

          <Button className="w-full" type="submit" disabled={isSubmitting}>
            <Lock size={16} aria-hidden="true" />
            {isSubmitting ? "Signing in..." : "Sign in"}
          </Button>
        </form>

        <p className="mt-5 text-center text-sm text-slate-400">
          Need access?{" "}
          <Link href="/signup" className="font-medium text-accent transition hover:text-emerald-300">
            Contact your administrator
          </Link>
        </p>
      </section>
    </main>
  );
}
