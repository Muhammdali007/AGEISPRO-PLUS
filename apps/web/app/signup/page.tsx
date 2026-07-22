"use client";

import Link from "next/link";
import { ShieldCheck, UserPlus } from "lucide-react";

export default function SignupPage() {
  return (
    <main className="flex min-h-screen items-center justify-center px-4 py-10">
      <section className="w-full max-w-md rounded-lg border border-border bg-panel p-6 shadow-2xl">
        <div className="mb-8 flex items-center gap-3">
          <div className="flex h-11 w-11 items-center justify-center rounded-md bg-accent text-slate-950">
            <ShieldCheck size={24} aria-hidden="true" />
          </div>
          <div>
            <h1 className="text-2xl font-semibold tracking-normal">Account access is administrator-managed</h1>
            <p className="text-sm text-slate-400">
              Public self-registration is disabled to protect privileged AegisPro roles from abuse.
            </p>
          </div>
        </div>

        <div className="space-y-4 rounded-lg border border-border bg-background/70 p-5 text-sm text-slate-300">
          <div className="flex items-start gap-3">
            <div className="mt-0.5 flex h-8 w-8 items-center justify-center rounded-md bg-accent text-slate-950">
              <UserPlus size={16} aria-hidden="true" />
            </div>
            <div className="space-y-2">
              <p>Ask an AegisPro administrator or supervisor to create your account for you.</p>
              <p>After they provision and activate your account, you can return here and sign in normally.</p>
            </div>
          </div>
        </div>

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
