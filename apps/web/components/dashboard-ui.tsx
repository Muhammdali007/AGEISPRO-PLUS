import { AlertTriangle, ArrowRight } from "lucide-react";
import Link from "next/link";
import type { ReactNode } from "react";
import { cn } from "@/lib/cn";

export function SectionCard({
  title,
  description,
  action,
  children,
  className
}: {
  title: string;
  description?: string;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={cn("rounded-[28px] border border-white/10 bg-panel/75 p-5 shadow-xl", className)}>
      <div className="flex flex-col gap-3 border-b border-white/10 pb-4 md:flex-row md:items-end md:justify-between">
        <div>
          <h2 className="text-lg font-semibold">{title}</h2>
          {description ? <p className="mt-1 text-sm text-slate-400">{description}</p> : null}
        </div>
        {action}
      </div>
      <div className="pt-4">{children}</div>
    </section>
  );
}

export function MetricCard({
  label,
  value,
  detail,
  tone = "default"
}: {
  label: string;
  value: string;
  detail: string;
  tone?: "default" | "alert" | "success";
}) {
  return (
    <div
      className={cn(
        "rounded-[24px] border border-white/10 p-5",
        tone === "alert" && "bg-red-500/10",
        tone === "success" && "bg-emerald-500/10",
        tone === "default" && "bg-black/15"
      )}
    >
      <p className="text-sm text-slate-400">{label}</p>
      <p className="mt-3 text-3xl font-semibold">{value}</p>
      <p className="mt-3 text-sm text-slate-400">{detail}</p>
    </div>
  );
}

export function EmptyState({
  title,
  description
}: {
  title: string;
  description: string;
}) {
  return (
    <div className="rounded-[24px] border border-dashed border-white/15 bg-black/10 px-5 py-10 text-center">
      <AlertTriangle className="mx-auto text-slate-500" size={28} aria-hidden="true" />
      <h3 className="mt-4 text-lg font-semibold">{title}</h3>
      <p className="mx-auto mt-2 max-w-xl text-sm text-slate-400">{description}</p>
    </div>
  );
}

export function InlineLink({ href, label }: { href: string; label: string }) {
  return (
    <Link
      href={href}
      className="inline-flex items-center gap-2 text-sm font-medium text-accent transition hover:text-emerald-300"
    >
      {label}
      <ArrowRight size={15} aria-hidden="true" />
    </Link>
  );
}
