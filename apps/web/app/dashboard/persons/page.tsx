"use client";

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { EmptyState, InlineLink, MetricCard, SectionCard } from "@/components/dashboard-ui";
import { listPersons } from "@/lib/api";
import { useAuthStore } from "@/lib/auth-store";
import { formatDateTime, labelize, statusTone } from "@/lib/format";
import { cn } from "@/lib/cn";

export default function PersonsPage() {
  const { accessToken, user } = useAuthStore();
  const personsQuery = useQuery({
    queryKey: ["persons", "list", accessToken],
    queryFn: () => listPersons(accessToken!),
    enabled: Boolean(accessToken)
  });

  const persons = personsQuery.data ?? [];

  return (
    <div className="space-y-6">
      <section className="grid gap-4 md:grid-cols-4">
        <MetricCard label="Known persons" value={`${persons.length}`} detail="Profiles available to the recognition workflow." />
        <MetricCard label="Active profiles" value={`${persons.filter((person) => person.is_active).length}`} detail="People currently eligible for recognition matching." tone="success" />
        <MetricCard label="Enrolled faces" value={`${persons.reduce((total, person) => total + person.face_image_count, 0)}`} detail="Face references stored across all profiles." />
        <MetricCard label="Recognition count" value={`${persons.reduce((total, person) => total + person.recognition_count, 0)}`} detail="Phase 6 sightings recorded through the incident pipeline." />
      </section>

      <SectionCard
        title="Known person registry"
        description="Manage the source-of-truth profiles that the recognition pipeline can match against."
        action={
          user?.role === "administrator" || user?.role === "supervisor" || user?.role === "operator" ? (
            <InlineLink href="/dashboard/persons/create" label="Add person" />
          ) : undefined
        }
      >
        {persons.length === 0 ? (
          <EmptyState
            title="No known persons enrolled"
            description="Add the first person profile to start capturing face metadata and recognition history."
          />
        ) : (
          <div className="space-y-3">
            {persons.map((person) => (
              <Link
                key={person.id}
                href={`/dashboard/persons/${person.id}`}
                className="grid gap-4 rounded-[24px] border border-white/10 bg-black/15 p-4 transition hover:border-accent/30 hover:bg-black/25 xl:grid-cols-[1.15fr_0.85fr_0.8fr_0.8fr_0.8fr]"
              >
                <div>
                  <p className="font-medium">{person.full_name}</p>
                  <p className="mt-2 text-sm text-slate-400">
                    {labelize(person.person_type)}
                    {person.reference_id ? ` - ${person.reference_id}` : ""}
                    {person.department ? ` - ${person.department}` : ""}
                  </p>
                </div>
                <QueueCell label="Title" value={person.title ?? "Not set"} />
                <QueueCell label="Last seen" value={formatDateTime(person.last_seen_at)} />
                <QueueCell label="Recognition count" value={`${person.recognition_count}`} />
                <div className="flex flex-wrap items-center gap-2">
                  <span className={cn("rounded-full px-2.5 py-1 text-xs font-medium", statusTone(person.is_active ? "ok" : "disabled"))}>
                    {person.is_active ? "Active" : "Inactive"}
                  </span>
                  <span className="rounded-full bg-black/20 px-2.5 py-1 text-xs font-medium text-slate-200 ring-1 ring-white/10">
                    {labelize(`${person.face_image_count} face${person.face_image_count === 1 ? "" : "s"}`)}
                  </span>
                </div>
              </Link>
            ))}
          </div>
        )}
      </SectionCard>
    </div>
  );
}

function QueueCell({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-xs uppercase tracking-[0.18em] text-slate-500">{label}</p>
      <p className="mt-2 text-sm text-slate-200">{value}</p>
    </div>
  );
}
