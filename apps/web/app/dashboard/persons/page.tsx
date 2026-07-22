"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Trash2 } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Button } from "@/components/button";
import { EmptyState, InlineLink, MetricCard, SectionCard } from "@/components/dashboard-ui";
import { deletePerson, listPersons } from "@/lib/api";
import { useAuthStore } from "@/lib/auth-store";
import { formatDateTime, labelize, statusTone } from "@/lib/format";
import { cn } from "@/lib/cn";

export default function PersonsPage() {
  const router = useRouter();
  const queryClient = useQueryClient();
  const { accessToken, user, logout } = useAuthStore();
  const personsQuery = useQuery({
    queryKey: ["persons", "list", accessToken],
    queryFn: () => listPersons(accessToken!),
    enabled: Boolean(accessToken)
  });

  const persons = personsQuery.data ?? [];
  const canManagePersons = user?.role === "administrator" || user?.role === "supervisor";
  const deletePersonMutation = useMutation({
    mutationFn: async (personId: string) => {
      if (!accessToken) {
        throw new Error("You need to sign in again before deleting a person.");
      }

      await deletePerson(accessToken, personId);
    },
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["persons", "list", accessToken] });
    },
    onError: (cause) => {
      if (cause instanceof Error && (cause.message === "Invalid credentials" || cause.message === "Session expired")) {
        logout();
        router.push("/login");
      }
    }
  });

  async function handleDelete(personId: string, fullName: string) {
    const confirmed = window.confirm(`Delete ${fullName}? This will remove the person profile and enrolled faces.`);
    if (!confirmed) {
      return;
    }

    try {
      await deletePersonMutation.mutateAsync(personId);
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : "Unable to delete person";
      window.alert(message);
    }
  }

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
          canManagePersons ? (
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
              <div
                key={person.id}
                className="grid gap-4 rounded-[24px] border border-white/10 bg-black/15 p-4 transition hover:border-accent/30 hover:bg-black/25 xl:grid-cols-[1.15fr_0.85fr_0.8fr_0.8fr_1fr]"
              >
                <div>
                  <p className="font-medium">{person.full_name}</p>
                  <p className="mt-2 text-sm text-slate-400">
                    {labelize(person.person_type)}
                    {person.reference_id ? ` - ${person.reference_id}` : ""}
                    {person.department ? ` - ${person.department}` : ""}
                  </p>
                  <Link
                    href={`/dashboard/persons/${person.id}`}
                    className="mt-3 inline-flex text-sm font-medium text-accent transition hover:text-emerald-300"
                  >
                    Open profile
                  </Link>
                </div>
                <QueueCell label="Title" value={person.title ?? "Not set"} />
                <QueueCell label="Last seen" value={formatDateTime(person.last_seen_at)} />
                <QueueCell label="Recognition count" value={`${person.recognition_count}`} />
                <div className="flex flex-wrap items-center gap-2 xl:justify-end">
                  <span className={cn("rounded-full px-2.5 py-1 text-xs font-medium", statusTone(person.is_active ? "ok" : "disabled"))}>
                    {person.is_active ? "Active" : "Inactive"}
                  </span>
                  <span className="rounded-full bg-black/20 px-2.5 py-1 text-xs font-medium text-slate-200 ring-1 ring-white/10">
                    {labelize(`${person.face_image_count} face${person.face_image_count === 1 ? "" : "s"}`)}
                  </span>
                  {canManagePersons ? (
                    <>
                      <Link
                        href={`/dashboard/persons/${person.id}`}
                        className="inline-flex h-10 items-center justify-center rounded-md border border-border px-4 text-sm font-medium text-slate-200 transition hover:bg-panelSoft"
                      >
                        Edit
                      </Link>
                      <Button
                        type="button"
                        variant="ghost"
                        className="border-red-400/30 text-red-200 hover:bg-red-500/10"
                        disabled={deletePersonMutation.isPending}
                        onClick={() => handleDelete(person.id, person.full_name)}
                      >
                        <Trash2 size={16} aria-hidden="true" />
                        Delete
                      </Button>
                    </>
                  ) : null}
                </div>
              </div>
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
