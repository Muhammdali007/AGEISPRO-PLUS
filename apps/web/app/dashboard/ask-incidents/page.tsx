"use client";

import { useMutation, useQuery } from "@tanstack/react-query";
import { Clock3, ExternalLink, Search, ShieldCheck } from "lucide-react";
import Link from "next/link";
import type { ReactNode } from "react";
import { useState } from "react";
import { Button } from "@/components/button";
import { EmptyState, MetricCard, SectionCard } from "@/components/dashboard-ui";
import {
  getVideoRagStatus,
  listCameras,
  queryVideoRag,
  type VideoRagQueryInput
} from "@/lib/api";
import { useAuthStore } from "@/lib/auth-store";
import { cn } from "@/lib/cn";
import { formatDateTime, formatPercent, labelize, statusTone } from "@/lib/format";

export default function AskIncidentsPage() {
  const { accessToken } = useAuthStore();
  const [question, setQuestion] = useState("");
  const [cameraId, setCameraId] = useState("");
  const [startAt, setStartAt] = useState("");
  const [endAt, setEndAt] = useState("");

  const camerasQuery = useQuery({
    queryKey: ["cameras", "list", accessToken],
    queryFn: () => listCameras(accessToken!),
    enabled: Boolean(accessToken)
  });
  const statusQuery = useQuery({
    queryKey: ["video-rag", "status", accessToken],
    queryFn: () => getVideoRagStatus(accessToken!),
    enabled: Boolean(accessToken),
    refetchInterval: 15000
  });
  const queryMutation = useMutation({
    mutationFn: (payload: VideoRagQueryInput) => queryVideoRag(accessToken!, payload)
  });

  const status = statusQuery.data;
  const result = queryMutation.data;

  if (!accessToken) {
    return <EmptyState title="Incident search unavailable" description="Sign in again to query incident evidence." />;
  }

  return (
    <div className="space-y-6">
      <section className="grid gap-4 md:grid-cols-4">
        <MetricCard label="Indexed" value={`${status?.ready ?? 0}`} detail="Retained incidents ready for semantic search." tone="success" />
        <MetricCard label="Queued" value={`${status?.queued ?? 0}`} detail="Waiting for local visual indexing." />
        <MetricCard label="Processing" value={`${status?.processing ?? 0}`} detail="Currently being analyzed by the local model." />
        <MetricCard label="Failed" value={`${status?.failed ?? 0}`} detail="Will retry automatically within the configured limit." tone={status?.failed ? "alert" : "default"} />
      </section>

      <SectionCard
        title="Ask about incidents"
        description="Search retained detector-created clips, snapshots, camera context, identities, and operator notes."
      >
        {status && !status.enabled ? (
          <div className="mb-4 rounded-2xl border border-amber-400/30 bg-amber-400/10 p-4 text-sm text-amber-100">
            Incident Video RAG is disabled. Set VIDEO_RAG_ENABLED=true and start the RAG worker.
          </div>
        ) : null}
        <form
          className="space-y-4"
          onSubmit={(event) => {
            event.preventDefault();
            const trimmed = question.trim();
            if (!trimmed) return;
            queryMutation.mutate({
              question: trimmed,
              camera_ids: cameraId ? [cameraId] : undefined,
              start_at: startAt ? new Date(startAt).toISOString() : undefined,
              end_at: endAt ? new Date(endAt).toISOString() : undefined,
              limit: 5
            });
          }}
        >
          <label className="block">
            <span className="text-sm font-medium text-slate-200">Question</span>
            <textarea
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              rows={3}
              maxLength={1000}
              placeholder="What happened at Gate 1 last night?"
              className="mt-2 w-full resize-y rounded-2xl border border-white/10 bg-black/20 px-4 py-3 text-slate-100 outline-none transition placeholder:text-slate-600 focus:border-accent/60 focus:ring-2 focus:ring-accent/20"
            />
          </label>
          <div className="grid gap-4 md:grid-cols-3">
            <FilterField label="Camera">
              <select value={cameraId} onChange={(event) => setCameraId(event.target.value)} className="mt-2 h-11 w-full rounded-xl border border-white/10 bg-slate-950 px-3 text-sm text-slate-200 outline-none focus:border-accent/60">
                <option value="">All cameras</option>
                {(camerasQuery.data ?? []).map((camera) => <option key={camera.id} value={camera.id}>{camera.name}</option>)}
              </select>
            </FilterField>
            <FilterField label="From">
              <input type="datetime-local" value={startAt} onChange={(event) => setStartAt(event.target.value)} className="mt-2 h-11 w-full rounded-xl border border-white/10 bg-slate-950 px-3 text-sm text-slate-200 outline-none focus:border-accent/60" />
            </FilterField>
            <FilterField label="To">
              <input type="datetime-local" value={endAt} onChange={(event) => setEndAt(event.target.value)} className="mt-2 h-11 w-full rounded-xl border border-white/10 bg-slate-950 px-3 text-sm text-slate-200 outline-none focus:border-accent/60" />
            </FilterField>
          </div>
          <Button type="submit" disabled={!question.trim() || queryMutation.isPending || status?.enabled === false}>
            <Search size={16} aria-hidden="true" />
            {queryMutation.isPending ? "Reviewing evidence…" : "Ask incidents"}
          </Button>
        </form>
      </SectionCard>

      {queryMutation.error instanceof Error ? (
        <EmptyState title="Unable to answer" description={queryMutation.error.message} />
      ) : null}

      {result ? (
        <SectionCard title="Grounded answer" description="Generated locally from the matching retained evidence below.">
          <div className="rounded-2xl border border-accent/20 bg-accent/5 p-5">
            <p className="whitespace-pre-wrap leading-7 text-slate-100">{result.answer}</p>
          </div>
          {result.warnings.map((warning) => (
            <p key={warning} className="mt-3 text-sm text-amber-200">{warning}</p>
          ))}
          {result.evidence.length ? (
            <div className="mt-5 grid gap-4 lg:grid-cols-2">
              {result.evidence.map((item) => (
                <article key={item.incident_id} className="rounded-3xl border border-white/10 bg-black/15 p-5">
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div>
                      <p className="font-semibold">{item.camera_name}</p>
                      <p className="mt-1 flex items-center gap-2 text-sm text-slate-400"><Clock3 size={14} />{formatDateTime(item.occurred_at)}</p>
                    </div>
                    <span className={cn("rounded-full px-2.5 py-1 text-xs font-medium", statusTone(item.detection_type))}>{labelize(item.detection_type)}</span>
                  </div>
                  <p className="mt-4 line-clamp-4 text-sm leading-6 text-slate-300">{item.matched_excerpt}</p>
                  <div className="mt-4 flex flex-wrap gap-3 text-xs text-slate-400">
                    <span>Detector confidence {formatPercent(item.confidence)}</span>
                    <span>Relevance {formatPercent(item.relevance_score)}</span>
                    {item.clip_start_seconds !== null ? <span>Clip offset {item.clip_start_seconds.toFixed(1)}s</span> : null}
                  </div>
                  <Link href={`/dashboard/incidents/${item.incident_id}`} className="mt-5 inline-flex items-center gap-2 text-sm font-medium text-accent hover:text-emerald-300">
                    Review protected evidence <ExternalLink size={14} aria-hidden="true" />
                  </Link>
                </article>
              ))}
            </div>
          ) : (
            <div className="mt-5"><EmptyState title="No matching evidence" description="Try a camera or time filter, or wait for queued incidents to finish indexing." /></div>
          )}
        </SectionCard>
      ) : null}

      <div className="flex gap-3 rounded-2xl border border-white/10 bg-black/15 p-4 text-sm text-slate-400">
        <ShieldCheck className="shrink-0 text-accent" size={20} aria-hidden="true" />
        <p>AI visual summaries are model-generated observations, not forensic conclusions. Verify important findings against the original incident evidence.</p>
      </div>
    </div>
  );
}

function FilterField({ label, children }: { label: string; children: ReactNode }) {
  return <label className="block"><span className="text-xs uppercase tracking-[0.16em] text-slate-500">{label}</span>{children}</label>;
}
