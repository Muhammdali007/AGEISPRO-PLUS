"use client";

import { useDeferredValue, useMemo, useState, useTransition } from "react";
import type { ReactNode } from "react";
import { Activity, Gauge, Radio, RefreshCw } from "lucide-react";
import Link from "next/link";
import { Button } from "@/components/button";
import { CameraStreamPanel } from "@/components/camera-stream-panel";
import { EmptyState } from "@/components/dashboard-ui";
import type { CameraLiveMonitorEntry, CameraLiveMonitorSummary, CameraStatus } from "@/lib/api";
import { cn } from "@/lib/cn";
import { formatDateTime, labelize, statusTone } from "@/lib/format";

const healthFilters: Array<{ label: string; value: CameraStatus | "all" }> = [
  { label: "All feeds", value: "all" },
  { label: "Online", value: "online" },
  { label: "Degraded", value: "degraded" },
  { label: "Offline", value: "offline" },
  { label: "Disabled", value: "disabled" },
  { label: "Unknown", value: "unknown" }
];

export function LiveMonitorGrid({
  accessToken,
  entries,
  summary,
  onRefreshHealth,
  refreshing
}: {
  accessToken: string;
  entries: CameraLiveMonitorEntry[];
  summary: CameraLiveMonitorSummary;
  onRefreshHealth: (filters?: { status_filter?: CameraStatus; group?: string }) => void;
  refreshing: boolean;
}) {
  const [selectedGroup, setSelectedGroup] = useState<string>("all");
  const [selectedHealth, setSelectedHealth] = useState<CameraStatus | "all">("all");
  const [isPending, startTransition] = useTransition();
  const deferredGroup = useDeferredValue(selectedGroup);
  const deferredHealth = useDeferredValue(selectedHealth);

  const groups = useMemo(
    () => ["all", ...Object.keys(summary.groups)],
    [summary.groups]
  );

  const filteredEntries = useMemo(
    () =>
      entries.filter((entry) => {
        const groupMatches =
          deferredGroup === "all" ? true : (entry.camera.group ?? "Ungrouped") === deferredGroup;
        const healthMatches =
          deferredHealth === "all" ? true : entry.stream.health_status === deferredHealth;
        return groupMatches && healthMatches;
      }),
    [deferredGroup, deferredHealth, entries]
  );

  return (
    <div className="space-y-5">
      <div className="flex flex-col gap-4 rounded-[24px] border border-white/10 bg-black/15 p-4">
        <div className="flex flex-col gap-3 xl:flex-row xl:items-center xl:justify-between">
          <div className="flex flex-wrap gap-3">
            <SummaryChip label="Live feeds" value={`${summary.live}/${summary.total || 0}`} tone="emerald" />
            <SummaryChip label="Browser-ready" value={`${summary.browser_ready}`} tone="cyan" />
            <SummaryChip label="Relay needed" value={`${summary.relay_required}`} tone="amber" />
            <SummaryChip label="Detection armed" value={`${summary.detection_enabled}`} tone="emerald" />
          </div>
          <Button
            type="button"
            variant="ghost"
            onClick={() =>
              onRefreshHealth({
                group: selectedGroup === "all" ? undefined : selectedGroup,
                status_filter: selectedHealth === "all" ? undefined : selectedHealth
              })
            }
            disabled={refreshing}
          >
            <RefreshCw size={16} className={cn(refreshing && "animate-spin")} />
            {refreshing ? "Refreshing health..." : "Refresh visible health"}
          </Button>
        </div>

        <div className="flex flex-col gap-4">
          <div className="flex flex-wrap gap-2">
            {healthFilters.map((filter) => (
              <button
                key={filter.value}
                type="button"
                onClick={() => startTransition(() => setSelectedHealth(filter.value))}
                className={cn(
                  "rounded-full border px-3 py-1.5 text-sm transition",
                  selectedHealth === filter.value
                    ? "border-accent/40 bg-accent/15 text-emerald-100"
                    : "border-white/10 bg-black/20 text-slate-300 hover:border-white/20 hover:text-white"
                )}
              >
                {filter.label}
              </button>
            ))}
          </div>

          <div className="flex flex-wrap gap-2">
            {groups.map((group) => (
              <button
                key={group}
                type="button"
                onClick={() => startTransition(() => setSelectedGroup(group))}
                className={cn(
                  "rounded-full border px-3 py-1.5 text-sm transition",
                  selectedGroup === group
                    ? "border-cyan-400/40 bg-cyan-500/15 text-cyan-100"
                    : "border-white/10 bg-black/20 text-slate-300 hover:border-white/20 hover:text-white"
                )}
              >
                {group === "all" ? "All groups" : `${group} (${summary.groups[group] ?? 0})`}
              </button>
            ))}
          </div>
        </div>

        {isPending ? <p className="text-sm text-slate-500">Updating the live grid filters...</p> : null}
      </div>

      {filteredEntries.length === 0 ? (
        <EmptyState
          title="No feeds match these filters"
          description="Try another health state or group to bring cameras back into the live monitoring grid."
        />
      ) : (
        <div className="grid gap-5 2xl:grid-cols-2">
          {filteredEntries.map((entry) => (
            <article key={entry.camera.id} className="rounded-[28px] border border-white/10 bg-panel/70 p-4 shadow-xl">
              <div className="mb-4 flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
                <div>
                  <div className="flex flex-wrap items-center gap-2">
                    <h3 className="text-lg font-semibold">{entry.camera.name}</h3>
                    <span className={cn("rounded-full px-2.5 py-1 text-xs font-medium", statusTone(entry.stream.health_status))}>
                      {labelize(entry.stream.health_status)}
                    </span>
                    <span className="rounded-full border border-white/10 bg-black/20 px-2.5 py-1 text-xs text-slate-300">
                      {labelize(entry.camera.source_type)}
                    </span>
                  </div>
                  <p className="mt-2 text-sm text-slate-400">
                    {entry.camera.location ?? entry.camera.source}
                  </p>
                </div>
                <div className="flex flex-wrap gap-2">
                  <MetricPill icon={<Radio size={14} aria-hidden="true" />} label={entry.stream.is_live ? "Live" : "Recorded"} />
                  <MetricPill icon={<Gauge size={14} aria-hidden="true" />} label={`${entry.camera.inference_fps} FPS`} />
                  <MetricPill icon={<Activity size={14} aria-hidden="true" />} label={formatDateTime(entry.camera.last_seen_at)} />
                </div>
              </div>

              <CameraStreamPanel
                accessToken={accessToken}
                camera={entry.camera}
                stream={entry.stream}
                variant="tile"
              />

              <div className="mt-4 flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
                <div className="flex flex-wrap gap-2 text-sm text-slate-400">
                  <span>Group: {entry.camera.group ?? "Ungrouped"}</span>
                  <span>Detection: {entry.camera.detection_enabled ? "Armed" : "Paused"}</span>
                  <span>{entry.stream.requires_relay ? "Relay workflow" : "Direct playback"}</span>
                </div>
                <Link
                  href={`/dashboard/cameras/${entry.camera.id}`}
                  className="inline-flex items-center gap-2 text-sm font-medium text-accent transition hover:text-emerald-300"
                >
                  Open camera details
                </Link>
              </div>
            </article>
          ))}
        </div>
      )}
    </div>
  );
}

function SummaryChip({
  label,
  value,
  tone
}: {
  label: string;
  value: string;
  tone: "emerald" | "cyan" | "amber";
}) {
  return (
    <div className="rounded-full border border-white/10 bg-black/20 px-4 py-2">
      <p className="text-xs uppercase tracking-[0.18em] text-slate-500">{label}</p>
      <p
        className={cn(
          "mt-1 text-sm font-medium",
          tone === "emerald" && "text-emerald-200",
          tone === "cyan" && "text-cyan-100",
          tone === "amber" && "text-amber-200"
        )}
      >
        {value}
      </p>
    </div>
  );
}

function MetricPill({ icon, label }: { icon: ReactNode; label: string }) {
  return (
    <span className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-black/20 px-3 py-1.5 text-xs text-slate-300">
      {icon}
      {label}
    </span>
  );
}
