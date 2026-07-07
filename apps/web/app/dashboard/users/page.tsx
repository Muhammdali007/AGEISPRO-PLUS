"use client";

import { useQuery } from "@tanstack/react-query";
import { EmptyState, InlineLink, MetricCard, SectionCard } from "@/components/dashboard-ui";
import { listUsers } from "@/lib/api";
import { useAuthStore } from "@/lib/auth-store";
import { labelize, statusTone } from "@/lib/format";
import { cn } from "@/lib/cn";

export default function UsersPage() {
  const { accessToken, user } = useAuthStore();
  const usersQuery = useQuery({
    queryKey: ["users", "list", accessToken],
    queryFn: () => listUsers(accessToken!),
    enabled: Boolean(accessToken)
  });

  const users = usersQuery.data ?? [];

  return (
    <div className="space-y-6">
      <section className="grid gap-4 md:grid-cols-4">
        <MetricCard label="Users" value={`${users.length}`} detail="Accounts available to the operations platform." />
        <MetricCard label="Administrators" value={`${users.filter((user) => user.role === "administrator").length}`} detail="Highest privilege operators." />
        <MetricCard label="Supervisors" value={`${users.filter((user) => user.role === "supervisor").length}`} detail="Review and oversight coverage." />
        <MetricCard label="Active accounts" value={`${users.filter((user) => user.is_active).length}`} detail="RBAC-aware access state from the backend." tone="success" />
      </section>

      <SectionCard
        title="User management"
        description="Phase 3 surfaces the role model and account inventory. Editing flows can layer onto the same contracts without redesign."
        action={
          user?.role === "administrator" ? (
            <InlineLink href="/dashboard/users/create" label="Create user" />
          ) : undefined
        }
      >
        {users.length === 0 ? (
          <EmptyState
            title="No users returned"
            description="The route and query plumbing are ready, but the current account does not have visible users or the environment is empty."
          />
        ) : (
          <div className="space-y-3">
            {users.map((user) => (
              <div
                key={user.id}
                className="grid gap-4 rounded-[24px] border border-white/10 bg-black/15 p-4 md:grid-cols-[1.1fr_0.8fr_0.8fr]"
              >
                <div>
                  <p className="font-medium">{user.full_name}</p>
                  <p className="mt-2 text-sm text-slate-400">{user.email}</p>
                </div>
                <div>
                  <p className="text-xs uppercase tracking-[0.18em] text-slate-500">Role</p>
                  <span className={cn("mt-3 inline-flex rounded-full px-2.5 py-1 text-xs font-medium", statusTone(user.role))}>
                    {labelize(user.role)}
                  </span>
                </div>
                <div>
                  <p className="text-xs uppercase tracking-[0.18em] text-slate-500">State</p>
                  <span
                    className={cn(
                      "mt-3 inline-flex rounded-full px-2.5 py-1 text-xs font-medium",
                      user.is_active ? "bg-emerald-500/15 text-emerald-200" : "bg-slate-500/20 text-slate-200"
                    )}
                  >
                    {user.is_active ? "Active" : "Inactive"}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </SectionCard>
    </div>
  );
}
