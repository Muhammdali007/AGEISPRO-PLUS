"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Trash2 } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Button } from "@/components/button";
import { EmptyState, InlineLink, MetricCard, SectionCard } from "@/components/dashboard-ui";
import { deleteUser, listUsers } from "@/lib/api";
import { useAuthStore } from "@/lib/auth-store";
import { labelize, statusTone } from "@/lib/format";
import { cn } from "@/lib/cn";

export default function UsersPage() {
  const router = useRouter();
  const queryClient = useQueryClient();
  const { accessToken, user: currentUser, logout } = useAuthStore();
  const usersQuery = useQuery({
    queryKey: ["users", "list", accessToken],
    queryFn: () => listUsers(accessToken!),
    enabled: Boolean(accessToken)
  });

  const users = usersQuery.data ?? [];
  const canManageUsers = currentUser?.role === "administrator" || currentUser?.role === "supervisor";
  const deleteUserMutation = useMutation({
    mutationFn: async (userId: string) => {
      if (!accessToken) {
        throw new Error("You need to sign in again before deleting a user.");
      }

      await deleteUser(accessToken, userId);
    },
    onSuccess: async (_result, userId) => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["users", "list", accessToken] }),
        queryClient.invalidateQueries({ queryKey: ["auth", "me", accessToken] })
      ]);

      if (userId === currentUser?.id) {
        logout();
        router.push("/login");
      }
    }
  });

  async function handleDelete(userId: string, fullName: string) {
    const confirmed = window.confirm(`Delete ${fullName}? This will permanently remove the account.`);
    if (!confirmed) {
      return;
    }

    try {
      await deleteUserMutation.mutateAsync(userId);
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : "Unable to delete user";
      window.alert(message);
    }
  }

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
        description="Create, update, and deactivate platform accounts with RBAC-aware access controls."
        action={
          canManageUsers ? (
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
            {users.map((managedUser) => {
              const canManageAccount =
                currentUser?.role === "administrator" || managedUser.role !== "administrator";

              return (
                <div
                  key={managedUser.id}
                  className="grid gap-4 rounded-[24px] border border-white/10 bg-black/15 p-4 md:grid-cols-[1.1fr_0.75fr_0.75fr_0.9fr]"
                >
                  <div>
                    <p className="font-medium">{managedUser.full_name}</p>
                    <p className="mt-2 text-sm text-slate-400">{managedUser.email}</p>
                  </div>
                  <div>
                    <p className="text-xs uppercase tracking-[0.18em] text-slate-500">Role</p>
                    <span className={cn("mt-3 inline-flex rounded-full px-2.5 py-1 text-xs font-medium", statusTone(managedUser.role))}>
                      {labelize(managedUser.role)}
                    </span>
                  </div>
                  <div>
                    <p className="text-xs uppercase tracking-[0.18em] text-slate-500">State</p>
                    <span
                      className={cn(
                        "mt-3 inline-flex rounded-full px-2.5 py-1 text-xs font-medium",
                        managedUser.is_active ? "bg-emerald-500/15 text-emerald-200" : "bg-slate-500/20 text-slate-200"
                      )}
                    >
                      {managedUser.is_active ? "Active" : "Inactive"}
                    </span>
                  </div>
                  <div className="flex flex-wrap items-center gap-3 md:justify-end">
                    {canManageUsers && canManageAccount ? (
                      <>
                        <Link
                          href={`/dashboard/users/${managedUser.id}`}
                          className="inline-flex h-10 items-center justify-center rounded-md border border-border px-4 text-sm font-medium text-slate-200 transition hover:bg-panelSoft"
                        >
                          Edit
                        </Link>
                        <Button
                          type="button"
                          variant="ghost"
                          className="border-red-400/30 text-red-200 hover:bg-red-500/10"
                          disabled={deleteUserMutation.isPending}
                          onClick={() => handleDelete(managedUser.id, managedUser.full_name)}
                        >
                          <Trash2 size={16} aria-hidden="true" />
                          Delete
                        </Button>
                      </>
                    ) : null}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </SectionCard>
    </div>
  );
}
