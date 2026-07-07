"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  BellRing,
  Camera,
  ChartColumn,
  ChevronRight,
  LogOut,
  Shield,
  Siren,
  Users
} from "lucide-react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import type { ReactNode } from "react";
import { useEffect } from "react";
import { Button } from "@/components/button";
import { fetchCurrentUser, subscribeToLiveEvents, type LiveEvent } from "@/lib/api";
import { labelize, statusTone } from "@/lib/format";
import { useAuthStore } from "@/lib/auth-store";
import { cn } from "@/lib/cn";

const navigation = [
  { href: "/dashboard", label: "Overview", icon: Shield },
  { href: "/dashboard/cameras", label: "Cameras", icon: Camera },
  { href: "/dashboard/incidents", label: "Incidents", icon: Siren },
  { href: "/dashboard/persons", label: "Persons", icon: Users },
  { href: "/dashboard/users", label: "Users", icon: Users },
  { href: "/dashboard/analytics", label: "Analytics", icon: ChartColumn }
];

export function DashboardShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const queryClient = useQueryClient();
  const { accessToken, hydrated, user, setUser, hydrate, logout } = useAuthStore();

  useEffect(() => {
    hydrate();
  }, [hydrate]);

  useEffect(() => {
    if (hydrated && !accessToken) {
      router.replace("/login");
    }
  }, [accessToken, hydrated, router]);

  const userQuery = useQuery({
    queryKey: ["auth", "me", accessToken],
    queryFn: () => fetchCurrentUser(accessToken!),
    enabled: hydrated && Boolean(accessToken),
    retry: false
  });

  useEffect(() => {
    if (userQuery.data) {
      setUser(userQuery.data);
    }
  }, [setUser, userQuery.data]);

  useEffect(() => {
    if (userQuery.error instanceof Error) {
      logout();
      router.replace("/login");
    }
  }, [logout, router, userQuery.error]);

  useEffect(() => {
    if (!accessToken) {
      return;
    }

    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    let closed = false;

    const connect = () =>
      subscribeToLiveEvents(
        accessToken,
        (event: LiveEvent) => {
          if (event.type === "system.connected") {
            return;
          }

          void queryClient.invalidateQueries({ queryKey: ["incidents"] });
          void queryClient.invalidateQueries({ queryKey: ["alerts"] });
          if (event.incident_id) {
            void queryClient.invalidateQueries({ queryKey: ["incident", event.incident_id] });
            void queryClient.invalidateQueries({ queryKey: ["incident-alerts", event.incident_id] });
          }
        },
        () => {
          if (!closed) {
            reconnectTimer = setTimeout(() => {
              disconnect = connect();
            }, 1500);
          }
        }
      );

    let disconnect = connect();

    return () => {
      closed = true;
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
      }
      disconnect();
    };
  }, [accessToken, queryClient]);

  if (!hydrated || (accessToken && userQuery.isLoading && !user)) {
    return <DashboardLoadingState />;
  }

  if (!accessToken) {
    return null;
  }

  return (
    <div className="min-h-screen">
      <div className="mx-auto flex min-h-screen max-w-7xl gap-6 px-4 py-6 lg:px-6">
        <aside className="hidden w-72 shrink-0 flex-col rounded-[28px] border border-white/10 bg-panel/80 p-5 shadow-2xl backdrop-blur lg:flex">
          <div className="flex items-center gap-3">
            <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-accent/90 text-slate-950">
              <Shield size={24} aria-hidden="true" />
            </div>
            <div>
              <p className="text-lg font-semibold">AegisPro</p>
              <p className="text-sm text-slate-400">Security operations center</p>
            </div>
          </div>

          <nav className="mt-8 space-y-2">
            {navigation.map(({ href, label, icon: Icon }) => {
              const active = pathname === href || pathname.startsWith(`${href}/`);

              return (
                <Link
                  key={href}
                  href={href}
                  className={cn(
                    "flex items-center justify-between rounded-2xl px-4 py-3 text-sm text-slate-300 transition hover:bg-white/5 hover:text-white",
                    active && "bg-white/8 text-white ring-1 ring-accent/30"
                  )}
                >
                  <span className="flex items-center gap-3">
                    <Icon size={18} aria-hidden="true" />
                    {label}
                  </span>
                  <ChevronRight size={16} aria-hidden="true" className={active ? "opacity-100" : "opacity-40"} />
                </Link>
              );
            })}
          </nav>

          <div className="mt-auto rounded-3xl border border-white/10 bg-black/20 p-4">
            <p className="text-xs uppercase tracking-[0.22em] text-slate-500">Operator</p>
            <p className="mt-3 text-base font-semibold">{user?.full_name ?? "Loading"}</p>
            <p className="text-sm text-slate-400">{user?.email}</p>
            <div className="mt-4 flex items-center justify-between">
              <span className={cn("rounded-full px-3 py-1 text-xs font-medium", statusTone(user?.role ?? "viewer"))}>
                {labelize(user?.role ?? "viewer")}
              </span>
              <Button variant="ghost" onClick={() => logoutAndRedirect(logout, router)}>
                <LogOut size={16} aria-hidden="true" />
                Sign out
              </Button>
            </div>
          </div>
        </aside>

        <div className="flex min-h-screen flex-1 flex-col gap-6">
          <header className="rounded-[28px] border border-white/10 bg-panel/70 px-5 py-4 shadow-xl backdrop-blur">
            <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
              <div>
                <p className="text-xs uppercase tracking-[0.24em] text-slate-500">Phase 8 monitoring</p>
                <h1 className="mt-2 text-2xl font-semibold">Live operations workspace</h1>
              </div>
              <div className="flex flex-wrap items-center gap-3">
                <div className="flex h-11 items-center gap-2 rounded-full border border-white/10 bg-black/20 px-4 lg:hidden">
                  <BellRing size={16} aria-hidden="true" className="text-accent" />
                  <span className="text-sm text-slate-300">{user?.full_name}</span>
                </div>
              </div>
            </div>
          </header>

          <main>{children}</main>
        </div>
      </div>
    </div>
  );
}

function DashboardLoadingState() {
  return (
    <main className="flex min-h-screen items-center justify-center px-4">
      <div className="w-full max-w-4xl animate-pulse rounded-[32px] border border-white/10 bg-panel/70 p-10 backdrop-blur">
        <div className="h-6 w-48 rounded-full bg-white/10" />
        <div className="mt-8 grid gap-4 md:grid-cols-3">
          <div className="h-24 rounded-3xl bg-white/10" />
          <div className="h-24 rounded-3xl bg-white/10" />
          <div className="h-24 rounded-3xl bg-white/10" />
        </div>
      </div>
    </main>
  );
}

function logoutAndRedirect(logout: () => void, router: ReturnType<typeof useRouter>) {
  logout();
  router.replace("/login");
}
