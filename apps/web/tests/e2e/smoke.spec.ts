import { expect, test, type Page, type Route } from "@playwright/test";

const adminEmail = "admin@aegispro.local";
const adminPassword = "ChangeMe123!";
const accessToken = "playwright-access-token";
const refreshToken = "playwright-refresh-token";

test.beforeEach(async ({ page }) => {
  await stubBackend(page);
  await clearBrowserState(page);
});

test("redirects anonymous dashboard visitors to login", async ({ page }) => {
  await page.goto("/dashboard");

  await expect(page).toHaveURL(/\/login$/);
  await expect(page.getByRole("heading", { name: "AegisPro" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Sign in" })).toBeVisible();
});

test("allows an administrator to sign in and navigate the main dashboard routes", async ({ page }) => {
  await loginViaUi(page);

  await expect(page.getByRole("heading", { name: "Live operations workspace" })).toBeVisible();
  await expect(page.getByText("Phase 9 optimization", { exact: true })).toBeVisible();

  await page.getByRole("link", { name: /Cameras/i }).click();
  await expect(page).toHaveURL(/\/dashboard\/cameras$/);
  await expect(page.getByText("Camera registry")).toBeVisible();

  await page.getByRole("link", { name: /Incidents/i }).click();
  await expect(page).toHaveURL(/\/dashboard\/incidents$/);
  await expect(page.getByText("Incident queue")).toBeVisible();

  await page.getByRole("link", { name: /Persons/i }).click();
  await expect(page).toHaveURL(/\/dashboard\/persons$/);
  await expect(page.getByText("Known person registry")).toBeVisible();

  await page.getByRole("link", { name: /Users/i }).click();
  await expect(page).toHaveURL(/\/dashboard\/users$/);
  await expect(page.getByText("User management")).toBeVisible();
});

test("loads the phase 9 analytics dashboard with optimization telemetry", async ({ page }) => {
  await loginViaUi(page);

  await page.goto("/dashboard/analytics");

  await expect(page.getByText("Operational monitoring and optimization")).toBeVisible();
  await expect(page.getByText("Incident volume")).toBeVisible();
  await expect(page.getByText("Detection mix")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Camera health" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "System health" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Optimization report" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Recent audit log" })).toBeVisible();
  await expect(page.getByText("alerts.clear")).toBeVisible();
});

async function clearBrowserState(page: Page) {
  await page.goto("/");
  await page.evaluate(() => {
    localStorage.clear();
    sessionStorage.clear();
  });
}

async function loginViaUi(page: Page) {
  await page.goto("/login");
  await page.getByLabel("Email").fill(adminEmail);
  await page.getByLabel("Password").fill(adminPassword);
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page).toHaveURL(/\/dashboard$/);
}

async function stubBackend(page: Page) {
  await page.route("**/backend/api/v1/**", async (route) => {
    const url = new URL(route.request().url());
    const path = url.pathname.replace("/backend", "");
    const method = route.request().method();

    if (path === "/api/v1/health/ready") {
      await fulfillJson(route, {
        status: "ok",
        database: "ok",
        redis: "ok"
      });
      return;
    }

    if (path === "/api/v1/auth/login" && method === "POST") {
      await fulfillJson(route, {
        access_token: accessToken,
        refresh_token: refreshToken,
        token_type: "bearer"
      });
      return;
    }

    if (path === "/api/v1/auth/me") {
      await fulfillJson(route, {
        id: "user-admin",
        email: adminEmail,
        full_name: "Playwright Admin",
        role: "administrator",
        is_active: true
      });
      return;
    }

    if (path === "/api/v1/cameras") {
      await fulfillJson(route, [
        {
          id: "camera-1",
          name: "North Gate",
          source_type: "http",
          source: "https://streams.aegispro.local/north-gate.m3u8",
          status: "online",
          location: "North Gate",
          group: "perimeter",
          tags: ["perimeter"],
          detection_enabled: true,
          inference_fps: 6,
          metadata: {},
          last_seen_at: "2026-07-07T12:00:00Z",
          health_checked_at: "2026-07-07T12:00:00Z",
          created_at: "2026-07-07T11:00:00Z",
          updated_at: "2026-07-07T12:00:00Z"
        }
      ]);
      return;
    }

    if (path === "/api/v1/incidents") {
      await fulfillJson(route, [
        {
          id: "incident-1",
          camera_id: "camera-1",
          detection_type: "fire",
          priority: "critical",
          status: "open",
          confidence: 0.91,
          occurred_at: "2026-07-07T11:45:00Z",
          bounding_boxes: [],
          snapshot_path: null,
          clip_path: null,
          recognized_identity: null,
          operator_notes: null,
          assigned_user_id: null,
          metadata: {},
          created_at: "2026-07-07T11:45:00Z",
          updated_at: "2026-07-07T11:45:00Z"
        }
      ]);
      return;
    }

    if (path === "/api/v1/alerts") {
      await fulfillJson(route, [
        {
          id: "alert-1",
          incident_id: "incident-1",
          priority: "critical",
          status: "active",
          title: "Fire detected",
          message: "Fire detected on camera North Gate with confidence 0.91.",
          acknowledged: false,
          acknowledged_by_id: null,
          acknowledged_at: null,
          created_at: "2026-07-07T11:45:00Z",
          updated_at: "2026-07-07T11:45:00Z"
        }
      ]);
      return;
    }

    if (path === "/api/v1/users") {
      await fulfillJson(route, [
        {
          id: "user-admin",
          email: adminEmail,
          full_name: "Playwright Admin",
          role: "administrator",
          is_active: true
        }
      ]);
      return;
    }

    if (path === "/api/v1/persons") {
      await fulfillJson(route, [
        {
          id: "person-1",
          full_name: "Known Visitor",
          person_type: "visitor",
          department: "Front Office",
          reference_id: "VIS-001",
          title: "Guest",
          is_active: true,
          face_profiles: [],
          face_image_count: 1,
          embedding_count: 1,
          visit_count: 2,
          recognition_count: 2,
          last_seen_at: "2026-07-07T11:40:00Z",
          last_recognized_at: "2026-07-07T11:40:00Z",
          metadata: {},
          created_at: "2026-07-07T10:00:00Z",
          updated_at: "2026-07-07T11:40:00Z"
        }
      ]);
      return;
    }

    if (path === "/api/v1/cameras/live-monitor") {
      await fulfillJson(route, {
        summary: {
          total: 1,
          online: 1,
          offline: 0,
          degraded: 0,
          disabled: 0,
          unknown: 0,
          live: 1,
          browser_ready: 1,
          relay_required: 0,
          detection_enabled: 1,
          groups: {
            perimeter: 1
          }
        },
        entries: [
          {
            camera: {
              id: "camera-1",
              name: "North Gate",
              source_type: "http",
              source: "https://streams.aegispro.local/north-gate.m3u8",
              status: "online",
              location: "North Gate",
              group: "perimeter",
              tags: ["perimeter"],
              detection_enabled: true,
              inference_fps: 6,
              metadata: {},
              last_seen_at: "2026-07-07T12:00:00Z",
              health_checked_at: "2026-07-07T12:00:00Z",
              created_at: "2026-07-07T11:00:00Z",
              updated_at: "2026-07-07T12:00:00Z"
            },
            stream: {
              camera_id: "camera-1",
              stream_kind: "hls",
              stream_url: "https://streams.aegispro.local/north-gate.m3u8",
              browser_supported: true,
              requires_relay: false,
              is_live: true,
              health_status: "online",
              health_message: "HTTP HLS feed available.",
              checked_at: "2026-07-07T12:00:00Z",
              controls: [],
              notes: ["HTTP HLS feed available."],
              browser_device_id: null
            }
          }
        ]
      });
      return;
    }

    if (path === "/api/v1/monitoring/overview") {
      await fulfillJson(route, {
        window: url.searchParams.get("window") ?? "24h",
        generated_at: "2026-07-07T12:00:00Z",
        kpis: {
          incident_volume: 8,
          active_alerts: 2,
          online_camera_ratio: 0.75,
          average_confidence: 0.88
        },
        incidents_over_time: [
          { bucket: "2026-07-07T09:00:00Z", label: "09:00", value: 1 },
          { bucket: "2026-07-07T10:00:00Z", label: "10:00", value: 3 },
          { bucket: "2026-07-07T11:00:00Z", label: "11:00", value: 4 }
        ],
        detection_mix: [
          { detection_type: "fire", count: 4 },
          { detection_type: "smoke", count: 2 },
          { detection_type: "unknown_person", count: 2 }
        ],
        camera_health: {
          total: 4,
          online: 3,
          offline: 1,
          degraded: 0,
          disabled: 0,
          unknown: 0,
          stale: 1,
          detection_enabled: 4,
          groups: {
            perimeter: 2,
            lobby: 2
          }
        },
        system_health: {
          generated_at: "2026-07-07T12:00:00Z",
          api: { status: "ok", detail: null },
          database: { status: "ok", detail: null },
          redis: { status: "ok", detail: null },
          ai: {
            status: "ok",
            inference_backend: "ultralytics",
            fallback_backend: null,
            recognition_backend: "hash",
            recognition_providers: ["CPUExecutionProvider"],
            model_device: null,
            gpu_available: false,
            gpu_name: null,
            gpu_memory_total_mb: null,
            gpu_memory_used_mb: null,
            gpu_utilization_percent: null,
            telemetry_supported: false,
            detail: "CUDA is not available on this host."
          }
        }
      });
      return;
    }

    if (path === "/api/v1/monitoring/camera-health") {
      await fulfillJson(route, {
        stale_threshold_minutes: 5,
        generated_at: "2026-07-07T12:00:00Z",
        summary: {
          total: 4,
          online: 3,
          offline: 1,
          degraded: 0,
          disabled: 0,
          unknown: 0,
          stale: 1,
          detection_enabled: 4,
          groups: {
            perimeter: 2,
            lobby: 2
          }
        },
        entries: [
          {
            camera_id: "camera-1",
            name: "North Gate",
            status: "online",
            group: "perimeter",
            last_seen_at: "2026-07-07T12:00:00Z",
            health_checked_at: "2026-07-07T12:00:00Z",
            stale: false,
            detection_enabled: true
          },
          {
            camera_id: "camera-2",
            name: "Lobby West",
            status: "offline",
            group: "lobby",
            last_seen_at: "2026-07-07T11:40:00Z",
            health_checked_at: "2026-07-07T12:00:00Z",
            stale: true,
            detection_enabled: true
          }
        ]
      });
      return;
    }

    if (path === "/api/v1/monitoring/system-health") {
      await fulfillJson(route, {
        generated_at: "2026-07-07T12:00:00Z",
        api: { status: "ok", detail: null },
        database: { status: "ok", detail: null },
        redis: { status: "ok", detail: null },
        ai: {
          status: "ok",
          inference_backend: "ultralytics",
          fallback_backend: null,
          recognition_backend: "hash",
          recognition_providers: ["CPUExecutionProvider"],
          model_device: null,
          gpu_available: false,
          gpu_name: null,
          gpu_memory_total_mb: null,
          gpu_memory_used_mb: null,
          gpu_utilization_percent: null,
          telemetry_supported: false,
          detail: "CUDA is not available on this host."
        }
      });
      return;
    }

    if (path === "/api/v1/monitoring/optimization") {
      await fulfillJson(route, {
        generated_at: "2026-07-07T12:00:00Z",
        database: {
          status: "ok",
          pool_size: 10,
          max_overflow: 20,
          pool_recycle_seconds: 1800,
          indexed_paths: [
            "incidents(occurred_at, detection_type)",
            "alerts(status, created_at)"
          ],
          resources: {
            incidents_total: 240,
            incidents_last_24h: 8,
            active_alerts_total: 2,
            alerts_last_24h: 4,
            audit_logs_total: 920,
            audit_logs_last_24h: 24
          },
          detail: "Monitoring aggregates are computed through filtered SQL queries to reduce Python-side memory pressure."
        },
        redis: {
          status: "ok",
          ping_ms: 3.2,
          used_memory_human: "1.21M",
          connected_clients: 4,
          pubsub_channels: 2,
          detail: "Redis health verified with a live ping and INFO sampling."
        },
        runtime: {
          status: "ok",
          inference_backend: "ultralytics",
          recognition_backend: "hash",
          gpu_available: false,
          gpu_utilization_percent: null,
          gpu_memory_used_mb: null,
          gpu_memory_total_mb: null,
          detail: "CUDA is not available on this host."
        },
        recommendations: [
          {
            title: "Database-side aggregation",
            detail: "Phase 9 monitoring now aggregates incidents and alerts in SQL before shaping dashboard responses.",
            severity: "info"
          }
        ]
      });
      return;
    }

    if (path === "/api/v1/monitoring/audit-logs") {
      await fulfillJson(route, {
        items: [
          {
            id: "audit-1",
            actor_user_id: "user-admin",
            actor_email: adminEmail,
            actor_role: "administrator",
            action: "alerts.clear",
            resource_type: "alert",
            resource_id: "alert-1",
            metadata: {
              incident_id: "incident-1"
            },
            created_at: "2026-07-07T11:50:00Z"
          }
        ],
        total: 1,
        limit: 12,
        offset: 0
      });
      return;
    }

    await route.abort();
  });
}

async function fulfillJson(route: Route, body: unknown) {
  await route.fulfill({
    status: 200,
    contentType: "application/json",
    body: JSON.stringify(body)
  });
}
