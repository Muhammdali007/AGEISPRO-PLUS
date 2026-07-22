# Phase 10: Production Deployment

Phase 10 packages the Phase 9 application boundary for a single-host production
deployment. The AI service remains independently replaceable and can be moved to
a GPU host later without changing API business logic.

## Architecture and trade-offs

```text
Internet
   |
Nginx (TLS, security headers, rate limiting)
   |--------------------------|
Next.js                    FastAPI/WebSocket
                              |
                     PostgreSQL / Redis / AI

Prometheus <- exporters and HTTP probes -> Grafana
Backup job -> PostgreSQL dump + evidence archive
```

- Only Nginx publishes production ports. Database, Redis, API, AI, monitoring,
  and frontend services remain on the Compose network.
- TLS terminates at Nginx. Certificates are mounted read-only so certificate
  issuance can use an external ACME client or managed certificate process.
- Gunicorn supervises multiple API workers. WebSocket clients must tolerate
  reconnects; Redis-backed fan-out remains the scaling path for multiple hosts.
- Prometheus and Grafana provide infrastructure monitoring. Container logs use
  bounded JSON rotation to prevent unbounded disk growth.
- Backups use portable PostgreSQL custom-format dumps and storage archives.
  They favor recoverability and simplicity over zero-downtime snapshots.

## Production files

- `docker-compose.yml`: shared service definitions.
- `docker-compose.prod.yml`: production isolation, health checks, logging,
  Gunicorn, Prometheus, exporters, and Grafana.
- `infra/nginx/production.conf`: HTTPS reverse proxy and secure headers.
- `infra/monitoring/`: scrape, probe, alert, and Grafana provisioning.
- `infra/backup/backup.sh`: database, evidence, environment snapshot, checksums,
  retention.
- `infra/backup/restore.sh`: checksum-verified guarded restore.

## Initial deployment

Requirements: Linux host, Docker Engine, Docker Compose 2.24 or newer, DNS
pointing at the host, and a TLS certificate.

1. Clone the repository to `/opt/aegispro`.
2. Copy `.env.production.example` to `.env.production`.
3. Replace every `replace-with-*` value. Use independent random values for the
   application secret, callback token, database, admin, SendGrid, and Grafana
   passwords or API keys.
4. Set `DOMAIN`, `CORS_ORIGINS=https://<domain>`, and both public web variables.
5. Place `fullchain.pem` and `privkey.pem` in `TLS_CERT_DIR`.
6. Place trained model files and signed promotion manifests in `storage/models/`.
   Production validation intentionally rejects simulated inference, hash
   recognition, fallbacks, missing checkpoints, and unsigned weapon/fire-smoke
   promotions.
7. Place the latest runtime validation report at the path configured by
   `AI_RUNTIME_GATE_REPORT_PATH`. The rollout gate expects `load`, `soak_8h`,
   `soak_24h`, and `soak_72h` results.
8. Validate and start:

```bash
docker compose --env-file .env.production \
  -f docker-compose.yml -f docker-compose.prod.yml config --quiet

docker compose --env-file .env.production \
  -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

Verify:

```bash
curl --fail https://<domain>/backend/api/v1/health
curl --fail https://<domain>/backend/api/v1/health/ready
docker compose --env-file .env.production \
  -f docker-compose.yml -f docker-compose.prod.yml ps
```

Grafana is available at `https://<domain>/ops/grafana/`. Prometheus is not
published directly.

## HTTPS certificate renewal

Use a host-level ACME client or managed certificate issuer. Renew into the
directory configured by `TLS_CERT_DIR`, then reload Nginx:

```bash
docker compose --env-file .env.production \
  -f docker-compose.yml -f docker-compose.prod.yml exec nginx nginx -s reload
```

Test renewal and certificate expiry monitoring before launch. Never place
private keys in Git.

## Backup and recovery

Run the backup job as root from cron:

```cron
15 2 * * * PROJECT_DIR=/opt/aegispro BACKUP_ROOT=/var/backups/aegispro RETENTION_DAYS=30 /opt/aegispro/infra/backup/backup.sh
```

Copy backups off-host to encrypted object storage. A local-only backup does not
protect against host loss. Suggested policy: 30 daily, 12 monthly, and at least
one quarterly restore drill.

Restore during a maintenance window:

```bash
CONFIRM_RESTORE=yes PROJECT_DIR=/opt/aegispro \
  ./infra/backup/restore.sh /var/backups/aegispro/<timestamp>
```

The restore script stops application writers, replaces the database and storage,
then restarts the stack. Verify login, camera inventory, incident evidence,
alerts, and monitoring after every restore drill.

### Administrator account recovery

If an active administrator forgets their password, they can use the login
page's password reset flow. AegisPro sends single-use SendGrid reset links that
expire after `PASSWORD_RESET_TOKEN_MINUTES`. Public reset responses are neutral
and do not reveal whether an email address belongs to an active administrator.

If another administrator can still sign in, they can also reset the password
from the dashboard user-management screen. Supervisors cannot modify
administrator accounts.

If email delivery is unavailable, every administrator is locked out, or the
target administrator is inactive/deleted, recover the bootstrap administrator
from the API host or API container:

```bash
cd apps/api
python -m app.management.reset_admin_password \
  --email admin@your-domain.example.com
```

The command prompts for the new password without echoing it. It resets and
reactivates an existing administrator account, or recreates the administrator if
the account was deleted. It refuses to promote a non-administrator account.

## Operations

- Review Grafana and Prometheus alerts daily.
- Forward Docker JSON logs to the organization log platform when deploying
  beyond one host.
- Alert on disk capacity because evidence, PostgreSQL, Prometheus, and Grafana
  share host storage.
- Apply OS and image security updates on a staged cadence.
- Rotate secrets and bootstrap credentials after first login.
- Scale `API_WORKERS` conservatively because each worker has its own database
  pool. Keep total possible connections below PostgreSQL limits.
- Keep `API_CONTINUOUS_DETECTION_BATCH_SIZE`,
  `API_CONTINUOUS_DETECTION_MAX_PENDING_PER_CAMERA`, and `AI_MODEL_BATCH_SIZE`
  aligned with the GPU host you validated during the load and soak runs.
- Keep `AI_MODEL_RUNTIME_AUTOINSTALL=false` in production so tracker dependencies
  are provisioned at build time rather than during live traffic.

## Rollback

Keep the previous image tags and a pre-deployment backup. To roll back, restore
the prior Compose/image version, run only backward-compatible database changes,
and restore the database only when the migration cannot be safely reversed.

## Validation checklist

- Compose configuration renders with no warnings or unresolved variables.
- HTTP redirects to HTTPS; TLS 1.2/1.3 and HSTS are active.
- Only ports 80 and 443 are externally reachable.
- API readiness, AI health, and web health checks pass.
- WebSocket alerts reconnect through Nginx.
- Grafana receives PostgreSQL, Redis, and service probe data.
- Log files rotate under sustained traffic.
- Backup checksums validate and a restore drill succeeds.
- Signed weapon and fire/smoke promotion manifests match the deployed checkpoint hashes.
- InsightFace and the promoted detector set pass a production smoke test.
- The load, 8-hour, 24-hour, and 72-hour runtime validation gates are present and passing.
