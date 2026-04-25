# Deployment

Push to `main` → GitHub Actions runs tests, builds a Docker image, publishes
it to `ghcr.io/martyjt/daytrader:main`. Watchtower (running from the
sibling cashflow stack) polls every 5 min and recreates the app container
when the digest changes.

External hostname: `daytrader.zimpla.nz`. Cloudflared resolves the
container by service name on the shared `edge` docker network.

## One-time host setup (marty@192.168.18.222)

These steps already ran during the D1 phase and are recorded here for
disaster recovery / a future host migration.

### 1. SSH key auth

The deploy laptop's `~/.ssh/daytrader_deploy_id.pub` is appended to the
host's `~marty/.ssh/authorized_keys`. After this you can drop the
password — `scripts/deploy_remote.py` defaults to key auth.

### 2. Working directory

```bash
mkdir -p ~/daytrader/{data,backups}
chmod 700 ~/daytrader
```

### 3. GHCR authentication (already set up by the cashflow stack)

The host's `~/.docker/config.json` is already authenticated to
`ghcr.io` with a PAT carrying `read:packages` scope. Watchtower mounts
that file so it can poll private images. No additional auth needed for
the daytrader image — it lives under the same account.

### 4. Drop in `docker-compose.yml` and `.env`

From the laptop:

```bash
DEPLOY_HOST_PASSWORD=...  # only on first contact; key auth thereafter
python scripts/deploy_remote.py --put docker-compose.yml ~/daytrader/docker-compose.yml
python scripts/deploy_remote.py --put .env.local ~/daytrader/.env
python scripts/deploy_remote.py "chmod 600 ~/daytrader/.env"
```

Build the `.env` from `.env.deploy.example` first, filling in real
values. Generate the two secrets locally:

```bash
python -c "import secrets; print('APP_SECRET_KEY=' + secrets.token_urlsafe(48))"
python -c "from cryptography.fernet import Fernet; print('APP_ENCRYPTION_KEY=' + Fernet.generate_key().decode())"
```

### 5. Bring up the stack

```bash
python scripts/deploy_remote.py "cd ~/daytrader && docker compose pull && docker compose up -d"
python scripts/deploy_remote.py "docker compose -f ~/daytrader/docker-compose.yml ps"
python scripts/deploy_remote.py "docker logs --tail 100 daytrader-app"
```

App is now reachable at `http://192.168.18.222:8083` from the LAN and
(once the cloudflared route is added) at `https://daytrader.zimpla.nz`
from anywhere.

### 6. Cloudflare Zero Trust route

In the dashboard:
* Zero Trust → Networks → Tunnels → (existing zimpla.nz tunnel) → Public Hostnames
* Add: subdomain `daytrader`, domain `zimpla.nz`, service `http://daytrader-app:8080`

The `edge` docker network is shared with the cloudflared sidecar, so it
can resolve `daytrader-app` directly. Cloudflare handles TLS.

## Ongoing workflow

1. Open a PR, merge to `main`.
2. Actions runs: tests → build → push to `ghcr.io/martyjt/daytrader:main`.
3. Within 5 minutes Watchtower pulls the new digest, stops the old
   container, starts the new one. The app's entrypoint runs
   `alembic upgrade head` before launching, so any pending migrations
   apply on first start. SQLite is gone; Postgres lives in the
   `daytrader_pgdata` named volume and survives container recreation.

Watch the loop:
```bash
python scripts/deploy_remote.py "docker logs watchtower --tail 30"
```

## Manual redeploy (without waiting for Watchtower)

```bash
python scripts/deploy_remote.py "cd ~/daytrader && docker compose pull && docker compose up -d"
```

## Rollback to a prior image

Every commit publishes `ghcr.io/martyjt/daytrader:sha-<short>`. To pin:

```bash
# Edit ~/daytrader/docker-compose.yml: change `:main` to `:sha-abcdef1`
python scripts/deploy_remote.py --cat ~/daytrader/docker-compose.yml > /tmp/dc.yml
# edit locally
python scripts/deploy_remote.py --put /tmp/dc.yml ~/daytrader/docker-compose.yml
python scripts/deploy_remote.py "cd ~/daytrader && docker compose pull && docker compose up -d"
```

Watchtower only chases `:main`, so a pinned SHA tag stays pinned until
you change it back.

## Backups

The `backup` service (Postgres-alpine) wakes hourly, runs a `pg_dump`
at 02:00 host time, gzips into `~/daytrader/backups/`, prunes anything
older than 14 days. Pull a snapshot to your laptop with:

```bash
python scripts/deploy_remote.py "ls -1t ~/daytrader/backups/ | head -3"
# pick the file you want, then:
python scripts/deploy_remote.py --cat ~/daytrader/backups/daytrader-YYYYMMDD-HHMM.sql.gz > /tmp/backup.sql.gz
```

Restore on a fresh host:
```bash
gunzip -c /tmp/backup.sql.gz | docker exec -i daytrader-db psql -U daytrader daytrader
```

## Troubleshooting

- **Build fails on `pytest`**: tests run before image push. Fix locally first.
- **`denied: requested access to the resource is denied`** on the host pulling: re-login to ghcr.io with a fresh PAT (the cashflow stack uses the same login).
- **App container restart-loop with `alembic` error**: a migration broke. Container won't serve traffic until the schema matches what the new code expects. Check `docker logs daytrader-app`, fix the migration on the laptop, push a new commit.
- **Watchtower not updating**: confirm the `com.centurylinklabs.watchtower.enable=true` label on the app, and that `~/.docker/config.json` is mounted into the watchtower container.
- **`POSTGRES_PASSWORD environment variable is not set`**: the `.env` file isn't readable by the docker daemon. Check `chmod 600 ~/daytrader/.env` and that the file actually exists in `~/daytrader/`.
