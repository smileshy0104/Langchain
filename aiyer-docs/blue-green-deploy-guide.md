# Seedance Hub Blue-Green Deployment Guide

## Overview

Seedance Hub currently runs as one Supervisor program behind Nginx:

- Deploy directory: `/usr/local/seedance-hub`
- Binary: `/usr/local/seedance-hub/main`
- Environment file: `/usr/local/seedance-hub/.env`
- Current HTTP port: `3030`
- Nginx vhost: `/usr/local/nginx/conf/vhost/seedance-hub.dev.wepcc.com.conf`
- Supervisor config: `/etc/supervisor/conf.d/seedance-hub.conf`

The blue-green layout keeps two independent application directories and switches Nginx between their ports:

```text
/usr/local/seedance-hub/
├── blue/
│   ├── main
│   ├── .env        # PORT=3031
│   └── logs/
├── green/
│   ├── main
│   ├── .env        # PORT=3030
│   └── logs/
├── .active_slot    # green or blue
└── nginx-upstream.conf
```

Defaults:

| Slot | Port | Supervisor program |
|---|---:|---|
| green | 3030 | `seedance-hub-green` |
| blue | 3031 | `seedance-hub-blue` |

The initial active slot is `green` so the first migration keeps the current `3030` backend active.

## One-Time Setup

Copy or pull the project scripts onto the server, then run:

```bash
sudo bash /usr/local/seedance-hub/scripts/blue-green-setup.sh
```

If the scripts are executed from a checked-out repository rather than the deploy directory:

```bash
sudo bash scripts/blue-green-setup.sh \
  --project seedance-hub \
  --deploy-dir /usr/local/seedance-hub \
  --binary-name main \
  --green-port 3030 \
  --blue-port 3031 \
  --active-slot green
```

The setup script:

1. Creates `blue/` and `green/` slot directories.
2. Copies the current `main` binary and `.env` into both slots.
3. Sets `green/.env` to `PORT=3030`.
4. Sets `blue/.env` to `PORT=3031`.
5. Writes `/etc/supervisor/conf.d/seedance-hub-bluegreen.conf`.
6. Writes `/usr/local/seedance-hub/nginx-upstream.conf`.
7. Writes `/usr/local/seedance-hub/.active_slot`.

The script never writes real database or Redis credentials into the repository. It only copies the server's existing `.env` locally on the server.

## Nginx Change

Edit `/usr/local/nginx/conf/vhost/seedance-hub.dev.wepcc.com.conf`.

Add the include inside the `server` block:

```nginx
include /usr/local/seedance-hub/nginx-upstream.conf;
```

Change the backend proxy target from the fixed port:

```nginx
proxy_pass http://127.0.0.1:3030;
```

to:

```nginx
proxy_pass http://127.0.0.1:$active_port;
```

Keep the existing streaming and proxy headers as-is.

Test and reload:

```bash
sudo /usr/local/nginx/sbin/nginx -t
sudo /usr/local/nginx/sbin/nginx -s reload
```

## Supervisor Migration

Stop the old single-instance program after the blue-green config is generated:

```bash
sudo supervisorctl stop seedance-hub
sudo supervisorctl reread
sudo supervisorctl update
```

If both new programs start correctly, remove or disable the old config:

```bash
sudo mv /etc/supervisor/conf.d/seedance-hub.conf /etc/supervisor/conf.d/seedance-hub.conf.bak
sudo supervisorctl reread
sudo supervisorctl update
```

Check status:

```bash
sudo supervisorctl status seedance-hub-green seedance-hub-blue
```

## Deploy A New Binary

Build the new binary as `main`, then deploy it:

```bash
bash scripts/blue-green-deploy.sh \
  --project seedance-hub \
  --deploy-dir /usr/local/seedance-hub \
  --binary ./bin/main \
  --binary-name main \
  --green-port 3030 \
  --blue-port 3031 \
  --health-path /health \
  --health-timeout 30
```

Deployment flow:

1. Read `.active_slot`.
2. Copy the new binary into the inactive slot.
3. Restart only the inactive Supervisor program.
4. Check `http://localhost:<inactive-port>/health`.
5. If healthy, update `nginx-upstream.conf`.
6. Run `nginx -t`.
7. Reload Nginx.
8. Update `.active_slot`.

If health checks fail, traffic remains on the old active slot.

## Manual Switch Or Rollback

Switch to green:

```bash
bash scripts/blue-green-switch.sh \
  --project seedance-hub \
  --deploy-dir /usr/local/seedance-hub \
  --port 3030
```

Switch to blue:

```bash
bash scripts/blue-green-switch.sh \
  --project seedance-hub \
  --deploy-dir /usr/local/seedance-hub \
  --port 3031
```

Emergency switch without health check:

```bash
bash scripts/blue-green-switch.sh \
  --project seedance-hub \
  --deploy-dir /usr/local/seedance-hub \
  --port 3030 \
  --skip-health
```

## Sudoers For CI

The dev workflow runs on a self-hosted GitHub Actions runner with labels:

```yaml
runs-on: [self-hosted, dev]
```

If the server runner uses a different label, either add the `dev` label to the runner in GitHub or update `.github/workflows/deploy-dev.yml`.

The production workflow runs on:

```yaml
runs-on: [self-hosted, prod]
```

For the dev deployment, the runner deploys on the same server and calls `scripts/blue-green-deploy.sh` directly. The runner user needs write access to:

```text
/usr/local/seedance-hub/blue
/usr/local/seedance-hub/green
/usr/local/seedance-hub/.active_slot
/usr/local/seedance-hub/nginx-upstream.conf
```

If the runner user is `github-runner`, set ownership after setup:

```bash
sudo chown -R github-runner:github-runner /usr/local/seedance-hub/blue /usr/local/seedance-hub/green
sudo chown github-runner:github-runner /usr/local/seedance-hub/.active_slot /usr/local/seedance-hub/nginx-upstream.conf
```

If a CI runner deploys without a root shell, allow only the required commands:

```text
github-runner ALL=(ALL) NOPASSWD: /usr/bin/supervisorctl restart seedance-hub-blue
github-runner ALL=(ALL) NOPASSWD: /usr/bin/supervisorctl restart seedance-hub-green
github-runner ALL=(ALL) NOPASSWD: /usr/local/nginx/sbin/nginx -t
github-runner ALL=(ALL) NOPASSWD: /usr/local/nginx/sbin/nginx -s reload
```

Adjust `github-runner` if the server uses a different deploy user.

Edit sudoers with:

```bash
sudo visudo -f /etc/sudoers.d/github-runner
```

Validate the runner can run the required commands:

```bash
sudo -u github-runner sudo /usr/bin/supervisorctl restart seedance-hub-blue
sudo -u github-runner sudo /usr/bin/supervisorctl restart seedance-hub-green
sudo -u github-runner sudo /usr/local/nginx/sbin/nginx -t
sudo -u github-runner sudo /usr/local/nginx/sbin/nginx -s reload
```

The dev workflow optionally reads this repository secret:

| Secret | Purpose | Default when unset |
|---|---|---|
| `SEEDANCE_HUB_DEPLOY_PATH` | Deployment directory used by `.github/workflows/deploy-dev.yml` | `/usr/local/seedance-hub` |

The production workflow asks for `target_host` at dispatch time. The prod runner must already have an SSH config entry or reachable hostname for that value.

## Checks

```bash
cat /usr/local/seedance-hub/.active_slot
cat /usr/local/seedance-hub/nginx-upstream.conf
curl -s http://localhost:3030/health
curl -s http://localhost:3031/health
curl -s http://localhost:31002/health
sudo supervisorctl status seedance-hub-green seedance-hub-blue
```

Expected health response:

```json
{"code":200,"data":{"status":"ok"},"message":"ok"}
```

JSON object key order may differ.
