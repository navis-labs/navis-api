# Deploying Navis to AWS — console runbook

This is the step-by-step for standing up **staging** and **production** on AWS ECS Fargate
using the AWS Console, with **Neon** as the database.

Follow it top to bottom, once per environment. Every value you must choose is named
explicitly, and every value you must **record** is marked `-> RECORD`. Keep a scratch file
open; the recorded values are what you paste into GitHub at the end.

Why a console runbook and not Terraform: the console is faster for the first environment and
needs no new tooling. The cost is that nothing stops staging and production from drifting
apart. This document is the mitigation — if you fill in the "Recorded values" table at the
end, you have a written description of both environments, and it doubles as the blueprint if
you later move to Terraform (`terraform import`).

---

## What you need before starting

- An AWS account with admin access. This runbook uses **`ap-southeast-1` (Singapore)**.

  That is a deliberate choice and not the obvious one, since the team and its users are in
  India. **Neon has no Mumbai region** — the closest it offers is Singapore — and measuring
  from an `ap-south-1` CloudShell gave a consistent **~64 ms** round trip to the Singapore
  endpoint (66.2 / 64.8 / 62.6 ms on `SELECT 1`).

  A single API request here issues many database queries, while the user pays the trip to the
  load balancer only once, so the arithmetic is decisive:

  | | AWS Mumbai | AWS Singapore |
  |---|---|---|
  | User → ALB | ~10 ms | ~65 ms |
  | Each query | ~64 ms | ~2 ms |
  | A 15-query request | **~970 ms** | **~95 ms** |

  Put the compute next to the database, not next to the user. This also explains the
  `SELECT 1` taking 1457 ms that is recorded in `db_pool_pre_ping`'s comment in
  `navis/platform/config/settings.py` — that was this same problem, unmeasured.

  Revisit only if Neon opens a Mumbai region.
- A Neon account.
- Admin access to the `navis-labs/navis-api` GitHub repository.
- A domain you control, if you want HTTPS (required for production OAuth).

Budget expectation, per month, at pre-launch scale with no users:

| | Staging | Production |
|---|---|---|
| ECS Fargate (Spot for staging) | $7 | $24 |
| Application Load Balancer | $18 | $18 |
| ElastiCache Redis (`cache.t4g.micro`) | $13 | $13 |
| ECR + CloudWatch Logs | $3 | $6 |
| Neon | $0 (free tier) | ~$19 (Launch) |
| **Total** | **~$41** | **~$80** |

These are approximate `ap-southeast-1` figures — verify in the AWS Pricing Calculator.
**Container and load-balancer charges are per hour the resource exists, not per request.**
An idle environment costs very nearly the same as a busy one. LLM provider spend (Azure
OpenAI / Anthropic) is usage-based and separate from all of the above.

There is no RDS line item because the database is Neon. There is no NAT gateway line item
because tasks run in public subnets — see "Why public subnets" in Part 2.

---

## Part 0 — Code changes required before deploying

These are small, but the deploy depends on them. Apply them, run the suite, and merge to
`development` before continuing.

### 0.1 `/ready` must return 503 when degraded

Today `/ready` returns HTTP 200 even when Postgres or Redis is unreachable, because the
handler returns a plain dict and FastAPI defaults to 200. An ALB health check pointed at it
would therefore **never** mark a task unhealthy, and a task whose database connection had
died would stay in service forever.

In `navis/main.py`:

```python
from fastapi import FastAPI, Response   # add Response to the existing import

    @app.get("/ready", tags=["platform"])
    async def ready(response: Response) -> dict:
        """Readiness: Postgres + Redis reachable. For LB/k8s gating."""
        result = await readiness()
        if result["status"] != "ready":
            response.status_code = 503
        return result
```

This also fixes the `docker-compose.yml` healthcheck, which is broken for the same reason.

### 0.2 Dockerfile — drop the build toolchain, add a non-root user

`build-essential` and `libpq-dev` are not needed: `asyncpg`, `bcrypt`, `cryptography` and
`greenlet` all ship manylinux wheels, and nothing in the project uses `psycopg`. Verified by
building both variants:

| | Current | After |
|---|---|---|
| Image size | 1.04 GB | 605 MB |
| Runs as | root | `uid=10001(navis)` |

435 MB smaller means faster ECR pushes, faster ECS cold starts and less ECR storage.

```dockerfile
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# No build-essential / libpq-dev: asyncpg, bcrypt, cryptography and greenlet all
# ship manylinux wheels, and nothing here uses psycopg. Removing them takes the
# image from 1.04GB to 605MB.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

RUN useradd --create-home --uid 10001 navis && chown -R navis:navis /app
USER navis

EXPOSE 8000

# This image serves THREE roles; the default command is only one of them.
#
#   api     CMD as below
#   worker  celery -A navis.platform.queue.celery_app:celery_app worker --pool=prefork
#   beat    celery -A navis.platform.queue.celery_app:celery_app beat
#
# Running the API alone ships a product where Gmail never becomes memory: signals
# and their outbox pointers are written and nothing ever relays them. Deploy all
# three with a restart policy — see docker-compose.yml / Procfile.
CMD ["uvicorn", "navis.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 0.3 Architecture guard tests read source files without an encoding

Three guard tests call `pathlib.read_text()` with no `encoding=`. On Windows the default is
cp1252, which crashes on the em-dash in the source, so **the guards silently do not run on a
Windows development machine**. They pass on Linux, so CI is unaffected — but fix them so
local runs mean something:

```python
path.read_text(encoding="utf-8")
```

in `tests/intelligence/test_intelligence_v2.py` and
`tests/execution/test_integration_boundary.py`.

### 0.4 Decide on Sentry

`navis/platform/monitoring/sentry.py` imports `sentry_sdk` optionally, but `sentry-sdk` is
**not in `requirements.txt`**. If you set `SENTRY_DSN` in production as things stand, you get
one warning line in the logs and no error reporting at all.

Either add `sentry-sdk` to `requirements.txt`, or do not set `SENTRY_DSN`.

---

## Part 1 — Neon

You need two isolated databases. Use **two separate Neon projects**, not two branches of one
project: branches share the parent project's settings and lifecycle, and you want production
unaffected by anything you do in staging.

### 1.1 Create the projects

1. Neon console -> **New Project**.
2. Name: `navis-staging`.
3. **Postgres version: 16.** Not the newest on offer. CI tests against
   `pgvector/pgvector:pg16` and local development runs pg16, and a production database on a
   major version CI never exercises is a blind spot. If you raise this, raise the CI image in
   the same change.
4. **Region: AWS Asia Pacific 1 (Singapore)** — the same region the AWS infrastructure runs
   in. See the region note at the top of this document for why that is Singapore and not
   Mumbai.
5. Services: enable **Postgres database** only. Object storage, Functions, AI Gateway and
   Neon Auth all stay **off** — this project has its own S3 usage, its own provider registry
   for Azure OpenAI, and a complete identity system in `navis/identity/` that Neon Auth would
   only collide with.
6. Database name: `navis`.
7. Repeat for a second project named `navis-production`. Use two separate **projects**, not
   two branches of one: branches share the parent's settings and lifecycle, and production
   should be unaffected by anything done in staging.

> Neon names a project's default branch `production`. That is Neon's own terminology and has
> nothing to do with this deployment's environments — the branch called `production` inside
> the `navis-staging` project **is the staging database**.

> **If you ever change the Neon region, re-measure before building on it.** From an
> `ap-southeast-1` CloudShell, `psql '<direct-url>' -c '\timing' -c 'SELECT 1'` three times
> and read the last two — the first includes connection setup. Same-region should be a few
> milliseconds. Anything above ~50 ms means the compute and the database are not co-located,
> and every API request will pay it many times over.

### 1.2 Enable pgvector

Migration `0038_pgvector_embeddings` runs `CREATE EXTENSION IF NOT EXISTS vector`. In each
project's SQL Editor, run it once up front so the migration is not the first thing to try it:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### 1.3 Record both connection strings

Each Neon project gives you two endpoint styles. **You need both, for different things.**

| Endpoint | Host contains | Use for |
|---|---|---|
| **Pooled** | `-pooler.` | The `api`, `worker` and `beat` services |
| **Direct** | no `-pooler` | The `migrate` task (Alembic) |

The pooled endpoint runs PgBouncer in transaction mode, which cannot hold server-side
prepared statements. The application handles this automatically — `Settings.db_is_pooled`
detects `-pooler` in the URL and sets asyncpg's `statement_cache_size` to zero — but
**Alembic should use the direct endpoint**, because migrations are long-lived DDL
transactions that do not belong on a transaction pooler.

`-> RECORD` for each environment:

- `DATABASE_URL` (pooled), e.g.
  `postgresql://user:pass@ep-xxx-pooler.<region>.aws.neon.tech/neondb?sslmode=require`
- `MIGRATE_DATABASE_URL` (direct), e.g.
  `postgresql://user:pass@ep-xxx.<region>.aws.neon.tech/neondb?sslmode=require`

Keep `?sslmode=require` in the string. The application strips it before handing the URL to
asyncpg and re-applies TLS through `connect_args` — `Settings.db_requires_ssl` keys off both
`sslmode=require` and the `neon.tech` hostname.

### 1.4 Scale-to-zero ON for staging, OFF for production

In each project's **Settings -> Compute**:

- `navis-staging`: leave **Scale to zero** enabled. Staging is idle most of the day and this
  is what keeps it on the free tier. Accept that the first request after idling takes a few
  hundred milliseconds extra while the compute wakes.
- `navis-production`: **disable** scale to zero. A cold start on a user-facing request is not
  a trade worth making, and `beat` touches the database every few seconds anyway, so it would
  rarely sleep.

---

## Part 2 — AWS foundation

### Why public subnets

Neon lives outside your VPC and is reached over the public internet, so every ECS task needs
outbound internet access regardless. The two ways to get it are a NAT gateway (~$45/month
plus per-GB processing) or giving each task a public IP in a public subnet ($0).

This runbook uses **public subnets with public IPs** for both environments. The tasks are
still not reachable from the internet: their security group accepts inbound traffic only from
the load balancer's security group. What a public IP buys is the outbound path to Neon, ECR
and the LLM providers.

Revisit this when you have paying users and a compliance reason to want tasks off the public
internet entirely. Until then a NAT gateway is $45/month for an isolation boundary your
security groups are already providing.

### 2.1 Use the default VPC

The default VPC already has a public subnet in every availability zone, an internet gateway
and correct route tables — exactly the topology this runbook needs, and building it by hand
is a dozen screens of opportunity for a typo.

1. **VPC console -> Your VPCs** -> confirm a VPC marked "Default VPC: Yes" exists in
   `ap-southeast-1`. If someone deleted it, use **Actions -> Create default VPC**.
2. **Subnets** -> filter by that VPC. Pick **two subnets in different availability zones**
   (e.g. `ap-southeast-1a` and `ap-southeast-1b`). An ALB requires at least two AZs.
3. Confirm each has **Auto-assign public IPv4 address = Yes**.

`-> RECORD`: VPC ID (`vpc-…`) and the two subnet IDs (`subnet-…`, `subnet-…`).

These two subnet IDs are what you paste into `migration-subnets` in the deploy workflows.

Staging and production share the default VPC here. If you want hard network separation,
create a second VPC for production — but you would then need a second set of subnets,
security groups and ALB, and the isolation gained is limited while both talk to the same
internet-facing Neon and Redis endpoints anyway.

### 2.2 Security groups

Create **three** per environment, in the default VPC. Naming below is staging; repeat with
`-prod` for production.

**a) `navis-alb-staging`** — the load balancer

| Direction | Type | Port | Source |
|---|---|---|---|
| Inbound | HTTP | 80 | `0.0.0.0/0` |
| Inbound | HTTPS | 443 | `0.0.0.0/0` |
| Outbound | All traffic | All | `0.0.0.0/0` |

**b) `navis-tasks-staging`** — the ECS tasks

| Direction | Type | Port | Source |
|---|---|---|---|
| Inbound | Custom TCP | 8000 | **security group** `navis-alb-staging` |
| Outbound | All traffic | All | `0.0.0.0/0` |

Set the inbound source to the ALB's *security group*, not a CIDR. That is what keeps the
tasks unreachable from the internet despite having public IPs.

**c) `navis-redis-staging`** — ElastiCache

| Direction | Type | Port | Source |
|---|---|---|---|
| Inbound | Custom TCP | 6379 | **security group** `navis-tasks-staging` |

`-> RECORD`: all three security group IDs (`sg-…`).

The tasks security group ID is what you paste into `migration-security-groups` in the deploy
workflows.

---

## Part 3 — ECR

One repository per environment. The deploy workflows already reference these names.

1. **ECR console -> Create repository** (private).
2. Name: `navis-api-staging`. Repeat for `navis-api-prod`.
3. Enable **Scan on push** — free, and it catches known CVEs in the base image.
4. Leave tag immutability **off**: the workflows tag by commit SHA, which is already
   immutable in practice, and turning it on makes re-running a failed deploy on the same
   commit fail at the push step.

### 3.1 Lifecycle policy — do this now, not later

Without one, every deploy leaves an image behind forever. On each repository,
**Lifecycle Policy -> Create rule**:

- Rule priority: `1`
- Description: `keep last 10 images`
- Image status: **Any**
- Match criteria: **Image count more than** `10`
- Action: **Expire**

`-> RECORD`: the registry URI, e.g. `<account-id>.dkr.ecr.ap-southeast-1.amazonaws.com`.

---

## Part 4 — Secrets in SSM Parameter Store

Use **SSM Parameter Store (Standard tier)**, not Secrets Manager. Standard parameters are
free; Secrets Manager charges $0.40 per secret per month, which for ~15 secrets across two
environments is about $12/month for no benefit you need at this stage. ECS injects both
through the same `secrets` block in a task definition.

Create each under **Systems Manager -> Parameter Store -> Create parameter**, type
**SecureString** (default KMS key), under `/navis/staging/…` and `/navis/production/…`.

### 4.1 Generate the key material first

Run these locally and paste the output; do not invent values by hand.

```bash
# SECRET_KEY — must be at least 32 characters or production refuses to boot
python -c "import secrets; print(secrets.token_urlsafe(48))"

# FIELD_ENCRYPTION_KEYS and SIGNALS_TOKEN_KEYS — Fernet keys, one each
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

**Use different values for staging and production.** Both Fernet keys encrypt data at rest —
PII fields and integration OAuth tokens respectively — and both accept a comma-separated
list, newest first, which is how you rotate later without downtime. Losing the production
values means losing the ability to read every encrypted row, so put them in a password
manager as well as in SSM.

### 4.2 The parameters

| Parameter name | Value |
|---|---|
| `/navis/<env>/SECRET_KEY` | generated above |
| `/navis/<env>/DATABASE_URL` | Neon **pooled** URL from 1.3 |
| `/navis/<env>/FIELD_ENCRYPTION_KEYS` | Fernet key |
| `/navis/<env>/SIGNALS_TOKEN_KEYS` | Fernet key |
| `/navis/<env>/GOOGLE_CLIENT_ID` | from Google Cloud Console |
| `/navis/<env>/GOOGLE_CLIENT_SECRET` | from Google Cloud Console |
| `/navis/<env>/SLACK_CLIENT_ID` | from Slack app config |
| `/navis/<env>/SLACK_CLIENT_SECRET` | from Slack app config |
| `/navis/<env>/SLACK_SIGNING_SECRET` | from Slack app config |
| `/navis/<env>/AZURE_OPENAI_API_KEY` | from Azure |
| `/navis/<env>/ANTHROPIC_API_KEY` | if used |
| `/navis/<env>/OPENAI_API_KEY` | if used |
| `/navis/<env>/MIGRATE_DATABASE_URL` | Neon **direct** URL from 1.3 |

`MIGRATE_DATABASE_URL` is referenced only by the `migrate` task definition, where it is
mapped to the container's `DATABASE_URL`.

---

## Part 5 — IAM roles

Three roles. Create the first two separately per environment so a compromised staging task
cannot read production parameters.

### 5.1 Task execution role — `navisTaskExecutionRole-staging`

The role **ECS itself** uses to pull the image and fetch secrets before your container starts.

1. **IAM -> Roles -> Create role**
2. Trusted entity: **AWS service** -> **Elastic Container Service** ->
   **Elastic Container Service Task**
3. Attach the managed policy **`AmazonECSTaskExecutionRolePolicy`**
4. Name it `navisTaskExecutionRole-staging`, create it, then add an **inline policy** named
   `read-navis-ssm`:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["ssm:GetParameters", "ssm:GetParameter"],
      "Resource": "arn:aws:ssm:ap-southeast-1:<ACCOUNT_ID>:parameter/navis/staging/*"
    },
    {
      "Effect": "Allow",
      "Action": ["kms:Decrypt"],
      "Resource": "*",
      "Condition": {
        "StringEquals": { "kms:ViaService": "ssm.ap-southeast-1.amazonaws.com" }
      }
    }
  ]
}
```

Without the `kms:Decrypt` statement, SecureString parameters fail to resolve and the task
stops before your code runs, with an error that does not obviously say why.

### 5.2 Task role — `navisTaskRole-staging`

The role **your application code** runs as. Same trusted entity. It needs no policies unless
you use S3 — if `S3_BUCKET_NAME` is set, attach an inline policy granting
`s3:GetObject`/`s3:PutObject` on that bucket.

Create it empty rather than skipping it: a distinct task role means you can grant the
application a permission later without also granting it to the ECS agent.

### 5.3 GitHub OIDC deploy role — `navisGitHubDeployRole`

This is what lets GitHub Actions deploy with no stored AWS access keys.

1. **IAM -> Identity providers -> Add provider** -> **OpenID Connect**
   - Provider URL: `https://token.actions.githubusercontent.com`
   - Audience: `sts.amazonaws.com`
2. **IAM -> Roles -> Create role** -> **Web identity** -> select that provider and audience.
3. Name: `navisGitHubDeployRole`.
4. Edit its **trust policy** to pin both the repository and the branches:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::<ACCOUNT_ID>:oidc-provider/token.actions.githubusercontent.com"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
        },
        "StringLike": {
          "token.actions.githubusercontent.com:sub": [
            "repo:navis-labs/navis-api:ref:refs/heads/development",
            "repo:navis-labs/navis-api:ref:refs/heads/main",
            "repo:navis-labs/navis-api:environment:staging",
            "repo:navis-labs/navis-api:environment:production"
          ]
        }
      }
    }
  ]
}
```

The `StringLike` list is the security boundary. Without it, **any** repository on GitHub could
assume this role. Do not replace it with `repo:navis-labs/navis-api:*`.

5. Attach an inline policy `navis-deploy`:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage",
        "ecr:InitiateLayerUpload",
        "ecr:UploadLayerPart",
        "ecr:CompleteLayerUpload",
        "ecr:PutImage"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "ecs:DescribeServices",
        "ecs:DescribeTaskDefinition",
        "ecs:DescribeTasks",
        "ecs:ListTasks",
        "ecs:RegisterTaskDefinition",
        "ecs:UpdateService",
        "ecs:RunTask"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": "iam:PassRole",
      "Resource": [
        "arn:aws:iam::<ACCOUNT_ID>:role/navisTaskExecutionRole-*",
        "arn:aws:iam::<ACCOUNT_ID>:role/navisTaskRole-*"
      ]
    }
  ]
}
```

`iam:PassRole` is required and is the one people forget: registering a task definition that
names an execution role counts as passing that role, and the deploy fails at
`RegisterTaskDefinition` without it.

`-> RECORD`: the role ARN, `arn:aws:iam::<ACCOUNT_ID>:role/navisGitHubDeployRole`.

---

## Part 6 — ElastiCache Redis

Redis is the Celery broker, the result backend, the cache and the event stream mirror. It is
not optional — without it no background work runs.

1. **ElastiCache -> Redis OSS caches -> Create**
2. **Design your own cache**, **Cluster mode: disabled**
3. Name: `navis-staging` / `navis-prod`
4. Node type: **`cache.t4g.micro`**
5. **Number of replicas: 0** — a replica doubles the cost and buys failover you do not need
   pre-launch. The queue is rebuildable; the source of truth is Postgres.
6. Subnet group: create one using the **same two subnets** from Part 2
7. Security group: `navis-redis-staging` from Part 2
8. Disable automatic backups for staging; keep 1 day for production
9. **Transit encryption mode: `Preferred`, never `Required`.** See below.

`-> RECORD`: the **Primary endpoint** hostname. For a replication group it is the
`master.` address, e.g. `master.navis-staging.wxbt2i.apse1.cache.amazonaws.com`.

#### Transit encryption mode must be `Preferred`

If in-transit encryption is left on with mode **`Required`**, the cluster drops plain TCP and
every client must speak `rediss://`. The cache and event-bus clients would cope —
`redis.asyncio.from_url` understands `rediss://` — but **Celery would not**: there is no
`broker_use_ssl` configuration in `navis/platform/queue/celery_app.py`, and kombu refuses a
`rediss://` broker URL that carries no `ssl_cert_reqs`, so `worker` and `beat` fail to start
at all.

That failure lands exactly on this product's worst shape: the API comes up healthy, and
nothing behind it runs.

`Preferred` accepts both plain and TLS connections, so the `redis://` URLs below work
unchanged while TLS stays available. Note that the **Enable** checkbox itself cannot be
turned off after the cluster is created — only the mode can be changed — so if you find a
cluster on `Required`, modify it to `Preferred` (Actions -> Modify, apply immediately)
rather than trying to disable encryption.

To move to `Required` later, add `broker_use_ssl` and `redis_backend_use_ssl` to the Celery
config and switch all three URLs to `rediss://` in the same change.

### 6.1 The three Redis URLs — read this carefully

ElastiCache listens on **6379**. Local development uses **6380** (see `docker-compose.yml`),
and the application's defaults point at `localhost:6380`.

You must therefore set **all three** of these in **every** task definition (api, worker and
beat), pointing at the same host and port:

```
REDIS_URL             = redis://<primary-endpoint>:6379/0
CELERY_BROKER_URL     = redis://<primary-endpoint>:6379/2
CELERY_RESULT_BACKEND = redis://<primary-endpoint>:6379/3
```

Setting only `REDIS_URL` is a real and quiet failure mode: `CELERY_BROKER_URL` would keep its
default of `redis://localhost:6380/2`, the worker would try to reach a Redis inside its own
task, find nothing, and the entire pipeline would stall behind a perfectly healthy-looking
API. This project has a test that exists precisely to catch that —
`tests/platform/test_deployment_config.py::test_celery_and_cache_agree_on_the_redis_endpoint`
asserts the broker, the result backend and the cache all resolve to the same host and port.

The database numbers (`/0`, `/2`, `/3`) differ on purpose and are correct as shown.

---

## Part 7 — Load balancer

### 7.1 Target group

1. **EC2 -> Target Groups -> Create target group**
2. Target type: **IP addresses** (required for Fargate `awsvpc` networking)
3. Name: `navis-api-staging-tg`
4. Protocol **HTTP**, port **8000**, the VPC from Part 2
5. Protocol version: **HTTP1**
6. **Health checks**:
   - Path: **`/ready`**
   - Healthy threshold: `2`
   - Unhealthy threshold: `3`
   - Timeout: `10` seconds
   - Interval: `30` seconds
   - Success codes: `200`
7. **Attributes -> Deregistration delay: `30` seconds** (the default 300 makes every deploy
   five minutes slower for no benefit at this scale)

`/ready` checks Postgres and Redis. **This only works if you applied the fix in Part 0.1** —
otherwise `/ready` returns 200 unconditionally and the health check is decorative. If you
have not applied it, use `/health` and understand that no task will ever be replaced for
being unhealthy.

### 7.2 Load balancer

1. **EC2 -> Load Balancers -> Create -> Application Load Balancer**
2. Name: `navis-staging-alb`
3. Scheme: **Internet-facing**, IP type **IPv4**
4. VPC and the **two subnets** from Part 2
5. Security group: `navis-alb-staging`
6. Listener: **HTTP:80** -> forward to `navis-api-staging-tg`

`-> RECORD`: the ALB **DNS name**.

### 7.3 HTTPS

Required for production, because OAuth callbacks from Google and Slack will not accept a
plain HTTP redirect URI.

1. **ACM (in `ap-southeast-1`) -> Request a public certificate** for e.g. `api.navis.app` and
   `staging-api.navis.app`. Validate by DNS.
2. Add an **HTTPS:443** listener on the ALB using that certificate, forwarding to the target
   group.
3. Change the **HTTP:80** listener action to **Redirect to HTTPS** (301).
4. In your DNS provider, point the hostname at the ALB DNS name with an A/ALIAS record.

### 7.4 Block `/metrics` from the internet

`/metrics` has no authentication — it is intended to be restricted at the network layer, and
behind an internet-facing ALB it is world-readable. Add a listener rule:

- Priority: `1`
- Condition: **Path pattern** = `/metrics`
- Action: **Return fixed response**, status `403`

Scrape it from inside the VPC instead when you add a Prometheus agent.

### 7.5 Optional: one ALB for both environments

An ALB is $18/month and is the largest fixed line item in each environment. You can serve both
from one by adding host-header listener rules:

- Host `staging-api.navis.app` -> `navis-api-staging-tg`
- Host `api.navis.app` -> `navis-api-prod-tg` (default rule)

This saves $18/month. The trade-off is that staging and production then share an availability
boundary — a misconfigured listener rule can affect both. Reasonable pre-launch; separate them
once you have users.

---

## Part 8 — ECS

### 8.1 Clusters

**ECS -> Clusters -> Create cluster**, twice:

- `navis-staging`, infrastructure **AWS Fargate (serverless)**
- `navis-prod`, same

Enable **Container Insights** on production only — it is roughly $2/month per environment and
staging does not need it.

### 8.2 Task definitions

You need **four per environment**: `api`, `worker`, `beat` and `migrate`. The family names
below are exactly what `.github/workflows/deploy.yml` expects — it renders
`${ecr-repository}-api`, `-worker`, `-beat` and runs `${ecr-repository}-migrate`. Do not
rename them.

| Family (staging) | Container name | CPU / Memory | Command |
|---|---|---|---|
| `navis-api-staging-api` | `api` | 256 / 512 | *(image default)* |
| `navis-api-staging-worker` | `worker` | 256 / 1024 | see below |
| `navis-api-staging-beat` | `beat` | 256 / 512 | see below |
| `navis-api-staging-migrate` | `migrate` | 256 / 512 | *(overridden at run time)* |

The container name matters as much as the family name — the workflow passes
`container-name: api` / `worker` / `beat` to the render step, and a mismatch fails the deploy.

For production, substitute `navis-api-prod-` and raise `api` to 512 / 1024 once you have real
traffic.

Create each with **ECS -> Task definitions -> Create new task definition**:

- Launch type: **AWS Fargate**
- Operating system: **Linux/X86_64**
- Task role: `navisTaskRole-staging`
- Task execution role: `navisTaskExecutionRole-staging`
- Container image: `<registry>/navis-api-staging:bootstrap` (a placeholder — every deploy
  overwrites it with the commit SHA)
- Port mappings: **8000/tcp** on the `api` task definition only
- Log configuration: **awslogs**, group `/ecs/navis-staging`, region `ap-southeast-1`, stream
  prefix matching the container name

**Commands** (enter as comma-separated values in the console's Command field):

`worker`:
```
celery,-A,navis.platform.queue.celery_app:celery_app,worker,--pool=prefork,--concurrency=2,--loglevel=INFO
```

`beat`:
```
celery,-A,navis.platform.queue.celery_app:celery_app,beat,--loglevel=INFO,--schedule=/tmp/celerybeat-schedule
```

`--pool=prefork` is mandatory and must never be changed to `threads`, `gevent` or `eventlet`:
`run_async` keeps one event loop per process for the worker's lifetime and asyncpg pools bind
to the loop that created them, so a thread pool would drive one loop from several threads and
corrupt it.

**Health check** on the `api` container only:

```
CMD-SHELL, python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/ready', timeout=5).status==200 else 1)"
```

Interval 30s, timeout 10s, retries 3, start period 60s.

### 8.3 Environment variables — every task definition

Set these as plain **Environment variables** in all four task definitions:

| Name | Staging | Production |
|---|---|---|
| `ENVIRONMENT` | `staging` | `production` |
| `DEBUG` | `false` | `false` |
| `SHOW_DOCS` | `true` | **`false`** |
| `LOG_FORMAT` | `json` | `json` |
| `LOG_LEVEL` | `INFO` | `INFO` |
| `RELOAD` | `false` | `false` |
| `CORS_ORIGINS` | your staging frontend origin | your production frontend origin |
| `FRONTEND_URL` | staging frontend URL | production frontend URL |
| `REDIS_URL` | `redis://<endpoint>:6379/0` | same shape |
| `CELERY_BROKER_URL` | `redis://<endpoint>:6379/2` | same shape |
| `CELERY_RESULT_BACKEND` | `redis://<endpoint>:6379/3` | same shape |
| `EXECUTION_INLINE_FALLBACK` | `false` | **`false`** |
| `SIGNALS_REJECT_LEGACY_PLAINTEXT` | `false` | **`true`** |
| `AWS_REGION` | `ap-southeast-1` | `ap-southeast-1` |
| `GOOGLE_REDIRECT_URI` | staging callback URL | production callback URL |
| `GOOGLE_GMAIL_REDIRECT_URI` | staging Gmail callback URL | production equivalent |
| `SLACK_REDIRECT_URI` | staging callback URL | production callback URL |
| `AZURE_OPENAI_ENDPOINT` | your Azure resource URL | same |
| `AZURE_OPENAI_DEPLOYMENT` | your **chat** deployment name | same |
| `AZURE_OPENAI_EMBED_DEPLOYMENT` | your **embedding** deployment name | same |
| `AZURE_OPENAI_API_VERSION` | leave unset to use the default | same |

And these as **Secrets**, with `valueFrom` set to the SSM parameter ARN from Part 4:

`SECRET_KEY`, `DATABASE_URL`, `FIELD_ENCRYPTION_KEYS`, `SIGNALS_TOKEN_KEYS`,
`GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `SLACK_CLIENT_ID`, `SLACK_CLIENT_SECRET`,
`SLACK_SIGNING_SECRET`, `AZURE_OPENAI_API_KEY`.

#### The four Azure variables are a set, and missing one degrades silently

This is the single easiest way to deploy something that looks completely healthy and is
quietly useless, so it is worth being precise about.

`navis/ml/providers/registry.py` selects the ML backend from configuration, and **falls back
to a deterministic hash-based provider whenever the configuration is incomplete**. It does not
raise, does not warn, and nothing in the health checks notices.

- **Generation** uses Azure only if `AZURE_OPENAI_API_KEY` **and** `AZURE_OPENAI_ENDPOINT`
  are both set. Otherwise it tries `ANTHROPIC_API_KEY`, and failing that returns the
  deterministic provider.
- **Embeddings** use Azure only if `AZURE_OPENAI_API_KEY` **and** `AZURE_OPENAI_ENDPOINT`
  **and** `AZURE_OPENAI_EMBED_DEPLOYMENT` are all three set. The embed deployment is required
  separately and on purpose: in Azure, chat and embedding models are different deployments,
  and configuring only the chat one would make every embed call 404.

Setting just `AZURE_OPENAI_API_KEY` — which is the natural thing to do, because it is the
only one that looks like a secret — leaves you with hash-based embeddings. Those are real
vectors of the right shape, so nothing fails: they are simply meaningless, and `/chat`
retrieval returns essentially random results.

It is also not free to fix later. Those vectors are written into `memory_embeddings` tagged
with the deterministic model name, so correcting the configuration does not repair existing
rows — you have to re-embed the corpus. Get these right before the first real data lands.

`AZURE_OPENAI_DEPLOYMENT` and `AZURE_OPENAI_EMBED_DEPLOYMENT` must be the **deployment
names** in your Azure OpenAI resource, not the underlying model names. Azure routes by
deployment. The embed deployment name is also recorded as the `model` column on every
embedding row, so changing it later invalidates the existing index — which the read/write
pin mechanism is designed to make survivable, but only if you do it deliberately.

On the **`migrate`** task definition only, map `DATABASE_URL` to the
`/navis/<env>/MIGRATE_DATABASE_URL` parameter instead, so Alembic uses Neon's direct endpoint.

#### Production will refuse to start if you get this wrong

`Settings._enforce_production_safety` raises at import time — before the process can serve a
single request — when `ENVIRONMENT=production` and any of the following is true:

- `SECRET_KEY` is empty, `change-me`, or shorter than 32 characters
- `DEBUG` is true
- `DATABASE_URL` contains `localhost` or `127.0.0.1`
- `SHOW_DOCS` is true
- `CORS_ORIGINS` contains `*`, or is empty
- `SIGNALS_REJECT_LEGACY_PLAINTEXT` is false
- `EXECUTION_INLINE_FALLBACK` is true
- `GOOGLE_PUBSUB_TOPIC` is set but the audience or the service account identity is not

This is deliberate and it is a feature: a production box that quietly behaved like a
development box is the failure it exists to prevent. If your production tasks crash-loop
immediately after their first deploy, read the CloudWatch log — the exception names every
problem it found.

Note that `SIGNALS_REJECT_LEGACY_PLAINTEXT=true` means any OAuth token still stored as
untagged plaintext will be **rejected on read**. On a brand-new production database there are
no such tokens, so this is safe. If you ever migrate an existing database in, run
`navis.signals.token_backfill.reencrypt_legacy_tokens` first.

### 8.4 Services

Create three per cluster.

**`navis-api-staging`**
- Task definition: `navis-api-staging-api`
- Desired tasks: **1** (production: 2 once you have users, for zero-downtime deploys)
- Networking: the two subnets, security group `navis-tasks-staging`,
  **Public IP: ENABLED** (required — this is the outbound path to Neon and ECR)
- Load balancing: **Application Load Balancer**, existing target group `navis-api-staging-tg`
- Health check grace period: **120** seconds
- Deployment: enable **Deployment circuit breaker** with **Rollback on failure**
- Capacity provider: **FARGATE_SPOT** for staging, **FARGATE** for production

**`navis-worker-staging`**
- Task definition: `navis-api-staging-worker`
- Desired tasks: **1**, no load balancer
- Same networking; **Public IP: ENABLED**
- Capacity provider: FARGATE_SPOT in both environments — a worker interruption is safe,
  because `task_acks_late` redelivers anything in flight

**`navis-beat-staging`**
- Task definition: `navis-api-staging-beat`
- Desired tasks: **1**, no load balancer
- Same networking; **Public IP: ENABLED**
- **Deployment configuration: minimum healthy percent `0`, maximum percent `100`**

That last setting is not optional. `beat` is the scheduler, and **exactly one instance may ever
run**. The defaults (100/200) make ECS start the new task before stopping the old one, so
during every deploy there would briefly be two schedulers and every periodic task — signal
sync, outbox relays, brief generation, watch renewal — would fire twice. Setting 100/0 makes
ECS stop the old task first. Never scale this service above 1.

---

## Part 9 — GitHub configuration

### 9.1 Environments

**Settings -> Environments**, create `staging` and `production`.

On GitHub Free with a private repository, environment protection rules such as required
reviewers are not available — they need Enterprise. This is why `deploy-production.yml` is
`workflow_dispatch`-only with a typed `deploy` confirmation: the deliberate manual action *is*
the approval gate. Do not assume the environment is protecting anything.

For the same reason, branch protection rules do not apply to private repositories on Free, so
CI cannot block a merge. It still runs and reports; only enforcement is missing.

### 9.2 Secrets and variables

In **each** environment:

| Kind | Name | Value |
|---|---|---|
| Secret | `AWS_DEPLOY_ROLE_ARN` | `arn:aws:iam::806168576303:role/navisGitHubDeployRole` (the role from 5.3; the same role serves both environments) |
| Variable | `HEALTHCHECK_URL` | staging: `https://staging-api.navislabs.in/ready` — production: `https://api.navislabs.in/ready` |

The health path is `/ready`, not `/health`: `/ready` is the one that actually checks the
database and Redis, and it is the same path the ALB target group polls.

If `HEALTHCHECK_URL` is unset the deploy's smoke test prints a warning and passes without
checking anything, so set it.

### 9.3 Fill in the workflow placeholders

**Staging is already filled in** — `deploy-staging.yml` carries the real subnets, security
group and service names, and they are in the Recorded values table at the end of this
document. Production still has the placeholders:

```yaml
migration-subnets: subnet-REPLACE_ME_A,subnet-REPLACE_ME_B
migration-security-groups: sg-REPLACE_ME
```

Replace them with the two subnet IDs from Part 2.1 and the **tasks** security group from
Part 2.2 when production is built.

One thing that is easy to get wrong: the ECS **service** names are not always what you would
guess. When a service is created from the console without an explicit name, the console
appends a random suffix — staging's worker and beat are
`navis-api-staging-worker-service-i49pghsm` and `navis-api-staging-beat-service-hpdb0ws2`,
while the api service is plain `navis-api-staging-api-service`. That suffix is part of the
name. Read the real names with:

```bash
aws ecs list-services --cluster navis-staging --region ap-southeast-1 --output text
```

A wrong service name does not fail early: migrations run and the api deploys, and only then
does the worker step fail with `ServiceNotFoundException`, leaving api on new code and
worker and beat on old.

Both environments run in public subnets, so both need `assign-public-ip: ENABLED`. **This is
already set in both workflow files** — staging had it, and production has it now. You do not
need to change it; this section explains why it is there.

The input defaults to `DISABLED`, which is correct only for private subnets with a NAT
gateway. Left at the default, the migration task can never pull its image: it sits there
until the job times out, with no error that explains why.

When production later moves to private subnets behind a NAT instance, **delete the
`assign-public-ip` line** from `deploy-production.yml` rather than setting it to `DISABLED`,
so the safer default applies again.

---

## Part 10 — First deploy

The deploy workflows expect the ECS services to already exist, so bootstrap once by hand.

1. Build and push a first image locally:

```bash
aws ecr get-login-password --region ap-southeast-1 \
  | docker login --username AWS --password-stdin <registry>

docker build -t <registry>/navis-api-staging:bootstrap .
docker push <registry>/navis-api-staging:bootstrap
```

2. Run the migration once from the ECS console: **Cluster -> Tasks -> Run new task**, task
   definition `navis-api-staging-migrate`, launch type FARGATE, your two subnets, the tasks
   security group, **public IP enabled**, and override the command to `alembic,upgrade,head`.
   Wait for it to stop and confirm **exit code 0**.

3. Confirm the three services reach a steady state, then:

```bash
curl -i https://staging-api.navis.app/health
curl -i https://staging-api.navis.app/ready
```

`/ready` must return **200** with `"database": true` and `"redis": true`.

4. Now merge to `development` and let the pipeline take over. From here every merge deploys
   staging automatically; production is the manual `Deploy production` workflow run against
   `main` with `deploy` typed into the confirmation box.

---

## Part 11 — Verify the pipeline actually runs

A healthy API proves almost nothing about this product. The pipeline is
`signals -> memory -> state -> reasoning -> decisions`, and it advances only if `worker` and
`beat` are both alive. Check all of these before calling the environment good:

- [ ] `/ready` returns 200 with both dependencies true
- [ ] ECS shows **three** services with running count 1 (api, worker, beat)
- [ ] `beat` running count is **exactly 1**
- [ ] Worker logs show periodic task receipts (`relay-signal-outbox` fires every 5 seconds)
- [ ] Beat logs show the schedule loaded, with no `Event loop is closed` errors
- [ ] Connect a Gmail account, then confirm rows appear in `signal_outbox` **and** that
      `published_at` becomes non-null within ~10 seconds
- [ ] Confirm `memory_records` grows after that
- [ ] Boot log does **not** contain `ENVIRONMENT=… the following are NOT enforced` in
      production
- [ ] Boot log does **not** contain `embedding pins SPLIT`
- [ ] **The real ML provider is live, not the deterministic fallback.** Run this in the
      container (ECS Exec, or locally with the same environment) and confirm it prints
      `AzureProvider` twice, not `DeterministicProvider`:

      ```bash
      python -c "
      from navis.ml.providers.registry import get_provider, get_embedding_provider
      print('generation:', type(get_provider()).__name__)
      print('embeddings:', type(get_embedding_provider()).__name__)
      "
      ```

      Nothing else in this checklist catches a silent fallback — see Part 8.3.

The sixth item is the one that matters most. An API that accepts writes while nothing relays
them into memory is this product's documented outage mode, and it presents as a perfectly
healthy API.

---

## Part 12 — Alarms

Create these in CloudWatch. The worker alarm is the important one.

| Alarm | Metric | Threshold |
|---|---|---|
| **Worker is dead** | ECS `RunningTaskCount`, service `navis-worker-prod` | `< 1` for 5 min |
| **Beat is dead** | ECS `RunningTaskCount`, service `navis-beat-prod` | `< 1` for 5 min |
| **Beat doubled** | same | `> 1` for 5 min |
| API 5xx | ALB `HTTPCode_Target_5XX_Count` | `> 10` in 5 min |
| Unhealthy targets | ALB `UnHealthyHostCount` | `>= 1` for 5 min |
| API CPU | ECS `CPUUtilization` | `> 80%` for 10 min |

Send them to an SNS topic with your email subscribed. Neon has its own alerting for database
metrics — configure it in the Neon console rather than CloudWatch.

Also set **CloudWatch Logs retention** on `/ecs/navis-staging` and `/ecs/navis-prod` to
**14 days**. The default is "Never expire", and log storage is a cost that compounds silently.

---

## Part 13 — Cost controls

Ranked by how much they save:

1. **No NAT gateway** — public subnets with tight security groups. Saves ~$45/month per
   environment. Already assumed throughout this runbook.
2. **SSM Parameter Store over Secrets Manager** — saves ~$12/month.
3. **Fargate Spot** for staging and for production workers — roughly 70% off those tasks.
4. **One ALB for both environments** via host-header rules (Part 7.5) — saves $18/month.
5. **Neon scale-to-zero on staging** — keeps staging on the free tier.
6. **ECR lifecycle policy** and **14-day log retention** — small individually, but both grow
   without bound if skipped.
7. **ARM64 / Graviton** — Fargate ARM is roughly 20% cheaper and every dependency here has an
   `aarch64` wheel. Requires building with `--platform linux/arm64` and setting the task
   definitions to `ARM64`. Worth doing once things are stable; do not change it mid-bringup.

Do **not** buy a Compute Savings Plan yet. It locks in a one- or three-year commitment, and
you do not yet know what size you need.

---

## Recorded values

Fill these in as you go. This table is the description of your infrastructure.

Staging was built on 2026-09-22. Production is not built yet.

The VPC and its subnets are the account's **default** VPC, so production will share them
unless it is given its own; only the security groups, and everything from Part 3 onward, are
per-environment.

| | Staging | Production |
|---|---|---|
| AWS region | `ap-southeast-1` | `ap-southeast-1` |
| VPC ID | `vpc-038f84f74e0351b35` | *(same default VPC)* |
| Internet gateway | `igw-0ccaba32a28f13dd0` | *(same)* |
| Subnet A (`ap-southeast-1a`) | `subnet-0a5bd7ad312ef9acd` | *(same)* |
| Subnet B (`ap-southeast-1b`) | `subnet-05fc7cbb8c61349d0` | *(same)* |
| SG — ALB | `sg-0edfa7d9001b00eb2` | |
| SG — tasks | `sg-058b623824f3bcec9` | |
| SG — redis | `sg-024c25dedf8d81fc3` | |
| ECR registry | `806168576303.dkr.ecr.ap-southeast-1.amazonaws.com` | *(same account)* |
| ECR repository | `navis-api-staging` | `navis-api-prod` |
| ECS cluster | `navis-staging` | `navis-prod` |
| ECS services | `navis-api-staging-api-service`, `navis-api-staging-worker-service-i49pghsm`, `navis-api-staging-beat-service-hpdb0ws2` | |
| Redis primary endpoint | `master.navis-staging.wxbt2i.apse1.cache.amazonaws.com:6379` | |
| Redis transit encryption | `preferred` — **not `required`** | `preferred` |
| ALB DNS name | `navis-staging-alb-1782157424.ap-southeast-1.elb.amazonaws.com` | |
| Target group | `navis-api-staging-tg` (ip, 8000, `/ready`) | |
| ACM certificate | `navislabs.in` + `*.navislabs.in` | *(same cert)* |
| Neon project | `navis-staging` | `navis-production` |
| Neon Postgres version | 16 | 16 |
| Neon region | AWS Asia Pacific 1 (Singapore) | AWS Asia Pacific 1 (Singapore) |
| Neon database / role | `navis` / `navis_owner` | |
| Neon pooled host | `ep-spring-tooth-b3qn525i-pooler.c-4.ap-southeast-1.aws.neon.tech` | |
| Neon direct host | `ep-spring-tooth-b3qn525i.c-4.ap-southeast-1.aws.neon.tech` | |
| Public hostname | `staging-api.navislabs.in` | `api.navislabs.in` |
| Task execution role | `navisTaskExecutionRole-staging` | `navisTaskExecutionRole-production` |
| Task role | `navisTaskRole-staging` | `navisTaskRole-production` |
| Deploy role ARN | `arn:aws:iam::806168576303:role/navisGitHubDeployRole` | *(same role, both environments)* |
| OIDC provider | `arn:aws:iam::806168576303:oidc-provider/token.actions.githubusercontent.com` | *(same)* |

The two values the deploy workflows need from this table are
`migration-subnets` (both subnet IDs, comma-separated) and `migration-security-groups`
(the **tasks** security group).

---

## Troubleshooting

**Task stops immediately, no application logs.** The execution role could not pull the image
or resolve a secret. Check `stoppedReason` on the stopped task. Almost always a missing
`kms:Decrypt` (Part 5.1) or a wrong SSM parameter ARN.

**Migration task never starts.** Almost always it cannot reach ECR. Confirm
`assign-public-ip: ENABLED` in the workflow and that the subnet auto-assigns public IPs. The
deploy step now reads ECS's `failures` array and prints the reason, so check the step log
before guessing.

**Migration step gives up after 20 minutes.** That is the `migration-timeout-minutes` input
on `deploy.yml`. The task may still be running and may still commit, so **check it before
redeploying — the schema may already have moved**. This repo has 47 migrations, and on a
first run against a distant Neon region each one pays the full round-trip latency, so raise
the input rather than re-running blindly.

**Production crash-loops on boot with "Insecure production config".** Expected and working as
designed — the exception lists every offending setting. See Part 8.3.

**API is healthy but nothing enters memory.** The worker is not running, or it is pointed at
the wrong Redis. Check that all three Redis variables are set in the worker task definition
(Part 6.1) and that `RunningTaskCount` for the worker service is 1.

**`prepared statement "__asyncpg_..." does not exist`.** The application is on Neon's pooled
endpoint but did not detect it. The host must contain `-pooler`; `Settings.db_is_pooled` keys
off that substring.

**Every periodic task fires twice.** Two `beat` tasks are running. Set the beat service's
deployment configuration to minimum healthy 0 / maximum 100 (Part 8.4) and confirm desired
count is 1.

**Deploy fails at `RegisterTaskDefinition` with an access-denied error.** The deploy role is
missing `iam:PassRole` (Part 5.3).
