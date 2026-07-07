"""End-to-end golden-path verification for Navis Identity APIs.

Run against a live server (local dev or staging) to verify the full auth lifecycle
works as designed BEFORE merging a release. Each step asserts the contract; a
single failure surfaces the broken seam.

Usage:
    python tools/identity/golden_path.py
    python tools/identity/golden_path.py --base-url https://staging.navis.example
    python tools/identity/golden_path.py --skip-throttle    # if Redis is down

Exit codes:
    0  every step passed
    1  one or more steps failed (see PASS/FAIL summary)
    2  the server is unreachable / readiness check failed
"""
from __future__ import annotations

import argparse
import sys
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx

# A password that satisfies the schema: >=8 chars, not in weak blocklist, not all
# same character. Same password reused for every signup/login in this run.
_TEST_PASSWORD = "GoldenPath!2026-Run"
# Server-side weak-password blocklist sample (must 422); see identity/schemas.py.
_WEAK_PASSWORD = "password"


@dataclass
class Step:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class Run:
    base_url: str
    skip_throttle: bool = False
    steps: list[Step] = field(default_factory=list)
    access: str = ""
    refresh: str = ""
    email: str = ""

    def record(self, name: str, ok: bool, detail: str = "") -> bool:
        self.steps.append(Step(name, ok, detail))
        mark = "PASS" if ok else "FAIL"
        line = f"  [{mark}] {name}"
        if detail:
            line += f"  -- {detail}"
        print(line)
        return ok


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def check_readiness(client: httpx.Client, run: Run) -> bool:
    """Server reachable AND Postgres up. Redis may be down (throttle fails open by
    design); when that happens we flip skip_throttle on so the lock test self-skips.
    Generous timeout because /ready waits on the slowest dependency probe — a missing
    Redis takes ~6s to fail."""
    try:
        r = client.get("/ready", timeout=15.0)
    except httpx.RequestError as exc:
        return run.record("readiness", False, f"server unreachable: {exc}")
    if r.status_code != 200:
        return run.record("readiness", False, f"status={r.status_code}")
    body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    checks = body.get("checks") or {}
    db_ok = bool(checks.get("database", True))
    redis_ok = bool(checks.get("redis", True))
    if not db_ok:
        return run.record("readiness", False, "database probe failed")
    if not redis_ok and not run.skip_throttle:
        run.skip_throttle = True
        return run.record("readiness", True, "DB up, Redis DOWN -> throttle test will skip")
    return run.record("readiness", True, f"checks={checks}")


def signup_new_user(client: httpx.Client, run: Run) -> bool:
    """201 + access/refresh tokens issued + tokens decodeable shape."""
    run.email = f"goldenpath+{uuid.uuid4().hex[:8]}@navis-test.example.com"
    body = {
        "email": run.email,
        "password": _TEST_PASSWORD,
        "full_name": "Golden Path",
        "organization_name": "Golden Path Co",
    }
    r = client.post("/api/v1/auth/signup", json=body)
    if r.status_code != 201:
        return run.record("signup new user", False, f"status={r.status_code} body={r.text[:200]}")
    data = r.json()
    if not data.get("access_token") or not data.get("refresh_token"):
        return run.record("signup new user", False, "missing access/refresh in response")
    run.access, run.refresh = data["access_token"], data["refresh_token"]
    return run.record("signup new user", True, f"email={run.email}")


def signup_duplicate_email_rejected(client: httpx.Client, run: Run) -> bool:
    """A second signup with the same email must be a clean 400 (not a 500)."""
    body = {
        "email": run.email, "password": _TEST_PASSWORD,
        "full_name": "Dup", "organization_name": "Dup Co",
    }
    r = client.post("/api/v1/auth/signup", json=body)
    return run.record(
        "signup duplicate email -> 400",
        r.status_code == 400,
        f"status={r.status_code}",
    )


def signup_weak_password_rejected(client: httpx.Client, run: Run) -> bool:
    """Weak-password blocklist enforced at the schema boundary (422 validation)."""
    body = {
        "email": f"weak+{uuid.uuid4().hex[:8]}@navis-test.example.com",
        "password": _WEAK_PASSWORD, "full_name": "Weak", "organization_name": "Weak Co",
    }
    r = client.post("/api/v1/auth/signup", json=body)
    return run.record(
        "weak password -> 422",
        r.status_code == 422,
        f"status={r.status_code}",
    )


def login_correct_credentials(client: httpx.Client, run: Run) -> bool:
    r = client.post(
        "/api/v1/auth/login",
        json={"email": run.email, "password": _TEST_PASSWORD},
    )
    if r.status_code != 200:
        return run.record("login (correct creds)", False, f"status={r.status_code}")
    data = r.json()
    ok = bool(data.get("access_token")) and bool(data.get("refresh_token"))
    if ok:
        run.access, run.refresh = data["access_token"], data["refresh_token"]
    return run.record("login (correct creds)", ok)


def me_returns_user_and_org(client: httpx.Client, run: Run) -> bool:
    r = client.get("/api/v1/auth/me", headers=_auth_headers(run.access))
    if r.status_code != 200:
        return run.record("/me returns 200", False, f"status={r.status_code}")
    data = r.json()
    keys_ok = {"user", "organization", "role"} <= set(data.keys())
    email_matches = (data.get("user") or {}).get("email") == run.email
    return run.record(
        "/me returns user + org",
        keys_ok and email_matches,
        "" if keys_ok and email_matches else f"keys_ok={keys_ok} email_match={email_matches}",
    )


def refresh_rotates_tokens(client: httpx.Client, run: Run) -> bool:
    """Refresh issues a NEW token pair and the old refresh is now invalidated."""
    old_refresh = run.refresh
    r = client.post("/api/v1/auth/refresh", json={"refresh_token": old_refresh})
    if r.status_code != 200:
        return run.record("refresh rotates tokens", False, f"status={r.status_code}")
    data = r.json()
    new_refresh = data.get("refresh_token", "")
    if not new_refresh or new_refresh == old_refresh:
        return run.record("refresh rotates tokens", False, "new refresh missing or identical")
    run.access, run.refresh = data["access_token"], new_refresh

    # Replay the OLD refresh — must fail (reuse detection).
    r2 = client.post("/api/v1/auth/refresh", json={"refresh_token": old_refresh})
    if r2.status_code != 401:
        return run.record(
            "refresh replay rejected",
            False,
            f"old-refresh reuse returned status={r2.status_code} (expected 401)",
        )
    run.record("refresh replay rejected", True, "old refresh -> 401")
    return True


def logout_revokes_session(client: httpx.Client, run: Run) -> bool:
    r = client.post("/api/v1/auth/logout", headers=_auth_headers(run.access))
    if r.status_code != 200:
        return run.record("logout 200", False, f"status={r.status_code}")
    run.record("logout 200", True)
    # After logout, the refresh token bound to that session must no longer mint tokens.
    r2 = client.post("/api/v1/auth/refresh", json={"refresh_token": run.refresh})
    return run.record(
        "refresh after logout -> 401",
        r2.status_code == 401,
        f"status={r2.status_code}",
    )


def login_throttle_locks_after_max_attempts(client: httpx.Client, run: Run) -> bool:
    """6 wrong passwords for the SAME email → the account is locked even with the
    correct password (IDENTITY-003). Skips on --skip-throttle when Redis is down."""
    if run.skip_throttle:
        return run.record("login throttle locks", True, "SKIPPED (Redis assumed unavailable)")
    # Make 6 attempts with bad password (default lock threshold is 5).
    for i in range(6):
        client.post(
            "/api/v1/auth/login",
            json={"email": run.email, "password": "WrongPassword!2026"},
        )
    # Now even the correct password should be rejected.
    r = client.post(
        "/api/v1/auth/login",
        json={"email": run.email, "password": _TEST_PASSWORD},
    )
    return run.record(
        "login locked after 6 fails",
        r.status_code == 401,
        f"status={r.status_code}",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument(
        "--skip-throttle",
        action="store_true",
        help="Skip the brute-force lock check (use when Redis is unavailable).",
    )
    args = parser.parse_args()

    run = Run(base_url=args.base_url, skip_throttle=args.skip_throttle)
    print(f"Navis Identity golden-path against {run.base_url}\n")

    # trust_env=False ignores HTTP_PROXY / HTTPS_PROXY env vars — VPN/corporate
    # proxies cannot route to localhost and silently time out otherwise.
    # 60s ceiling because a missing Redis makes event publish wait on the bus's
    # 15s handler timeout PLUS readiness/mirror probes (~6s each); even with Redis
    # up, the safety margin absorbs uvicorn --reload jitter without flapping.
    with httpx.Client(base_url=run.base_url, timeout=60.0, trust_env=False) as client:
        if not check_readiness(client, run):
            print("\nReadiness FAILED — aborting. Is the server up? Are DB + Redis reachable?")
            return 2

        # Core lifecycle
        if not signup_new_user(client, run):
            print("\nSignup failed — cannot continue.")
            return 1
        signup_duplicate_email_rejected(client, run)
        signup_weak_password_rejected(client, run)
        login_correct_credentials(client, run)
        me_returns_user_and_org(client, run)
        refresh_rotates_tokens(client, run)
        logout_revokes_session(client, run)

        # Security gate (separate from happy path — exercises Redis throttle)
        login_throttle_locks_after_max_attempts(client, run)

    failed = [s for s in run.steps if not s.passed]
    total = len(run.steps)
    print(f"\n{total - len(failed)}/{total} steps passed.")
    if failed:
        print("\nFAILED:")
        for s in failed:
            print(f"  - {s.name}: {s.detail}")
        return 1
    print("All golden-path checks PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
