# Navis Identity — manual & golden-path tests

Two artifacts that exercise the Identity API contract against a live server:

| File | Purpose |
|---|---|
| `golden_path.py` | Scripted end-to-end happy + security paths. Returns exit 0 on success. |
| `navis-identity.postman_collection.json` | Postman/Bruno/Insomnia collection for manual exploration. |

## golden_path.py

Run against a local server:

```powershell
# 1. Start the API (in another terminal)
uvicorn navis.main:app --reload

# 2. Run the script
python tools/identity/golden_path.py
```

Run against staging:

```powershell
python tools/identity/golden_path.py --base-url https://staging.navis.example
```

If Redis is down (throttle fails open), skip the lock test:

```powershell
python tools/identity/golden_path.py --skip-throttle
```

What it asserts (in order):

1. `/ready` — DB + Redis reachable
2. `POST /auth/signup` — 201, tokens issued
3. `POST /auth/signup` (duplicate) — 400, no 500
4. `POST /auth/signup` (weak password) — 422 schema rejection
5. `POST /auth/login` — 200, tokens issued
6. `GET /auth/me` — returns user + organization + role; email matches
7. `POST /auth/refresh` — rotates tokens; old refresh now invalid (replay → 401)
8. `POST /auth/logout` — 200; refresh after logout → 401 (session revoked)
9. `POST /auth/login` × 6 wrong passwords — throttle locks the account (correct password also → 401)

Exit codes: `0` all passed · `1` ≥1 step failed · `2` server unreachable.

## navis-identity.postman_collection.json

Import into Postman, Bruno, or Insomnia. The collection variables include `base_url`, `test_email`, `test_password`, `access_token`, `refresh_token`, `old_refresh`. Pre-request and test scripts auto-extract tokens and assert status codes, so running the **Core auth flow** folder top-to-bottom executes the full lifecycle hands-free.

For Google SSO, the request body needs a real `code` (preferred) or `id_token` pasted from the frontend's Google OAuth2 client — there's no way to mint a valid one server-side.
