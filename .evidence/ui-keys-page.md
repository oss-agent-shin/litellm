# Evidence: LiteLLM UI `/ui/?page=api-keys` rendering against a real Postgres-backed proxy

This file is intentionally non-executable. Its only purpose is to carry the
runtime evidence in the PR body — a screenshot of the LiteLLM admin UI keys
page served by a real proxy backed by the Prisma/Postgres datastore (i.e.
`STORE_MODEL_IN_DB=True`, login via the real form, not localStorage).

## What was run

- `litellm-up` → Postgres + the proxy on a free port (here `:34035`), with
  `--use_prisma_db_push` against the pre-provisioned `DATABASE_URL`.
- `litellm-status` → `{"status":"ready","port":34035,"error":null}`.
- `litellm-shot http://127.0.0.1:34035 "?page=api-keys" /home/user/keys.png` →
  drove headless chromium through the real `admin` / `sk-1234` login form and
  navigated to `/ui/?page=api-keys`.

## Captured page text (sanity check that the screenshot is the keys page, not a login redirect)

```
SETTINGS
Settings
New
+ Create New Key
Filters
Reset Filters
Showing 1 - 0 of 0 results
...
Key ID | Key Alias | Status | Secret Key | Team | Organization | User
       | Created At | Created By | Updated At | Last Active | Expires
       | Spend (USD) | Budget (USD) | Budget Reset | Models | Rate Limits

No keys found
```

The empty result set ("No keys found") is expected — the ephemeral dev DB is
fresh on each `litellm-up`, so no keys exist yet. The point being proven here
is that the proxy started, the DB connected, the UI was served by the proxy
(no separate `next dev`), and the keys route lazy-compiled and rendered.

See the PR body for the embedded screenshot.
