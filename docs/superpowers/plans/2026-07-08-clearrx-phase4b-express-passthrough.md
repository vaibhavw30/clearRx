# ClearRx Phase 4b — Express Streaming Passthrough Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `POST /api/query` (JSON proxy) and `POST /api/query/stream` (SSE passthrough) to the Express API so the frontend can reach the new ML RAG endpoints, and stand up the proxy's first test suite (Vitest + supertest, mocking the ML `fetch`).

**Architecture:** `api/server.js` gains two routes that proxy to the ML service at `process.env.ML_BASE`. `/api/query` awaits the ML JSON and forwards it; `/api/query/stream` sets `text/event-stream` headers and pipes the ML response body straight through unbuffered. To make the single-file server testable, it is refactored to `export default app` and guard `app.listen` behind `NODE_ENV !== 'test'`, so supertest imports the app without binding a port. Tests mock `node-fetch` so no live ML service is needed.

**Tech Stack:** Node ESM, Express 5, node-fetch v3, Vitest, supertest.

## Global Constraints

- `api/` is ESM (`"type": "module"`) — use `import`/`export`, not `require`.
- The ML base URL is always `process.env.ML_BASE || 'http://localhost:8000'` (existing convention) — do not hardcode a different default.
- Tests must NOT hit a live ML service — mock `node-fetch`. Tests must NOT bind a real port — import the exported `app` via supertest.
- Do not change any existing route's behavior; this phase is strictly additive except for the `export default app` + listen-guard refactor.
- SSE passthrough must be unbuffered: set headers, `res.flushHeaders()`, then pipe the ML body.
- Run tests with `npm test` from `api/`. TDD: failing test first → RED → implement → GREEN. Commit after every task.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `api/package.json` | deps + scripts | Modify: add `vitest`+`supertest` devDeps; `test` script |
| `api/vitest.config.js` | test config | Create |
| `api/vitest.setup.js` | test env stubs | Create |
| `api/server.js` | Express app + routes | Modify: `export default app`, guard `listen`, add 2 proxy routes |
| `api/tests/health.test.js` | harness smoke test | Create |
| `api/tests/query.test.js` | `/api/query` proxy test | Create |
| `api/tests/query_stream.test.js` | `/api/query/stream` passthrough test | Create |

---

### Task 1: Test harness + exportable app

Make `server.js` importable by supertest (export the app, don't listen under test), add Vitest + supertest, and prove the harness works with one smoke test that mocks `node-fetch`.

**Files:**
- Modify: `api/package.json`
- Create: `api/vitest.config.js`, `api/vitest.setup.js`, `api/tests/health.test.js`
- Modify: `api/server.js` (export + listen guard)

**Interfaces:**
- Produces: `api/server.js` `export default app` (the Express app, no port bound under test). The `node-fetch` default export is what tests mock via `vi.mock('node-fetch', ...)`.

- [ ] **Step 1: Add deps + test script**

In `api/package.json`, set the `test` script and add devDependencies:

```json
  "scripts": {
    "start": "node server.js",
    "dev": "nodemon server.js",
    "test": "vitest run"
  },
```

```json
  "devDependencies": {
    "nodemon": "^3.1.10",
    "supertest": "^7.0.0",
    "vitest": "^2.1.0"
  }
```

Then install: `cd api && npm install`.

- [ ] **Step 2: Create the Vitest config + env setup**

`api/vitest.config.js`:

```js
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'node',
    setupFiles: ['./vitest.setup.js'],
  },
});
```

`api/vitest.setup.js` (stubs env so importing `server.js` — which constructs the Supabase client at module load — never throws, and marks the test env so `app.listen` is skipped):

```js
process.env.NODE_ENV = 'test';
process.env.SUPABASE_URL ||= 'http://localhost';
process.env.SUPABASE_SERVICE_ROLE_KEY ||= 'test-key';
process.env.ML_BASE ||= 'http://ml.test';
```

- [ ] **Step 3: Write the failing smoke test**

`api/tests/health.test.js`:

```js
import { describe, it, expect, vi, beforeEach } from 'vitest';
import request from 'supertest';

vi.mock('node-fetch', () => ({ default: vi.fn() }));
import fetch from 'node-fetch';
import app from '../server.js';

beforeEach(() => vi.mocked(fetch).mockReset());

describe('GET /api/health', () => {
  it('returns 200 and ml_service=false when the ML service is unreachable', async () => {
    vi.mocked(fetch).mockRejectedValue(new Error('down'));
    const res = await request(app).get('/api/health');
    expect(res.status).toBe(200);
    expect(res.body.services.ml_service).toBe(false);
  });
});
```

- [ ] **Step 4: Run test to verify it fails**

Run: `cd api && npm test`
Expected: FAIL — importing `../server.js` throws / has no default export (the app isn't exported yet), or the process tries to `listen`.

- [ ] **Step 5: Export the app and guard `listen`**

In `api/server.js`, replace the final `app.listen(...)` block:

```js
const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
  console.log(`🚀 DDI Assistant API running on port ${PORT}`);
  console.log(`📊 Health check: http://localhost:${PORT}/api/health`);
  console.log(`🔗 ML Service: ${process.env.ML_BASE || 'http://localhost:8000'}`);
});
```

with:

```js
const PORT = process.env.PORT || 3001;
if (process.env.NODE_ENV !== 'test') {
  app.listen(PORT, () => {
    console.log(`🚀 DDI Assistant API running on port ${PORT}`);
    console.log(`📊 Health check: http://localhost:${PORT}/api/health`);
    console.log(`🔗 ML Service: ${process.env.ML_BASE || 'http://localhost:8000'}`);
  });
}

export default app;
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd api && npm test`
Expected: PASS (1 test).

- [ ] **Step 7: Commit**

```bash
git add api/package.json api/package-lock.json api/vitest.config.js api/vitest.setup.js api/tests/health.test.js api/server.js
git commit -m "test(api): add vitest+supertest harness; export app for testing"
```

---

### Task 2: `POST /api/query` JSON proxy

**Files:**
- Modify: `api/server.js`
- Create: `api/tests/query.test.js`

**Interfaces:**
- Consumes: `fetch` (node-fetch), `process.env.ML_BASE`, `express.json()` body parsing (already mounted).
- Produces: `POST /api/query` — forwards the request body to ML `POST /query` and relays the ML JSON + status; `502` on ML unreachable.

- [ ] **Step 1: Write the failing test**

`api/tests/query.test.js`:

```js
import { describe, it, expect, vi, beforeEach } from 'vitest';
import request from 'supertest';

vi.mock('node-fetch', () => ({ default: vi.fn() }));
import fetch from 'node-fetch';
import app from '../server.js';

beforeEach(() => vi.mocked(fetch).mockReset());

describe('POST /api/query', () => {
  it('proxies the query to the ML service and relays its JSON', async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: true, status: 200,
      json: async () => ({ answer: 'Increased bleeding risk.', citations: [{ source_doc_id: 'int_warfarin_ibuprofen' }] }),
    });
    const res = await request(app).post('/api/query').send({ query: 'warfarin ibuprofen' });
    expect(res.status).toBe(200);
    expect(res.body.answer).toBe('Increased bleeding risk.');
    expect(res.body.citations[0].source_doc_id).toBe('int_warfarin_ibuprofen');
    const [url, opts] = vi.mocked(fetch).mock.calls[0];
    expect(url).toContain('/query');
    expect(opts.method).toBe('POST');
    expect(JSON.parse(opts.body).query).toBe('warfarin ibuprofen');
  });

  it('returns 502 when the ML service is unreachable', async () => {
    vi.mocked(fetch).mockRejectedValue(new Error('down'));
    const res = await request(app).post('/api/query').send({ query: 'x' });
    expect(res.status).toBe(502);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd api && npm test`
Expected: FAIL — `POST /api/query` hits the 404 handler (route not defined), so status is 404 not 200/502.

- [ ] **Step 3: Add the route**

In `api/server.js`, add (near the other `/api/...` routes, before the error-handling middleware):

```js
// Proxy: free-text RAG query (non-streaming) -> ML /query
app.post('/api/query', async (req, res) => {
  try {
    const mlServiceUrl = process.env.ML_BASE || 'http://localhost:8000';
    const mlResponse = await fetch(`${mlServiceUrl}/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body),
    });
    const data = await mlResponse.json();
    res.status(mlResponse.status).json(data);
  } catch (error) {
    console.error('Error proxying /api/query:', error.message);
    res.status(502).json({ error: 'ML service unavailable' });
  }
});
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd api && npm test`
Expected: PASS (health + both query tests).

- [ ] **Step 5: Commit**

```bash
git add api/server.js api/tests/query.test.js
git commit -m "feat(api): add /api/query proxy to ML RAG endpoint"
```

---

### Task 3: `POST /api/query/stream` SSE passthrough

Pipe the ML SSE stream straight through, unbuffered, so tokens reach the browser as they arrive.

**Files:**
- Modify: `api/server.js`
- Create: `api/tests/query_stream.test.js`

**Interfaces:**
- Consumes: `fetch` (node-fetch v3 — its `Response.body` is a Node `Readable` stream), `process.env.ML_BASE`.
- Produces: `POST /api/query/stream` — sets `text/event-stream` headers, `flushHeaders()`, and pipes the ML response body to the client; `502` on ML unreachable (before headers are flushed).

- [ ] **Step 1: Write the failing test**

`api/tests/query_stream.test.js`:

```js
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { Readable } from 'node:stream';
import request from 'supertest';

vi.mock('node-fetch', () => ({ default: vi.fn() }));
import fetch from 'node-fetch';
import app from '../server.js';

beforeEach(() => vi.mocked(fetch).mockReset());

describe('POST /api/query/stream', () => {
  it('forwards the ML SSE stream with event-stream headers', async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: true, status: 200,
      body: Readable.from(['data: Increased \n\n', 'data: bleeding risk.\n\n',
                           'event: citations\ndata: []\n\n', 'data: [DONE]\n\n']),
    });
    const res = await request(app).post('/api/query/stream').send({ query: 'warfarin ibuprofen' });
    expect(res.status).toBe(200);
    expect(res.headers['content-type']).toContain('text/event-stream');
    expect(res.text).toContain('data: Increased ');
    expect(res.text).toContain('data: bleeding risk.');
    expect(res.text).toContain('data: [DONE]');
    const [url, opts] = vi.mocked(fetch).mock.calls[0];
    expect(url).toContain('/query/stream');
    expect(opts.method).toBe('POST');
  });

  it('returns 502 when the ML service is unreachable', async () => {
    vi.mocked(fetch).mockRejectedValue(new Error('down'));
    const res = await request(app).post('/api/query/stream').send({ query: 'x' });
    expect(res.status).toBe(502);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd api && npm test`
Expected: FAIL — `POST /api/query/stream` hits the 404 handler.

- [ ] **Step 3: Add the route**

In `api/server.js`, add (next to `/api/query`, before the error-handling middleware):

```js
// Proxy: streamed RAG query -> ML /query/stream, piped through as SSE
app.post('/api/query/stream', async (req, res) => {
  try {
    const mlServiceUrl = process.env.ML_BASE || 'http://localhost:8000';
    const mlResponse = await fetch(`${mlServiceUrl}/query/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body),
    });
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.flushHeaders();
    mlResponse.body.on('error', () => res.end());
    mlResponse.body.pipe(res);
  } catch (error) {
    console.error('Error proxying /api/query/stream:', error.message);
    res.status(502).json({ error: 'ML service unavailable' });
  }
});
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd api && npm test`
Expected: PASS (all suites: health + query + query_stream).

- [ ] **Step 5: Commit**

```bash
git add api/server.js api/tests/query_stream.test.js
git commit -m "feat(api): add /api/query/stream SSE passthrough to ML"
```

---

## Self-Review

- **Spec coverage (design §4b):** `/api/query` proxy (Task 2), `/api/query/stream` SSE passthrough (Task 3), first Vitest+supertest proxy tests + exportable app (Task 1). "Existing routes keep pointing at `ML_BASE` (now the new app)" — no change needed: they already use `process.env.ML_BASE`, and the new ML app serves the same paths/contracts (Phase 4a), so this is satisfied by construction; no task required.
- **Placeholder scan:** none — every step has complete config/route/test code.
- **Type consistency:** `process.env.ML_BASE || 'http://localhost:8000'`, the `vi.mock('node-fetch', () => ({ default: vi.fn() }))` + `vi.mocked(fetch)` pattern, `export default app`, and the `NODE_ENV !== 'test'` listen guard are used identically across all three tasks. The mocked `Response` shape (`ok`/`status`/`json()` for JSON; `ok`/`status`/`body` as a `Readable` for streaming) matches how the routes consume it.
- **YAGNI:** no ret/timeout/reconnect logic beyond the existing convention; SSE passthrough is a straight pipe. Live end-to-end streaming is validated on the provisioned machine (needs the ML app + Ollama running), not in these mocked tests.
