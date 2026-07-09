# ClearRx Phase 4c — Frontend Streaming UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a live-streaming "Ask ClearRx" panel to the frontend: a pure SSE frame parser, an `apiService.streamQuery` method that reads the Express `/api/query/stream` ReadableStream, and an `AskClearRx` component (mounted on the dashboard) that renders tokens as they arrive with grounded citations.

**Architecture:** Three units behind clean seams. `services/sse.ts` is a pure, stateful SSE frame parser (unit-tested, buffers partial frames). `apiService.streamQuery` wires `fetch` → `ReadableStream` reader → parser → callbacks (unit-tested with a fake stream). `AskClearRx` is a thin React component consuming `streamQuery` (verified by typecheck/build; component logic is wiring). Chosen test scope: **Vitest (node env), logic-only** — parser + streamQuery are unit-tested; the component is not (no jsdom/RTL).

**Tech Stack:** React 18 + Vite + TypeScript, shadcn/ui (Card/Button; a plain styled `<textarea>` since shadcn `Textarea` isn't installed), Vitest (node).

## Global Constraints

- Frontend is React + Vite + TS with the `@` → `./src` path alias (`vite.config.ts`). The Vitest config must mirror that alias.
- The frontend calls **Express** at `import.meta.env.VITE_API_BASE_URL || 'http://localhost:3001'` (the existing `API_BASE_URL` in `api.ts`). The stream endpoint is `POST /api/query/stream` (Phase 4b).
- SSE frame shape (from Phase 4a/4b): token frames `data: <token>\n\n`; a citations frame `event: citations\ndata: <json-array>\n\n`; terminal `data: [DONE]\n\n`. Tokens are single-line (Phase 4a puts each token on one `data:` line).
- Tests are Vitest **node** environment, offline (mock `fetch` / build a `ReadableStream` in-process) — no live Express/ML, no jsdom.
- Additive only: do not change existing `api.ts` methods or dashboard behavior beyond mounting the new panel.
- Work from `frontend/`. Run tests with `npm test`; typecheck/build with `npm run build`; lint with `npm run lint`.
- Commit after every task. Node 18+ provides `fetch`, `Response`, `ReadableStream`, `TextEncoder`, `TextDecoder` as globals.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `frontend/package.json` | scripts + dev deps | Modify: add `test` script + `vitest` |
| `frontend/vitest.config.ts` | test config (node env, `@` alias) | Create |
| `frontend/src/services/sse.ts` | pure SSE frame parser | Create |
| `frontend/src/services/__tests__/sse.test.ts` | parser tests | Create |
| `frontend/src/services/api.ts` | API client | Modify: add `Citation` + `streamQuery` |
| `frontend/src/services/__tests__/streamQuery.test.ts` | streamQuery tests | Create |
| `frontend/src/components/AskClearRx.tsx` | streaming query panel | Create |
| `frontend/src/pages/NewDashboard.tsx` | dashboard | Modify: mount `<AskClearRx />` |

---

### Task 1: Vitest harness + SSE parser

**Files:**
- Modify: `frontend/package.json`
- Create: `frontend/vitest.config.ts`
- Create: `frontend/src/services/sse.ts`
- Test: `frontend/src/services/__tests__/sse.test.ts`

**Interfaces:**
- Produces: `type SSEEvent = {type:'token'; data:string} | {type:'citations'; data:unknown} | {type:'done'}` and `createSSEParser(): { feed(text: string): SSEEvent[] }` in `services/sse.ts`. `feed` buffers partial frames and returns the events completed by this chunk.

- [ ] **Step 1: Write the failing test**

Create `frontend/src/services/__tests__/sse.test.ts`:

```ts
import { describe, it, expect } from 'vitest';
import { createSSEParser } from '@/services/sse';

describe('createSSEParser', () => {
  it('parses a single token frame', () => {
    const p = createSSEParser();
    expect(p.feed('data: hello\n\n')).toEqual([{ type: 'token', data: 'hello' }]);
  });

  it('buffers a frame split across feeds', () => {
    const p = createSSEParser();
    expect(p.feed('data: hel')).toEqual([]);                 // incomplete frame
    expect(p.feed('lo\n\n')).toEqual([{ type: 'token', data: 'hello' }]);
  });

  it('parses a citations event then done', () => {
    const p = createSSEParser();
    const evs = p.feed('event: citations\ndata: [{"source_doc_id":"int_x"}]\n\ndata: [DONE]\n\n');
    expect(evs).toEqual([
      { type: 'citations', data: [{ source_doc_id: 'int_x' }] },
      { type: 'done' },
    ]);
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run (from `frontend/`): `npm test` — after the package.json edit in Step 3 the runner exists; before implementing `sse.ts` the suite FAILS with a module-resolution error for `@/services/sse`.
(If `npm test` is not yet a script, that's expected — Step 3 adds it.)

- [ ] **Step 3: Implement harness + parser**

In `frontend/package.json`, add a `test` script (alongside the existing scripts) and `vitest` to `devDependencies`:

```json
    "test": "vitest run"
```
```json
    "vitest": "^2.1.0"
```

Then run `npm install` (from `frontend/`).

Create `frontend/vitest.config.ts`:

```ts
import { defineConfig } from 'vitest/config';
import path from 'path';

export default defineConfig({
  resolve: { alias: { '@': path.resolve(__dirname, './src') } },
  test: { environment: 'node' },
});
```

Create `frontend/src/services/sse.ts`:

```ts
export type SSEEvent =
  | { type: 'token'; data: string }
  | { type: 'citations'; data: unknown }
  | { type: 'done' };

/**
 * Stateful parser for the ClearRx SSE stream. `feed` accepts an arbitrary
 * text chunk (frames may be split across chunks), buffers any incomplete
 * trailing frame, and returns the events completed so far.
 */
export function createSSEParser(): { feed(text: string): SSEEvent[] } {
  let buffer = '';
  return {
    feed(text: string): SSEEvent[] {
      buffer += text;
      const events: SSEEvent[] = [];
      let idx: number;
      while ((idx = buffer.indexOf('\n\n')) !== -1) {
        const frame = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        if (!frame) continue;
        const lines = frame.split('\n');
        const isCitations = lines.some((l) => l.startsWith('event: citations'));
        const dataLine = lines.find((l) => l.startsWith('data: '));
        if (dataLine === undefined) continue;
        const payload = dataLine.slice('data: '.length);
        if (isCitations) {
          events.push({ type: 'citations', data: JSON.parse(payload) });
        } else if (payload === '[DONE]') {
          events.push({ type: 'done' });
        } else {
          events.push({ type: 'token', data: payload });
        }
      }
      return events;
    },
  };
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `npm test`
Expected: PASS (3 parser tests).

- [ ] **Step 5: Commit**

```bash
git add package.json package-lock.json vitest.config.ts src/services/sse.ts src/services/__tests__/sse.test.ts
git commit -m "test(frontend): add vitest harness + SSE frame parser"
```

---

### Task 2: `streamQuery` on the API client

**Files:**
- Modify: `frontend/src/services/api.ts`
- Test: `frontend/src/services/__tests__/streamQuery.test.ts`

**Interfaces:**
- Consumes: `createSSEParser` (Task 1).
- Produces: `export interface Citation { source_doc_id: string; section?: string | null; url?: string | null }` and a method `streamQuery(query: string, handlers: { onToken: (t: string) => void; onCitations?: (c: Citation[]) => void; onDone?: () => void; onError?: (e: unknown) => void }): Promise<void>` on `ApiService` (available via the exported `apiService` singleton). It POSTs to `${baseUrl}/api/query/stream`, reads the response body stream through the parser, and dispatches callbacks. On non-OK/no-body or a thrown error it calls `onError`.

- [ ] **Step 1: Write the failing test**

Create `frontend/src/services/__tests__/streamQuery.test.ts`:

```ts
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { apiService, type Citation } from '@/services/api';

function sseResponse(frames: string[]): Response {
  const enc = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const f of frames) controller.enqueue(enc.encode(f));
      controller.close();
    },
  });
  return new Response(stream, {
    status: 200,
    headers: { 'Content-Type': 'text/event-stream' },
  });
}

beforeEach(() => {
  vi.restoreAllMocks();
});

describe('apiService.streamQuery', () => {
  it('streams tokens, then citations, then done', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      sseResponse([
        'data: Increased \n\n',
        'data: bleeding risk.\n\n',
        'event: citations\ndata: [{"source_doc_id":"int_warfarin_ibuprofen"}]\n\n',
        'data: [DONE]\n\n',
      ]),
    );
    const tokens: string[] = [];
    let citations: Citation[] = [];
    let done = false;
    await apiService.streamQuery('warfarin ibuprofen', {
      onToken: (t) => tokens.push(t),
      onCitations: (c) => { citations = c; },
      onDone: () => { done = true; },
    });
    expect(tokens.join('')).toBe('Increased bleeding risk.');
    expect(citations[0].source_doc_id).toBe('int_warfarin_ibuprofen');
    expect(done).toBe(true);
  });

  it('calls onError on a non-ok response', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response('', { status: 502 }));
    let err: unknown = null;
    await apiService.streamQuery('x', { onToken: () => {}, onError: (e) => { err = e; } });
    expect(err).toBeTruthy();
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `npm test`
Expected: FAIL — `streamQuery` / `Citation` not exported from `@/services/api`.

- [ ] **Step 3: Implement**

In `frontend/src/services/api.ts`, add the import at the top (with the other imports/const):

```ts
import { createSSEParser } from '@/services/sse';
```

Add the `Citation` interface near the other exported interfaces:

```ts
export interface Citation {
  source_doc_id: string;
  section?: string | null;
  url?: string | null;
}
```

Add this method to the `ApiService` class (e.g. right after `request<T>`):

```ts
  async streamQuery(
    query: string,
    handlers: {
      onToken: (t: string) => void;
      onCitations?: (c: Citation[]) => void;
      onDone?: () => void;
      onError?: (e: unknown) => void;
    },
  ): Promise<void> {
    try {
      const response = await fetch(`${this.baseUrl}/api/query/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      });
      if (!response.ok || !response.body) {
        handlers.onError?.(new Error(`stream request failed: ${response.status}`));
        return;
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      const parser = createSSEParser();
      let finished = false;
      while (!finished) {
        const { value, done } = await reader.read();
        if (done) break;
        for (const ev of parser.feed(decoder.decode(value, { stream: true }))) {
          if (ev.type === 'token') handlers.onToken(ev.data);
          else if (ev.type === 'citations') handlers.onCitations?.(ev.data as Citation[]);
          else if (ev.type === 'done') { handlers.onDone?.(); finished = true; }
        }
      }
      if (!finished) handlers.onDone?.();
    } catch (e) {
      handlers.onError?.(e);
    }
  }
```

- [ ] **Step 4: Run to verify it passes**

Run: `npm test`
Expected: PASS (both streamQuery tests + the Task 1 parser tests).

- [ ] **Step 5: Commit**

```bash
git add src/services/api.ts src/services/__tests__/streamQuery.test.ts
git commit -m "feat(frontend): add streamQuery SSE client method"
```

---

### Task 3: `AskClearRx` panel + dashboard mount

**Files:**
- Create: `frontend/src/components/AskClearRx.tsx`
- Modify: `frontend/src/pages/NewDashboard.tsx`

**Interfaces:**
- Consumes: `apiService.streamQuery`, `Citation` (Task 2); shadcn `Button`, `Card*`.
- Produces: `export function AskClearRx()` — a self-contained panel. No unit test (chosen scope): verified by `npm run build` (typecheck) + `npm run lint`.

- [ ] **Step 1: Create the component**

Create `frontend/src/components/AskClearRx.tsx`:

```tsx
import { useState } from 'react';
import { apiService, type Citation } from '@/services/api';
import { Button } from '@/components/ui/button';
import {
  Card, CardContent, CardDescription, CardHeader, CardTitle,
} from '@/components/ui/card';

export function AskClearRx() {
  const [query, setQuery] = useState('');
  const [answer, setAnswer] = useState('');
  const [citations, setCitations] = useState<Citation[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const ask = async () => {
    if (!query.trim() || loading) return;
    setLoading(true);
    setAnswer('');
    setCitations([]);
    setError(null);
    await apiService.streamQuery(query.trim(), {
      onToken: (t) => setAnswer((a) => a + t),
      onCitations: (c) => setCitations(c),
      onDone: () => setLoading(false),
      onError: () => {
        setError('Sorry, the answer service is unavailable right now.');
        setLoading(false);
      },
    });
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Ask ClearRx</CardTitle>
        <CardDescription>
          Ask about a drug interaction in plain language. This is not medical advice.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <textarea
          className="w-full min-h-[80px] rounded-md border border-input bg-background p-2 text-sm"
          placeholder="e.g. Can I take ibuprofen with warfarin?"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <Button onClick={ask} disabled={loading || !query.trim()}>
          {loading ? 'Answering…' : 'Ask'}
        </Button>
        {error && <p className="text-sm text-destructive">{error}</p>}
        {answer && <p className="whitespace-pre-wrap text-sm">{answer}</p>}
        {citations.length > 0 && (
          <div className="text-xs text-muted-foreground">
            <p className="font-medium">Sources</p>
            <ul className="list-disc pl-4">
              {citations.map((c, i) => (
                <li key={i}>
                  {c.url ? (
                    <a href={c.url} target="_blank" rel="noreferrer" className="underline">
                      {c.source_doc_id}
                    </a>
                  ) : (
                    c.source_doc_id
                  )}
                </li>
              ))}
            </ul>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
```

- [ ] **Step 2: Mount it on the dashboard**

In `frontend/src/pages/NewDashboard.tsx`, add the import with the other component imports:

```tsx
import { AskClearRx } from '@/components/AskClearRx';
```

Render `<AskClearRx />` prominently in the dashboard's main content — place it as the first card inside the primary content container (the main column wrapping the existing dashboard `Card`s, after the `Header`). Insert a single line `<AskClearRx />` at the top of that container. Do not alter any existing markup.

- [ ] **Step 3: Verify typecheck + lint + build**

Run (from `frontend/`):
```bash
npm run build
```
Expected: TypeScript compiles and the Vite build succeeds (this is the gate for the untested component — a type error in the component or its use of `streamQuery`/`Citation` fails here).

Run: `npm run lint`
Expected: no new lint errors in the added files.

Run: `npm test`
Expected: PASS (the Task 1 + Task 2 suites still green; no new tests added this task).

- [ ] **Step 4: Commit**

```bash
git add src/components/AskClearRx.tsx src/pages/NewDashboard.tsx
git commit -m "feat(frontend): add live-streaming Ask ClearRx panel to dashboard"
```

---

## Operator run book (manual, full-stack smoke on the provisioned machine)

After Phases 4a/4b/4c and with Ollama + the Pinecone index live:
```bash
# ML (new app)         : cd ml && source venv/bin/activate && uvicorn app.main:app --port 8000
# Express              : cd api && ML_BASE=http://localhost:8000 npm start
# Frontend             : cd frontend && VITE_API_BASE_URL=http://localhost:3001 npm run dev
```
Open the dashboard, type a drug-interaction question in the "Ask ClearRx" panel, and confirm tokens render live with a Sources list. This is the end-to-end streaming path (browser → Express SSE passthrough → ML `/query/stream` → Ollama).

---

## Self-Review

- **Spec coverage (design §4c/§5/§6):** `parseSSE` (Task 1) + `streamQuery` (Task 2) + `AskClearRx` mounted (Task 3); token-render-live + citations + disclaimer in the component; chosen test scope (Vitest node, logic-only) honored — parser + streamQuery unit-tested, component verified by build/lint. SSE contract matches Phase 4a/4b (`data:`/`event: citations`/`[DONE]`).
- **Placeholder scan:** none — every code step is complete. Task 3 Step 2 gives an exact insertion instruction rather than pasting the whole 400-line dashboard (the surrounding JSX is the deliverable's context, not a gap).
- **Type consistency:** `SSEEvent`/`createSSEParser().feed`, `Citation{source_doc_id,section?,url?}`, and `streamQuery(query, {onToken,onCitations?,onDone?,onError?})` are used identically across Tasks 1–3. `streamQuery` posts to `/api/query/stream` (Phase 4b route) via the Express `baseUrl`. The parser's event union is consumed exhaustively in `streamQuery`.
