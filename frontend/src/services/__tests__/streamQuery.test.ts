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
