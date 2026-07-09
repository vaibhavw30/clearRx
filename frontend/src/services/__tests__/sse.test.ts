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
