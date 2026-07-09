import { describe, it, expect, vi, beforeEach } from 'vitest';
import { Readable } from 'node:stream';
import request from 'supertest';

vi.mock('node-fetch', () => ({ default: vi.fn() }));
import fetch from 'node-fetch';
import app from '../server.js';

beforeEach(() => { vi.mocked(fetch).mockReset(); });

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
