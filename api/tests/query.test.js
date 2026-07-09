import { describe, it, expect, vi, beforeEach } from 'vitest';
import request from 'supertest';

vi.mock('node-fetch', () => ({ default: vi.fn() }));
import fetch from 'node-fetch';
import app from '../server.js';

beforeEach(() => {
  vi.mocked(fetch).mockReset();
});

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
