import { describe, it, expect, vi, beforeEach } from 'vitest';
import request from 'supertest';

vi.mock('node-fetch', () => ({ default: vi.fn() }));
import fetch from 'node-fetch';
import app from '../server.js';

beforeEach(() => {
  vi.mocked(fetch).mockReset();
});

describe('GET /api/health', () => {
  it('returns 200 and ml_service=false when the ML service is unreachable', async () => {
    vi.mocked(fetch).mockRejectedValue(new Error('down'));
    const res = await request(app).get('/api/health');
    expect(res.status).toBe(200);
    expect(res.body.services.ml_service).toBe(false);
  });
});
