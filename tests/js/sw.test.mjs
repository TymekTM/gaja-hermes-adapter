// sw.js must send live endpoints straight to the network: its fetch handler must not call
// respondWith() for /api/*, /ws, /health or /metrics, or they queue behind the shell cache.
// Run with: node --test tests/js/*.test.mjs (repo has no JS test runner).
import test from 'node:test';
import assert from 'node:assert/strict';

import { loadWebModule } from './support.mjs';

const handlers = {};
globalThis.self = globalThis; // sw.js registers its listeners on self
globalThis.addEventListener = (name, fn) => { handlers[name] = fn; };

const cacheRows = new Map(); // url -> Response-like
globalThis.caches = {
  match: async (req) => cacheRows.get(req.url),
  open: async () => ({ put: async (req, resp) => { cacheRows.set(req.url, resp); } }),
  keys: async () => [...cacheRows.keys()],
  delete: async () => true,
};
globalThis.fetch = async (req) => ({ status: 200, type: 'basic', clone() { return this; } });

await loadWebModule('sw.js'); // no exports; registering the listeners is the point

const BASE = 'https://gaja.test';

function dispatch(url, method = 'GET') {
  const event = {
    request: { method, url },
    responded: undefined,
    respondWith(p) { this.responded = p; },
    waitUntil() {},
  };
  handlers.fetch(event);
  return event;
}

test('live endpoints bypass the service worker entirely', () => {
  for (const path of ['/api/history', '/ws', '/health', '/metrics']) {
    const event = dispatch(BASE + path);
    assert.equal(event.responded, undefined, path + ' must not be handled by the cache');
  }
});

test('the app shell still goes through the cache path', async () => {
  const event = dispatch(BASE + '/style.css');
  assert.ok(event.responded, 'shell assets must be handled by the fetch handler');
  const res = await event.responded;
  assert.equal(res.status, 200);
});
