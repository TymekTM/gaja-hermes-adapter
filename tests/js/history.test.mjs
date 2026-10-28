// web/history.js must use ONE turn id end to end: the client generates it, POSTs it, and syncServer()
// must re-put the same row instead of inserting a second copy keyed by the server's own uuid.
import test from 'node:test';
import assert from 'node:assert/strict';

import { installIndexedDbShim, loadWebModule } from './support.mjs';

installIndexedDbShim();

// Fake /api/history: stores the id the client sent (a server that mints its own id is what caused the
// duplicate rows before the fix). `serverIds` records exactly what arrived.
const server = [];
globalThis.fetch = async (url, opts = {}) => {
  const method = opts.method || 'GET';
  if (method === 'POST') {
    const body = JSON.parse(opts.body);
    server.push({
      id: body.id ?? `srv_${server.length}`,
      ts: 1000 + server.length,
      role: body.role,
      text: body.text,
      lang: body.lang,
      tool: body.tool ?? null,
    });
    return { ok: true, json: async () => ({ ok: true }) };
  }
  if (method === 'GET') return { ok: true, json: async () => ({ turns: server.slice(-50) }) };
  return { ok: true, json: async () => ({}) };
};

const { historyStore } = await loadWebModule('history.js', ['i18n.js']);

test('a turn keeps one id from the client through the server round trip', async () => {
  const first = await historyStore.addTurn({ role: 'user', text: 'first message' });
  const second = await historyStore.addTurn({ role: 'assistant', text: 'second message' });
  await new Promise((resolve) => setTimeout(resolve, 50)); // let both POSTs land

  assert.deepEqual(server.map((turn) => turn.id), [first.id, second.id], 'the server must receive the client ids');

  await historyStore.syncServer(50);
  const all = await historyStore.getAllTurns();

  assert.equal(all.length, 2, 'syncing two turns must not duplicate them');
  assert.deepEqual(all.map((turn) => turn.id).sort(), [first.id, second.id].sort());
  assert.deepEqual(all.map((turn) => turn.text).sort(), ['first message', 'second message']);
});
