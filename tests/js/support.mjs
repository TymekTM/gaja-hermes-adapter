// Loads a browser ES module from web/ in Node. web/ has no package.json, so the module (and the
// relative imports it needs) is copied into a temp dir that declares {"type":"module"}.
// Run the JS tests with: node --test tests/js/*.test.mjs
import { cpSync, mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const WEB = new URL('../../web/', import.meta.url);

export async function loadWebModule(name, deps = []) {
  const dir = mkdtempSync(join(tmpdir(), 'gaja-web-test-'));
  for (const file of [name, ...deps]) cpSync(new URL(file, WEB), join(dir, file));
  writeFileSync(join(dir, 'package.json'), '{"type":"module"}');
  return import('file://' + join(dir, name));
}

/** Minimal IndexedDB shim: one object store, keyed by `id`, enough for web/history.js. */
export function installIndexedDbShim() {
  const rows = new Map();
  const objectStore = {
    put: (item) => rows.set(item.id, item),
    clear: () => rows.clear(),
    getAll: () => {
      const req = {};
      queueMicrotask(() => {
        req.result = [...rows.values()];
        if (req.onsuccess) req.onsuccess();
      });
      return req;
    },
    createIndex: () => {},
  };
  const db = {
    objectStoreNames: { contains: () => true },
    transaction: () => ({ objectStore: () => objectStore }),
    createObjectStore: () => objectStore,
  };
  globalThis.window = globalThis;
  globalThis.indexedDB = {
    open: () => {
      const req = {};
      queueMicrotask(() => {
        req.result = db;
        if (req.onupgradeneeded) req.onupgradeneeded({ target: { result: db } });
        if (req.onsuccess) req.onsuccess({ target: { result: db } });
      });
      return req;
    },
  };
  const local = new Map();
  globalThis.localStorage = {
    getItem: (key) => (local.has(key) ? local.get(key) : null),
    setItem: (key, value) => local.set(key, String(value)),
    removeItem: (key) => local.delete(key),
  };
  return rows;
}
