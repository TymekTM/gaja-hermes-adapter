// Conversation history: client-side persistence in IndexedDB + server sync

import { t } from './i18n.js';

const DB_NAME = 'gaja_history_db';
const STORE_NAME = 'turns';
const DB_VERSION = 1;

// Mirrors the server's `TurnCreate.id` pattern: [A-Za-z0-9_-]{1,64}.
function newTurnId() {
  return 't_' + Date.now() + '_' + Math.random().toString(36).slice(2, 7);
}

class HistoryStore {
  constructor() {
    this.db = null;
    this.initPromise = this.openDB();
  }

  async openDB() {
    if (!('indexedDB' in window)) return null;
    return new Promise((resolve) => {
      const req = indexedDB.open(DB_NAME, DB_VERSION);
      req.onupgradeneeded = (e) => {
        const db = e.target.result;
        if (!db.objectStoreNames.contains(STORE_NAME)) {
          const store = db.createObjectStore(STORE_NAME, { keyPath: 'id' });
          store.createIndex('ts', 'ts', { unique: false });
        }
      };
      req.onsuccess = (e) => {
        this.db = e.target.result;
        resolve(this.db);
      };
      req.onerror = () => resolve(null);
    });
  }

  async addTurn(turn) {
    const item = {
      id: turn.id || newTurnId(),
      ts: turn.ts || Date.now() / 1000,
      role: turn.role,
      text: turn.text,
      lang: turn.lang || 'pl',
      tool: turn.tool || null,
    };

    // Store in IndexedDB
    try {
      const db = await this.initPromise;
      if (db) {
        const tx = db.transaction(STORE_NAME, 'readwrite');
        tx.objectStore(STORE_NAME).put(item);
      } else {
        const list = JSON.parse(localStorage.getItem('gaja.local_history') || '[]');
        list.push(item);
        if (list.length > 200) list.shift();
        localStorage.setItem('gaja.local_history', JSON.stringify(list));
      }
    } catch (_) {}

    // Post to server (asynchronous). The server keeps this id, so syncServer() re-puts the same row.
    try {
      fetch('/api/history', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: item.id, role: item.role, text: item.text, lang: item.lang, tool: item.tool }),
      }).catch(() => {});
    } catch (_) {}

    return item;
  }

  async getAllTurns() {
    try {
      const db = await this.initPromise;
      if (db) {
        return new Promise((resolve) => {
          const tx = db.transaction(STORE_NAME, 'readonly');
          const req = tx.objectStore(STORE_NAME).getAll();
          req.onsuccess = () => {
            const res = req.result || [];
            res.sort((a, b) => a.ts - b.ts);
            resolve(res);
          };
          req.onerror = () => resolve([]);
        });
      } else {
        const list = JSON.parse(localStorage.getItem('gaja.local_history') || '[]');
        list.sort((a, b) => a.ts - b.ts);
        return list;
      }
    } catch (_) {
      return [];
    }
  }

  async clear() {
    try {
      const db = await this.initPromise;
      if (db) {
        const tx = db.transaction(STORE_NAME, 'readwrite');
        tx.objectStore(STORE_NAME).clear();
      }
      localStorage.removeItem('gaja.local_history');
    } catch (_) {}

    try {
      await fetch('/api/history', { method: 'DELETE' });
    } catch (_) {}
  }

  async syncServer(limit = 50) {
    try {
      const res = await fetch('/api/history?limit=' + limit);
      if (!res.ok) return;
      const data = await res.json();
      const turns = data.turns || [];
      const db = await this.initPromise;
      if (db) {
        const tx = db.transaction(STORE_NAME, 'readwrite');
        const store = tx.objectStore(STORE_NAME);
        for (const t of turns) {
          store.put(t);
        }
      }
    } catch (_) {}
  }
}

export const historyStore = new HistoryStore();

export class HistoryDrawer {
  constructor({ drawer, backdrop, app, openBtn, closeBtn, searchInput, clearBtn, listEl, onInsert }) {
    this.drawer = drawer;
    this.backdrop = backdrop;
    this.app = app;
    this.openBtn = openBtn;
    this.closeBtn = closeBtn;
    this.searchInput = searchInput;
    this.clearBtn = clearBtn;
    this.listEl = listEl;
    this.onInsert = onInsert || (() => {});
    this.isOpen = false;
    this.allTurns = [];

    openBtn.addEventListener('click', () => (this.isOpen ? this.close() : this.open()));
    closeBtn.addEventListener('click', () => this.close());
    backdrop.addEventListener('click', () => { if (this.isOpen) this.close(); });

    searchInput.addEventListener('input', () => this.filterList());
    clearBtn.addEventListener('click', () => this.handleClear());
  }

  async open() {
    if (this.isOpen) return;
    this.isOpen = true;
    this.drawer.inert = false;
    this.app.inert = true;
    this.backdrop.hidden = false;
    this.openBtn.setAttribute('aria-expanded', 'true');

    await historyStore.syncServer(50);
    this.allTurns = await historyStore.getAllTurns();
    this.searchInput.value = '';
    this.renderTurns(this.allTurns);

    requestAnimationFrame(() => {
      this.drawer.classList.add('is-open');
      this.closeBtn.focus({ preventScroll: true });
    });
  }

  close() {
    if (!this.isOpen) return;
    this.isOpen = false;
    this.drawer.classList.remove('is-open');
    this.drawer.inert = true;
    this.app.inert = false;
    this.backdrop.hidden = true;
    this.openBtn.setAttribute('aria-expanded', 'false');
    this.openBtn.focus({ preventScroll: true });
  }

  async handleClear() {
    if (!confirm(t('history.confirm_clear') || 'Wyczyścić całą historię?')) return;
    await historyStore.clear();
    this.allTurns = [];
    this.renderTurns([]);
  }

  filterList() {
    const q = this.searchInput.value.trim().toLowerCase();
    if (!q) {
      this.renderTurns(this.allTurns);
      return;
    }
    const filtered = this.allTurns.filter((t) => (t.text || '').toLowerCase().includes(q));
    this.renderTurns(filtered);
  }

  renderTurns(turns) {
    this.listEl.textContent = '';
    if (!turns.length) {
      const p = document.createElement('p');
      p.className = 'history-empty';
      p.textContent = t('history.empty') || 'Brak zapisanych rozmów.';
      this.listEl.appendChild(p);
      return;
    }

    const sorted = [...turns].reverse();
    for (const item of sorted) {
      const li = document.createElement('li');
      li.className = 'history-item role-' + item.role;

      const meta = document.createElement('div');
      meta.className = 'history-meta';

      const roleSpan = document.createElement('span');
      roleSpan.className = 'history-role';
      roleSpan.textContent = item.role === 'user' ? (t('chat.you') || 'Ty') : (t('chat.gaja') || 'Gaja');

      const timeSpan = document.createElement('time');
      timeSpan.className = 'history-time';
      const date = new Date((item.ts || 0) * 1000);
      timeSpan.textContent = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) + ' · ' + date.toLocaleDateString();

      meta.append(roleSpan, timeSpan);
      if (item.tool) {
        const toolBadge = document.createElement('span');
        toolBadge.className = 'history-tool';
        toolBadge.textContent = item.tool;
        meta.appendChild(toolBadge);
      }

      const body = document.createElement('div');
      body.className = 'history-text';
      body.textContent = item.text;

      li.append(meta, body);
      this.listEl.appendChild(li);
    }
  }
}
