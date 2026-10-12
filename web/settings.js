// Settings sheet: a bottom sheet on phones, floating panel on wide screens.
// Extended with Vocabulary, Memory, Timers, Devices, and Diagnostics views.

import { t } from './i18n.js';

const WIDE = window.matchMedia('(min-width: 900px) and (min-height: 480px)');

export class SettingsSheet {
  constructor({ sheet, backdrop, app, openBtn, closeBtn, grip, onOpen, onClose }) {
    this.sheet = sheet;
    this.backdrop = backdrop;
    this.app = app;
    this.openBtn = openBtn;
    this.closeBtn = closeBtn;
    this.onOpen = onOpen || (() => {});
    this.onClose = onClose || (() => {});
    this.isOpen = false;
    this.lastFocus = null;

    openBtn.addEventListener('click', () => (this.isOpen ? this.close() : this.open()));
    closeBtn.addEventListener('click', () => this.close());
    backdrop.addEventListener('click', () => this.close());
    this.attachDrag(grip);
  }

  open() {
    if (this.isOpen) return;
    this.isOpen = true;
    this.lastFocus = document.activeElement;
    this.onOpen();
    this.sheet.inert = false;
    this.app.inert = true;
    this.backdrop.hidden = false;
    this.openBtn.setAttribute('aria-expanded', 'true');
    requestAnimationFrame(() => {
      this.sheet.classList.add('is-open');
      this.closeBtn.focus({ preventScroll: true });
    });
  }

  close() {
    if (!this.isOpen) return false;
    this.isOpen = false;
    this.sheet.classList.remove('is-open');
    this.sheet.style.removeProperty('--drag');
    this.sheet.inert = true;
    this.app.inert = false;
    this.backdrop.hidden = true;
    this.openBtn.setAttribute('aria-expanded', 'false');
    const back = this.lastFocus && document.contains(this.lastFocus) ? this.lastFocus : this.openBtn;
    back.focus({ preventScroll: true });
    this.onClose();
    return true;
  }

  attachDrag(grip) {
    const sheet = this.sheet;
    let startY = 0;
    let dy = 0;
    let startT = 0;
    let dragging = false;
    let moved = false;

    const onMove = (e) => {
      if (!dragging) return;
      dy = Math.max(0, e.clientY - startY);
      if (dy > 4) moved = true;
      sheet.style.setProperty('--drag', `${dy}px`);
    };
    const onUp = () => {
      if (!dragging) return;
      dragging = false;
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
      window.removeEventListener('pointercancel', onUp);
      sheet.classList.remove('dragging');
      const velocity = dy / Math.max(1, performance.now() - startT);
      sheet.style.removeProperty('--drag');
      if (moved && (dy > 110 || velocity > 0.6)) this.close();
    };

    const head = sheet.querySelector('.sheet-head');
    [grip, head].forEach((h) => {
      if (!h) return;
      h.addEventListener('pointerdown', (e) => {
        if (WIDE.matches || e.button > 0 || e.target.closest('button')) return;
        dragging = true;
        moved = false;
        startY = e.clientY;
        startT = performance.now();
        dy = 0;
        sheet.classList.add('dragging');
        window.addEventListener('pointermove', onMove);
        window.addEventListener('pointerup', onUp);
        window.addEventListener('pointercancel', onUp);
      });
    });
  }
}

// ---------------------------------------------------------------------------------------------
// Extended panels: Vocab, Memory, Timers, Devices, Diagnostics

export class ExtendedSettingsManager {
  constructor() {
    this.vocabTerms = [];
    this.vocabSuggestions = [];
    this.bindForms();
  }

  async refreshAll() {
    await Promise.allSettled([
      this.loadVocab(),
      this.loadMemory(),
      this.loadTimers(),
      this.loadDevices(),
      this.loadDiagnostics(),
    ]);
  }

  bindForms() {
    const addVocabForm = document.getElementById('form-add-vocab');
    if (addVocabForm) {
      addVocabForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const termInput = document.getElementById('input-vocab-term');
        const aliasInput = document.getElementById('input-vocab-aliases');
        const langSelect = document.getElementById('select-vocab-lang');
        const term = (termInput.value || '').trim();
        if (!term) return;
        const aliases = (aliasInput.value || '')
          .split(',')
          .map((a) => a.trim())
          .filter(Boolean);
        const lang = langSelect.value || 'pl';

        const userTerms = this.vocabTerms.filter((t) => !t.builtin);
        userTerms.push({ term, aliases, lang, builtin: false });
        await this.saveVocabTerms(userTerms);
        termInput.value = '';
        aliasInput.value = '';
      });
    }

    const addMemoryForm = document.getElementById('form-add-memory');
    if (addMemoryForm) {
      addMemoryForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const input = document.getElementById('input-memory-text');
        const text = (input.value || '').trim();
        if (!text) return;
        try {
          const res = await fetch('/api/memory', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text }),
          });
          if (res.ok) {
            input.value = '';
            await this.loadMemory();
          }
        } catch (_) {}
      });
    }
  }

  // --- Vocabulary ---
  async loadVocab() {
    const sec = document.getElementById('sec-vocab');
    if (!sec) return;
    try {
      const res = await fetch('/api/vocab');
      if (res.status === 404 || !res.ok) {
        sec.hidden = true;
        return;
      }
      const data = await res.json();
      sec.hidden = false;
      this.renderVocab(data);
    } catch (_) {
      sec.hidden = true;
    }
  }

  async saveVocabTerms(userTerms) {
    const sec = document.getElementById('sec-vocab');
    try {
      const res = await fetch('/api/vocab', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ terms: userTerms }),
      });
      if (res.ok) {
        const data = await res.json();
        this.renderVocab(data);
      }
    } catch (_) {}
  }

  renderVocab(data) {
    this.vocabTerms = data.terms || [];
    this.vocabSuggestions = data.suggestions || [];

    const listEl = document.getElementById('vocab-terms-list');
    if (listEl) {
      listEl.textContent = '';
      if (!this.vocabTerms.length) {
        const p = document.createElement('p');
        p.className = 'field-note';
        p.textContent = t('vocab.empty') || 'Brak słów w słowniku.';
        listEl.appendChild(p);
      } else {
        this.vocabTerms.forEach((item, idx) => {
          const div = document.createElement('div');
          div.className = 'vocab-item';
          const info = document.createElement('div');
          info.className = 'vocab-info';
          const strong = document.createElement('strong');
          strong.textContent = item.term;
          const aliasesSpan = document.createElement('span');
          aliasesSpan.className = 'vocab-aliases';
          aliasesSpan.textContent = (item.aliases && item.aliases.length) ? ` (${item.aliases.join(', ')})` : '';
          const langBadge = document.createElement('span');
          langBadge.className = 'badge';
          langBadge.textContent = item.lang || 'pl';

          info.append(strong, aliasesSpan, ' ', langBadge);
          div.appendChild(info);

          if (!item.builtin) {
            const delBtn = document.createElement('button');
            delBtn.type = 'button';
            delBtn.className = 'btn-del';
            delBtn.textContent = '×';
            delBtn.title = t('vocab.delete') || 'Usuń słowo';
            delBtn.addEventListener('click', async () => {
              const remaining = this.vocabTerms.filter((t, i) => !t.builtin && i !== idx);
              await this.saveVocabTerms(remaining);
            });
            div.appendChild(delBtn);
          }
          listEl.appendChild(div);
        });
      }
    }

    const sugEl = document.getElementById('vocab-sug-list');
    const sugSec = document.getElementById('vocab-sug-wrap');
    if (sugEl && sugSec) {
      sugEl.textContent = '';
      if (!this.vocabSuggestions.length) {
        sugSec.hidden = true;
      } else {
        sugSec.hidden = false;
        this.vocabSuggestions.forEach((sug) => {
          const card = document.createElement('div');
          card.className = 'sug-card';
          const info = document.createElement('div');
          info.className = 'sug-info';
          const strong = document.createElement('strong');
          strong.textContent = sug.term;
          const heard = document.createElement('p');
          heard.className = 'sug-meta';
          heard.textContent = `${t('vocab.heard') || 'Słyszano'}: ${(sug.heard || []).join(', ')} (${sug.count || 1}x, ${sug.reason || 'near-miss'})`;
          info.append(strong, heard);

          const actions = document.createElement('div');
          actions.className = 'sug-actions';
          const btnApprove = document.createElement('button');
          btnApprove.type = 'button';
          btnApprove.className = 'btn small primary';
          btnApprove.textContent = t('vocab.approve') || 'Zatwierdź';
          btnApprove.addEventListener('click', async () => {
            try {
              const res = await fetch(`/api/vocab/suggestions/${sug.id}/approve`, { method: 'POST' });
              if (res.ok) this.renderVocab(await res.json());
            } catch (_) {}
          });

          const btnReject = document.createElement('button');
          btnReject.type = 'button';
          btnReject.className = 'btn small';
          btnReject.textContent = t('vocab.reject') || 'Odrzuć';
          btnReject.addEventListener('click', async () => {
            try {
              const res = await fetch(`/api/vocab/suggestions/${sug.id}/reject`, { method: 'POST' });
              if (res.ok) this.renderVocab(await res.json());
            } catch (_) {}
          });

          actions.append(btnApprove, btnReject);
          card.append(info, actions);
          sugEl.appendChild(card);
        });
      }
    }
  }

  // --- Memory ---
  async loadMemory() {
    const sec = document.getElementById('sec-memory');
    if (!sec) return;
    try {
      const res = await fetch('/api/memory');
      if (res.status === 404 || !res.ok) {
        sec.hidden = true;
        return;
      }
      const data = await res.json();
      sec.hidden = false;
      this.renderMemory(data.facts || []);
    } catch (_) {
      sec.hidden = true;
    }
  }

  renderMemory(facts) {
    const listEl = document.getElementById('memory-list');
    if (!listEl) return;
    listEl.textContent = '';
    if (!facts.length) {
      const p = document.createElement('p');
      p.className = 'field-note';
      p.textContent = t('memory.empty') || 'Brak zapamiętanych faktów.';
      listEl.appendChild(p);
      return;
    }

    facts.forEach((fact) => {
      const item = document.createElement('div');
      item.className = 'memory-item';
      const text = document.createElement('span');
      text.className = 'memory-text';
      text.textContent = fact.text;

      const delBtn = document.createElement('button');
      delBtn.type = 'button';
      delBtn.className = 'btn-del';
      delBtn.textContent = '×';
      delBtn.title = t('memory.delete') || 'Zapomnij fakt';
      delBtn.addEventListener('click', async () => {
        try {
          const res = await fetch(`/api/memory/${fact.id}`, { method: 'DELETE' });
          if (res.ok) await this.loadMemory();
        } catch (_) {}
      });

      item.append(text, delBtn);
      listEl.appendChild(item);
    });
  }

  // --- Timers ---
  async loadTimers() {
    const sec = document.getElementById('sec-timers');
    if (!sec) return;
    try {
      const res = await fetch('/api/timers');
      if (res.status === 404 || !res.ok) {
        sec.hidden = true;
        return;
      }
      const data = await res.json();
      sec.hidden = false;
      this.renderTimers(data.timers || []);
    } catch (_) {
      sec.hidden = true;
    }
  }

  renderTimers(timers) {
    const listEl = document.getElementById('timers-list');
    if (!listEl) return;
    listEl.textContent = '';
    if (!timers.length) {
      const p = document.createElement('p');
      p.className = 'field-note';
      p.textContent = t('timers.empty') || 'Brak aktywnych minutników.';
      listEl.appendChild(p);
      return;
    }

    timers.forEach((tm) => {
      const item = document.createElement('div');
      item.className = 'timer-item';
      const info = document.createElement('div');
      info.className = 'timer-info';
      const strong = document.createElement('strong');
      strong.textContent = tm.label || t('timers.untitled') || 'Minutnik';
      const due = document.createElement('span');
      due.className = 'timer-due';
      const dueStr = tm.due ? new Date(tm.due).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '';
      due.textContent = ` (${tm.kind || 'timer'}, ${dueStr})`;
      info.append(strong, due);

      const delBtn = document.createElement('button');
      delBtn.type = 'button';
      delBtn.className = 'btn-del';
      delBtn.textContent = '×';
      delBtn.title = t('timers.delete') || 'Usuń minutnik';
      delBtn.addEventListener('click', async () => {
        try {
          const res = await fetch(`/api/timers/${tm.id}`, { method: 'DELETE' });
          if (res.ok) await this.loadTimers();
        } catch (_) {}
      });

      item.append(info, delBtn);
      listEl.appendChild(item);
    });
  }

  // --- Devices ---
  async loadDevices() {
    const sec = document.getElementById('sec-devices');
    if (!sec) return;
    try {
      const res = await fetch('/api/devices');
      if (res.status === 404 || !res.ok) {
        sec.hidden = true;
        return;
      }
      const data = await res.json();
      sec.hidden = false;
      this.renderDevices(data.devices || []);
    } catch (_) {
      sec.hidden = true;
    }
  }

  renderDevices(devices) {
    const listEl = document.getElementById('devices-list');
    if (!listEl) return;
    listEl.textContent = '';
    if (!devices.length) {
      const p = document.createElement('p');
      p.className = 'field-note';
      p.textContent = t('devices.empty') || 'Brak skonfigurowanych urządzeń.';
      listEl.appendChild(p);
      return;
    }

    devices.forEach((dev) => {
      const card = document.createElement('div');
      card.className = `device-card ${dev.online ? 'is-online' : 'is-offline'}`;

      const head = document.createElement('div');
      head.className = 'device-head';
      const dot = document.createElement('span');
      dot.className = 'device-dot';
      const name = document.createElement('strong');
      name.textContent = dev.name || dev.id;
      const statusBadge = document.createElement('span');
      statusBadge.className = 'device-status';
      statusBadge.textContent = dev.online ? (t('devices.online') || 'Online') : (t('devices.offline') || 'Offline');
      head.append(dot, name, statusBadge);

      const meta = document.createElement('div');
      meta.className = 'device-meta';
      const seen = dev.last_seen ? new Date(dev.last_seen * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : (t('devices.never') || 'nigdy');
      meta.textContent = `${t('devices.last_seen') || 'Ostatnio widoczny'}: ${seen}`;

      card.append(head, meta);
      listEl.appendChild(card);
    });
  }

  // --- Diagnostics ---
  async loadDiagnostics() {
    const sec = document.getElementById('sec-diagnostics');
    if (!sec) return;
    try {
      const res = await fetch('/api/diagnostics');
      if (res.status === 404 || !res.ok) {
        sec.hidden = true;
        return;
      }
      const data = await res.json();
      sec.hidden = false;
      this.renderDiagnostics(data);
    } catch (_) {
      sec.hidden = true;
    }
  }

  renderDiagnostics(data) {
    // GPU
    const gpuEl = document.getElementById('diag-gpu');
    if (gpuEl) {
      const gpu = data.gpu;
      if (!gpu) {
        gpuEl.textContent = t('diag.no_gpu') || 'Brak dedykowanej karty GPU / Niedostępne';
      } else {
        gpuEl.textContent = `${gpu.name} · VRAM: ${Math.round(gpu.memory_used_mb)} / ${Math.round(gpu.memory_total_mb)} MiB (${gpu.utilization_pct || 0}% gpu, ${gpu.temperature_c || 0}°C)`;
      }
    }

    // Latency Stages
    const stagesEl = document.getElementById('diag-stages');
    if (stagesEl) {
      const s = data.stages || {};
      const stt = s.stt_ms ? `STT: ${s.stt_ms.p50}ms` : '';
      const rtr = s.router_ms ? `Router: ${s.router_ms.p50}ms` : '';
      const llm = s.llm_ttft_ms ? `TTFT: ${s.llm_ttft_ms.p50}ms` : '';
      const tts = s.tts_first_ms ? `TTS: ${s.tts_first_ms.p50}ms` : '';
      const e2e = s.e2e_first_audio_ms ? `E2E: ${s.e2e_first_audio_ms.p50}ms` : '';
      const parts = [stt, rtr, llm, tts, e2e].filter(Boolean);
      stagesEl.textContent = parts.length ? parts.join(' · ') : (t('diag.no_stages') || 'Oczekiwanie na tury rozmowy…');
    }

    // Wake Log
    const wakesEl = document.getElementById('diag-wakes');
    if (wakesEl) {
      wakesEl.textContent = '';
      const wakes = data.wakes || [];
      if (!wakes.length) {
        const p = document.createElement('p');
        p.className = 'field-note';
        p.textContent = t('diag.no_wakes') || 'Brak zarejestrowanych wybudzeń.';
        wakesEl.appendChild(p);
      } else {
        wakes.slice(-10).reverse().forEach((w) => {
          const row = document.createElement('div');
          row.className = `wake-row ${w.accepted ? 'wake-ok' : 'wake-rej'}`;
          const badge = document.createElement('span');
          badge.className = 'wake-badge';
          badge.textContent = w.accepted ? '✓' : '✗';
          const info = document.createElement('span');
          info.className = 'wake-info';
          const time = new Date((w.ts || 0) * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
          info.textContent = `${time} · ${w.device_id || 'dev'} [${w.word || 'wake'} score=${w.score ?? '-'}] ${w.reason || ''}`;
          row.append(badge, info);
          wakesEl.appendChild(row);
        });
      }
    }
  }
}
