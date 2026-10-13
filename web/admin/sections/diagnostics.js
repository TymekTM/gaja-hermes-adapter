// Diagnostyka: /health summary, GPU and wake log from /api/diagnostics, latency from /metrics.

import { h, icon, clear, pageHead, errorState, fmtMs, formatClock } from '../ui.js';

const STAGES = [
  ['stt_ms', 'Rozpoznanie mowy (STT)'],
  ['router_ms', 'Router'],
  ['tool_select_ms', 'Wybór narzędzia'],
  ['tool_ms', 'Narzędzie'],
  ['llm_ttft_ms', 'Pierwszy token LLM'],
  ['first_sentence_ms', 'Pierwsze zdanie'],
  ['tts_first_ms', 'Pierwszy dźwięk TTS'],
  ['e2e_first_audio_ms', 'Od końca mowy do dźwięku'],
];
const KIND_LABELS = { audio: 'mowa', text: 'tekst', result: 'wynik Hermesa' };
const STATES = { ok: 'ok', warn: 'warn', bad: 'bad', off: '' };

export default {
  id: 'diagnostyka',
  title: 'Diagnostyka',
  icon: 'pulse',
  render(container, api) {
    const body = h('div.a-diag');
    const stamp = h('span.a-muted.a-tabular');
    const refresh = h('button.btn.a-sm', { type: 'button' }, icon('refresh'), 'Odśwież');
    let kind = 'all';

    async function load() {
      refresh.disabled = true;
      const [health, diag, metrics] = await Promise.allSettled([api.get('/health'), api.get('/api/diagnostics'), api.get('/metrics')]);
      refresh.disabled = false;
      stamp.textContent = `odświeżono ${formatClock(new Date())}`;
      if (health.status === 'rejected' && diag.status === 'rejected' && metrics.status === 'rejected') {
        clear(body).append(errorState(`Adapter nie odpowiada: ${health.reason.message}`, load));
        return;
      }
      clear(body).append(
        health.status === 'fulfilled' ? servicesBlock(health.value) : failed('Stan usług', health.reason),
        diag.status === 'fulfilled' ? gpuBlock(diag.value.gpu) : failed('GPU', diag.reason),
        metrics.status === 'fulfilled' ? latencyBlock(metrics.value) : failed('Opóźnienia', metrics.reason),
        diag.status === 'fulfilled' ? wakeBlock(diag.value.wakes || []) : null,
      );
    }

    refresh.addEventListener('click', load);

    function failed(title, err) {
      return h('section.a-block', h('h2.a-h2', title), h('p.a-muted', `Brak danych: ${err.message}`));
    }

    // ---- services
    function servicesBlock(hl) {
      const cards = [];
      cards.push(status('LLM', hl.llm ? 'ok' : 'bad', hl.llm ? 'działa' : 'nie odpowiada'));
      cards.push(status('Hermes', hl.hermes ? 'ok' : 'bad', hl.hermes ? 'gotowy' : 'niegotowy', `backend: ${hl.hermes_backend}`));
      const sttBits = [`silnik: ${hl.stt_engine}`, `urządzenie: ${hl.stt_device}`];
      if (hl.stt_requested && hl.stt_requested !== hl.stt_engine) sttBits.push(`żądany: ${hl.stt_requested}`);
      if (hl.stt_fallback) sttBits.push(`powód awarii: ${hl.stt_fallback}`);
      cards.push(status('Rozpoznawanie mowy', hl.stt_fallback || hl.stt_degraded ? 'warn' : 'ok', hl.stt_fallback ? 'tryb awaryjny' : hl.stt_degraded ? 'ograniczone' : 'działa', sttBits.join(' · ')));
      const loaded = new Set(hl.engines_loaded || []);
      cards.push(status('Synteza mowy', loaded.size ? 'ok' : 'warn', `${loaded.size} z ${(hl.engines || []).length} silników w pamięci`,
        (hl.engines || []).map((e) => `${e}${loaded.has(e) ? ' ✓' : ''}`).join(' · '),
        Object.entries(hl.engine_by_language || {}).map(([l, e]) => `${l}: ${e}`).join(' · ')));
      const d = hl.devices || {};
      cards.push(status('Urządzenia głosowe', d.enabled ? (d.listening ? 'ok' : 'warn') : 'off', !d.enabled ? 'wyłączone' : d.listening ? `${(d.connected || []).length} połączonych` : 'nie nasłuchuje', (d.connected || []).join(', ')));
      const r = hl.router || {};
      cards.push(status('Router', r.enabled ? 'ok' : 'off', r.enabled ? 'włączony' : 'wyłączony', r.enabled ? `próg ${r.threshold}${r.tools ? ' · narzędzia włączone' : ''}` : ''));
      return h('section.a-block', h('h2.a-h2', 'Stan usług'), h('div.a-status-grid', cards));
    }

    function status(title, state, text, ...details) {
      return h(`div.a-status.${STATES[state] || ''}`,
        h('div.a-status-head', h('span.a-dot', { 'aria-hidden': 'true' }), h('span.a-status-title', title)),
        h('p.a-status-text', text),
        details.filter(Boolean).map((t) => h('p.a-status-detail', t)));
    }

    // ---- GPU
    function gpuBlock(gpu) {
      if (!gpu) return h('section.a-block', h('h2.a-h2', 'GPU'), h('p.a-muted', 'Brak danych: nvidia-smi jest niedostępne.'));
      const used = gpu.memory_used_mb;
      const total = gpu.memory_total_mb;
      const pct = total ? Math.round((used / total) * 100) : 0;
      return h('section.a-block', h('h2.a-h2', 'GPU'),
        h('div.a-gpu',
          h('p.a-gpu-name', gpu.name, h('span.a-muted', ` · sterownik ${gpu.driver_version}`)),
          h('div.a-meter', { role: 'meter', 'aria-label': 'Pamięć GPU', 'aria-valuemin': '0', 'aria-valuemax': String(Math.round(total)), 'aria-valuenow': String(Math.round(used)) },
            h(`div.a-meter-fill${pct >= 90 ? '.hot' : ''}`, { style: `width:${pct}%` })),
          h('p.a-gpu-stats.a-tabular', `VRAM ${Math.round(used)} / ${Math.round(total)} MiB (${pct}%) · GPU ${Math.round(gpu.utilization_pct)}% · ${Math.round(gpu.temperature_c)}°C`)));
    }

    // ---- latency
    function latencyBlock(m) {
      const kinds = Object.keys(m.by_kind || {});
      if (kind !== 'all' && !kinds.includes(kind)) kind = 'all';
      const src = kind === 'all' ? m : m.by_kind[kind];
      const turns = src.turns ?? m.turns;
      const wrap = h('section.a-block', h('h2.a-h2', 'Opóźnienia ', h('span.a-count', `${m.turns} tur`)));
      if (kinds.length > 1) {
        wrap.append(h('div.seg.a-kind-seg', { role: 'radiogroup', 'aria-label': 'Rodzaj tury' },
          ['all', ...kinds].map((k) => h('label', h('input', { type: 'radio', name: 'kind', value: k, checked: k === kind, onchange: () => { kind = k; load(); } }), h('span', k === 'all' ? 'wszystkie' : KIND_LABELS[k] || k)))));
      }
      if (!m.turns) {
        wrap.append(h('p.a-muted', 'Brak zarejestrowanych tur. Statystyki pojawią się po pierwszej rozmowie.'));
      } else {
        const rows = STAGES.filter(([key]) => src[key]).map(([key, label]) =>
          h('tr', h('th', { scope: 'row' }, label), h('td.a-tabular', fmtMs(src[key].p50)), h('td.a-tabular', fmtMs(src[key].p90))));
        wrap.append(rows.length
          ? h('div.a-table-wrap', h('table.a-table', h('thead', h('tr', h('th', 'Etap'), h('th', 'Mediana'), h('th', 'P90'))), h('tbody', rows)))
          : h('p.a-muted', `Brak pomiarów dla tego rodzaju tur (${turns}).`));
      }
      const counters = Object.entries(m.counters || {});
      if (counters.length) wrap.append(h('ul.a-chips.a-counters', counters.map(([k, v]) => h('li.a-chip', `${k}: `, h('b.a-tabular', String(v))))));
      return wrap;
    }

    // ---- wake log
    function wakeBlock(wakes) {
      const recent = wakes.slice(-10).reverse();
      return h('section.a-block', h('h2.a-h2', 'Ostatnie wybudzenia'),
        recent.length
          ? h('ul.a-rows', recent.map((w) => h('li.a-row',
              h(`span.a-badge${w.accepted ? '.ok' : '.warn'}`, w.accepted ? 'przyjęte' : 'odrzucone'),
              h('div.a-row-main',
                h('p.a-row-title', `${w.device_id || 'urządzenie'} · ${w.word || 'wybudzenie'}`),
                h('p.a-row-sub.a-tabular', `${formatClock(new Date((w.ts || 0) * 1000))} · wynik ${w.score ?? '–'}${w.reason ? ` · ${w.reason}` : ''}${w.transcript ? ` · „${w.transcript}”` : ''}`)))))
          : h('p.a-muted', 'Brak zarejestrowanych wybudzeń.'));
    }

    container.append(pageHead('Diagnostyka', 'Stan usług, GPU i opóźnienia pipeline’u.', h('div.a-head-tools', stamp, refresh)), body);
    load();
  },
};
