// Wake-word sample recorder (/record.html).
//
// Raw microphone (echo cancellation / noise suppression / AGC off) -> capture-processor.js (16 kHz PCM16
// frames) -> WAV in the browser -> POST /api/wakeword/samples (gaja/wakeword_samples.py).
// The script of prompts lives in record-script.js. Progress is counted on the server per session,
// so a reload (or another device) resumes where the session stopped.

import { AudioController } from './audio.js';
import { t, setUiLang, resolveUiLang, applyStatic } from './i18n.js';
import { SECTIONS, MAX_TAKE_S, MAX_AMBIENT_S, AMBIENT_CHUNK_S } from './record-script.js';

const $ = (id) => document.getElementById(id);
const SR = 16000;
const TRIM_MS = 100; // drops the click of the record/stop key; the speaker waits ~0.5 s anyway
const CLIP_PEAK = 0.99;
const SILENT_PEAK = 0.01; // about -40 dBFS
const SHORT_S = 0.8;
const AMBIENT_DONE_FRACTION = 0.5; // an ambient prompt counts as done after half the target length

// ---------------------------------------------------------------------------------------------
// settings (shared with the main app: localStorage['gaja.settings'])

function loadSettings() {
  try { return JSON.parse(localStorage.getItem('gaja.settings') || '{}') || {}; } catch (_) { return {}; }
}
function saveSetting(key, value) {
  try {
    const all = loadSettings();
    all[key] = value;
    localStorage.setItem('gaja.settings', JSON.stringify(all));
  } catch (_) { /* private mode */ }
}
const settings = { mic: '', lang: 'pl', ...loadSettings() };
setUiLang(resolveUiLang(settings.lang));
applyStatic();
document.title = `${t('rec.title')} · Gaja`;

function store(key, value) {
  try {
    if (value === undefined) return localStorage.getItem(key);
    localStorage.setItem(key, value);
  } catch (_) { /* private mode */ }
  return null;
}

// ---------------------------------------------------------------------------------------------
// state

const ALL_PROMPTS = SECTIONS.flatMap((s) => s.prompts.map((p) => ({ ...p, section: s })));
const st = {
  mode: 'off', // off | ready | recording | review | saving
  session: '',
  sessions: [],
  progress: {}, // label -> prompt_id -> { count, duration_s, next_take }
  cur: ALL_PROMPTS[0],
  // audio graph
  stream: null, ctx: null, node: null, source: null, silent: null,
  micLabel: '',
  // recording
  frames: [], n: 0, t0: 0, chunkFrames: [], chunkN: 0, chunksSent: 0, ambientSamples: 0,
  take: null, // pending take under review: { pcm, blob, url, peak, rms, clipped, dur }
  lastSaved: [], // stack of { label, prompt_id, take } for the undo button
  failed: [], // ambient chunks that could not be uploaded
  uploading: Promise.resolve(),
  wakeLock: null,
  audioEl: null,
};
const mic = new AudioController(); // only used for the device list and openMicStream

const isAmbient = (p = st.cur) => !!p.section.ambient;
const prog = (p = st.cur) => (st.progress[p.section.label] || {})[p.id] || { count: 0, duration_s: 0, next_take: 0 };
const fmtTime = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
const toDb = (x) => 20 * Math.log10(Math.max(x, 1e-5));
const dbText = (x) => toDb(x).toFixed(1);

function promptFraction(p) {
  const g = prog(p);
  return isAmbient(p) ? Math.min(1, g.duration_s / p.target) : Math.min(1, g.count / p.target);
}
function promptDone(p) {
  const g = prog(p);
  return isAmbient(p) ? g.duration_s >= p.target * AMBIENT_DONE_FRACTION : g.count >= p.target;
}

// ---------------------------------------------------------------------------------------------
// small UI helpers

let toastTimer = 0;
function toast(text, kind = 'info') {
  const el = $('toast');
  el.textContent = text;
  el.className = `toast${kind === 'error' ? '' : ' info'}`;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, kind === 'error' ? 5000 : 1800);
}
function setStatus(text) { $('rec-status').textContent = text || ''; }
function setWarn(text) {
  const el = $('rec-warn');
  el.textContent = text || '';
  el.hidden = !text;
}

// ---------------------------------------------------------------------------------------------
// server

async function api(path, init) {
  let res;
  try {
    res = await fetch(path, init);
  } catch (err) {
    throw Object.assign(new Error(t('rec.err_server', { err: err.message || 'network' })), { network: true });
  }
  if (!res.ok) {
    let detail = '';
    try { detail = (await res.json()).detail || ''; } catch (_) { /* not JSON */ }
    throw Object.assign(new Error(`${res.status} ${typeof detail === 'string' ? detail : JSON.stringify(detail)}`.trim()), { status: res.status });
  }
  return res.json();
}

async function refreshProgress() {
  try {
    const data = await api(`/api/wakeword/progress?session=${encodeURIComponent(st.session)}`);
    st.progress = data.labels || {};
  } catch (err) {
    setStatus(err.message);
  }
  render();
}

async function refreshSessions() {
  try { st.sessions = (await api('/api/wakeword/sessions')).sessions || []; } catch (_) { st.sessions = []; }
  renderSessions();
}

function newSessionId() {
  const d = new Date();
  const p = (n) => String(n).padStart(2, '0');
  return `s-${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}

async function uploadWav(blob, p, take, seconds) {
  const q = new URLSearchParams({
    session: st.session, label: p.section.label, prompt_id: p.id, take: String(take), mic: st.micLabel || '',
  });
  const res = await api(`/api/wakeword/samples?${q}`, { method: 'POST', headers: { 'Content-Type': 'audio/wav' }, body: blob });
  const entry = ((st.progress[p.section.label] ||= {})[p.id] ||= { count: 0, duration_s: 0, next_take: 0 });
  entry.count = res.count;
  entry.next_take = res.next_take;
  entry.duration_s = (entry.duration_s || 0) + (res.duration_s || seconds || 0);
  return res;
}

/** Upload with the next free take number; one retry on 409 (someone else took the number) after re-reading progress. */
async function uploadNext(blob, p, seconds) {
  for (let attempt = 0; ; attempt++) {
    const take = prog(p).next_take;
    try {
      return await uploadWav(blob, p, take, seconds);
    } catch (err) {
      if (err.status !== 409 || attempt >= 2) throw err;
      await refreshProgress();
    }
  }
}

// ---------------------------------------------------------------------------------------------
// WAV

function concat(frames, total) {
  const out = new Int16Array(total);
  let o = 0;
  for (const f of frames) { out.set(f, o); o += f.length; }
  return out;
}

function encodeWav(pcm) {
  const header = new ArrayBuffer(44);
  const v = new DataView(header);
  const ascii = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  ascii(0, 'RIFF'); v.setUint32(4, 36 + pcm.byteLength, true); ascii(8, 'WAVE');
  ascii(12, 'fmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, SR, true); v.setUint32(28, SR * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  ascii(36, 'data'); v.setUint32(40, pcm.byteLength, true);
  return new Blob([header, pcm], { type: 'audio/wav' });
}

function analyse(pcm) {
  let peak = 0;
  let sum = 0;
  for (let i = 0; i < pcm.length; i++) {
    const a = Math.abs(pcm[i]);
    if (a > peak) peak = a;
    sum += pcm[i] * pcm[i];
  }
  return { peak: peak / 32768, rms: Math.sqrt(sum / Math.max(1, pcm.length)) / 32768 };
}

function qualityWarning({ peak }, seconds, short = true) {
  if (peak < SILENT_PEAK) return t('rec.warn_silent');
  if (peak >= CLIP_PEAK) return t('rec.warn_clipped', { db: dbText(peak) });
  if (short && seconds < SHORT_S) return t('rec.warn_short');
  return '';
}

// ---------------------------------------------------------------------------------------------
// microphone + capture

const lvl = { peak: 0, hold: 0, holdAt: 0 };

async function openMic() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) { setStatus(t('rec.err_insecure')); return; }
  try {
    const stream = await mic.openMicStream(settings.mic, { raw: true });
    if (mic.micFellBack) { settings.mic = ''; saveSetting('mic', ''); }
    const ctx = new (window.AudioContext || window.webkitAudioContext)({ latencyHint: 'interactive' });
    await ctx.resume();
    await ctx.audioWorklet.addModule('capture-processor.js');
    const track = stream.getAudioTracks()[0];
    const source = ctx.createMediaStreamSource(stream);
    const node = new AudioWorkletNode(ctx, 'mic-capture-processor', { numberOfOutputs: 1, outputChannelCount: [1] });
    node.port.onmessage = (e) => { if (e.data && e.data.type === 'audio') onFrame(new Int16Array(e.data.pcm16)); };
    const silent = ctx.createGain();
    silent.gain.value = 0;
    source.connect(node);
    node.connect(silent);
    silent.connect(ctx.destination);
    Object.assign(st, { stream, ctx, node, source, silent, micLabel: track ? track.label : '' });
    track.addEventListener('ended', () => { if (st.stream === stream) { closeMic(); setStatus(t('rec.err_mic')); } });
    st.mode = 'ready';
    setStatus('');
    renderMics();
    requestAnimationFrame(drawMeter);
  } catch (err) {
    closeMic();
    setStatus(err && err.name === 'NotAllowedError' ? t('rec.err_denied') : t('rec.err_mic'));
  }
  render();
}

function closeMic() {
  try { st.node && st.node.disconnect(); st.source && st.source.disconnect(); } catch (_) { /* gone */ }
  if (st.stream) st.stream.getTracks().forEach((tr) => tr.stop());
  if (st.ctx) st.ctx.close().catch(() => {});
  Object.assign(st, { stream: null, ctx: null, node: null, source: null, silent: null });
  if (st.mode === 'ready' || st.mode === 'recording') st.mode = 'off';
  lvl.peak = lvl.hold = 0;
}

function onFrame(frame) {
  let peak = 0;
  for (let i = 0; i < frame.length; i++) { const a = Math.abs(frame[i]); if (a > peak) peak = a; }
  lvl.peak = Math.max(lvl.peak, peak / 32768);
  if (st.mode !== 'recording') return;
  if (isAmbient()) {
    st.chunkFrames.push(frame);
    st.chunkN += frame.length;
    st.n += frame.length;
    if (st.chunkN >= AMBIENT_CHUNK_S * SR) cutAmbientChunk();
    if (st.n >= MAX_AMBIENT_S * SR) stopRecording();
  } else {
    st.frames.push(frame);
    st.n += frame.length;
    if (st.n >= MAX_TAKE_S * SR) stopRecording();
  }
}

function drawMeter() {
  if (!st.stream) { $('meter-fill').style.transform = 'scaleX(0)'; return; }
  const now = performance.now();
  const fraction = (x) => Math.min(1, Math.max(0, (toDb(x) + 60) / 60));
  const level = lvl.peak;
  lvl.peak *= 0.82; // fast attack (frames raise it), smooth release
  if (level >= lvl.hold || now - lvl.holdAt > 900) { lvl.hold = level; lvl.holdAt = now; }
  const f = fraction(level);
  $('meter-fill').style.transform = `scaleX(${f})`;
  $('meter-hold').style.left = `${fraction(lvl.hold) * 100}%`;
  $('meter').dataset.zone = level >= CLIP_PEAK ? 'clip' : level < SILENT_PEAK ? 'low' : 'ok';
  $('meter').setAttribute('aria-valuenow', String(Math.round(toDb(level))));
  if (st.mode === 'recording') {
    const secs = st.n / SR;
    setStatus(isAmbient()
      ? t('rec.ambient_status', { n: st.chunksSent, time: fmtTime(secs) })
      : `${t('rec.recording')} ${secs.toFixed(1)} s`);
  }
  requestAnimationFrame(drawMeter);
}

async function holdWakeLock() {
  try { if (navigator.wakeLock) st.wakeLock = await navigator.wakeLock.request('screen'); } catch (_) { /* optional */ }
}
function releaseWakeLock() {
  if (st.wakeLock) { st.wakeLock.release().catch(() => {}); st.wakeLock = null; }
}

// ---------------------------------------------------------------------------------------------
// recording flow

function startRecording() {
  if (st.mode !== 'ready') return;
  stopPlayback();
  Object.assign(st, { frames: [], n: 0, chunkFrames: [], chunkN: 0, chunksSent: 0, take: null });
  setWarn('');
  setStatus(t('rec.recording'));
  st.mode = 'recording';
  holdWakeLock();
  render();
}

function stopRecording() {
  if (st.mode !== 'recording') return;
  releaseWakeLock();
  if (isAmbient()) {
    cutAmbientChunk(true);
    st.mode = 'saving';
    render();
    st.uploading.then(() => {
      st.mode = st.stream ? 'ready' : 'off';
      setStatus(st.failed.length ? t('rec.unsent', { n: st.failed.length }) : t('rec.saved'));
      refreshProgress();
    });
    return;
  }
  let pcm = concat(st.frames, st.n);
  const trim = Math.round((TRIM_MS / 1000) * SR);
  if (pcm.length > 2 * trim + 0.3 * SR) pcm = pcm.slice(trim, pcm.length - trim);
  const stats = analyse(pcm);
  const dur = pcm.length / SR;
  const blob = encodeWav(pcm);
  st.take = { pcm, blob, url: URL.createObjectURL(blob), dur, ...stats };
  st.mode = 'review';
  setWarn(qualityWarning(stats, dur));
  setStatus(t('rec.take_info', { s: dur.toFixed(1), db: dbText(stats.peak) }));
  drawWave(pcm);
  render();
  if ($('rec-autoplay').checked) playTake();
}

function discardTake() {
  if (st.mode !== 'review') return;
  stopPlayback();
  if (st.take) URL.revokeObjectURL(st.take.url);
  st.take = null;
  st.mode = 'ready';
  setWarn('');
  setStatus('');
  render();
}

async function saveTake() {
  if (st.mode !== 'review' || !st.take) return;
  const take = st.take;
  const p = st.cur;
  st.mode = 'saving';
  setStatus(t('rec.saving'));
  render();
  try {
    const res = await uploadNext(take.blob, p, take.dur);
    st.lastSaved.push({ label: p.section.label, prompt_id: p.id, take: res.take, prompt: p });
    URL.revokeObjectURL(take.url);
    st.take = null;
    st.mode = 'ready';
    setStatus('');
    setWarn('');
    toast(t('rec.saved'));
    if (promptDone(p)) advance();
    refreshSessions();
  } catch (err) {
    st.mode = 'review';
    setStatus(t('rec.save_failed', { err: err.message }));
  }
  render();
}

async function undoLast() {
  const last = st.lastSaved[st.lastSaved.length - 1];
  if (!last || st.mode !== 'ready') return;
  try {
    await api(`/api/wakeword/samples/${encodeURIComponent(st.session)}/${last.label}/${last.prompt_id}/${last.take}`, { method: 'DELETE' });
    st.lastSaved.pop();
    st.cur = last.prompt;
    toast(t('rec.undone'));
    await refreshProgress();
    refreshSessions();
  } catch (err) {
    setStatus(t('rec.save_failed', { err: err.message }));
  }
  render();
}

// ambient: chunks of up to AMBIENT_CHUNK_S are uploaded in the background, one at a time
function cutAmbientChunk(final = false) {
  if (!st.chunkN || (final && st.chunkN < 0.5 * SR)) { st.chunkFrames = []; st.chunkN = 0; return; }
  const pcm = concat(st.chunkFrames, st.chunkN);
  st.chunkFrames = [];
  st.chunkN = 0;
  const stats = analyse(pcm);
  const warn = qualityWarning(stats, pcm.length / SR, false);
  if (warn) setWarn(warn);
  queueChunk({ blob: encodeWav(pcm), prompt: st.cur, seconds: pcm.length / SR });
}

function queueChunk(chunk) {
  st.uploading = st.uploading.then(async () => {
    for (let attempt = 0; attempt < 3; attempt++) {
      try {
        await uploadNext(chunk.blob, chunk.prompt, chunk.seconds);
        st.chunksSent += 1;
        render();
        return;
      } catch (err) {
        if (err.status && err.status < 500 && err.status !== 409) break; // a rejected chunk will not get better
        await new Promise((r) => setTimeout(r, 1000 * (attempt + 1) ** 2));
      }
    }
    st.failed.push(chunk);
    render();
  });
}

function retryFailed() {
  const chunks = st.failed.splice(0);
  chunks.forEach(queueChunk);
  render();
  st.uploading.then(() => { refreshProgress(); });
}

// ---------------------------------------------------------------------------------------------
// playback + waveform

function playTake() {
  if (!st.take) return;
  stopPlayback();
  st.audioEl = new Audio(st.take.url);
  st.audioEl.play().catch(() => {});
}
function stopPlayback() {
  if (st.audioEl) { st.audioEl.pause(); st.audioEl = null; }
}

function drawWave(pcm) {
  const canvas = $('wave');
  const g = canvas.getContext('2d');
  const { width: w, height: h } = canvas;
  g.clearRect(0, 0, w, h);
  g.fillStyle = getComputedStyle(canvas).color; // 'currentColor' would resolve against the canvas backing store's default (black)
  // normalised so a quiet take is still readable; the level meter and the dBFS text carry the real loudness
  let top = 0;
  for (let i = 0; i < pcm.length; i++) { const a = Math.abs(pcm[i]); if (a > top) top = a; }
  const gain = top > 0 ? Math.min(32, (0.9 * 32768) / top) : 1;
  const step = Math.max(1, Math.floor(pcm.length / w));
  for (let x = 0; x < w; x++) {
    let lo = 0;
    let hi = 0;
    for (let i = x * step; i < Math.min(pcm.length, (x + 1) * step); i++) { if (pcm[i] < lo) lo = pcm[i]; if (pcm[i] > hi) hi = pcm[i]; }
    const y1 = h / 2 - Math.min(1, (hi * gain) / 32768) * (h / 2);
    const y2 = h / 2 - Math.max(-1, (lo * gain) / 32768) * (h / 2);
    g.fillRect(x, y1, 1, Math.max(1, y2 - y1));
  }
}

// ---------------------------------------------------------------------------------------------
// rendering

function advance() {
  const i = ALL_PROMPTS.indexOf(st.cur);
  const next = [...ALL_PROMPTS.slice(i + 1), ...ALL_PROMPTS.slice(0, i)].find((p) => !promptDone(p));
  if (next) selectPrompt(next);
}

function selectPrompt(p) {
  if (st.mode === 'recording' || st.mode === 'saving') return;
  if (st.mode === 'review') discardTake();
  st.cur = p;
  store('gaja.rec.prompt', p.id);
  setWarn('');
  if (st.mode === 'ready') setStatus('');
  render();
}

function render() {
  const p = st.cur;
  const g = prog(p);
  const amb = isAmbient(p);
  document.body.dataset.rec = st.mode;

  // overall
  let done = 0;
  let total = 0;
  for (const q of ALL_PROMPTS) {
    if (isAmbient(q)) { total += 1; done += promptDone(q) ? 1 : 0; } else { total += q.target; done += Math.min(prog(q).count, q.target); }
  }
  $('rec-overall').textContent = t('rec.overall', { done, total });
  $('rec-overall-fill').style.transform = `scaleX(${total ? done / total : 0})`;

  // stage
  const progressText = amb
    ? `${fmtTime(g.duration_s)} / ${fmtTime(p.target)}`
    : `${g.count} / ${p.target}`;
  $('stage-kicker').textContent = `${p.section.title} · ${progressText}`;
  $('stage-say').textContent = p.say;
  $('stage-say').classList.toggle('is-long', p.say.length > 28);
  $('stage-how').textContent = amb ? `${p.how} ${t('rec.ambient_how')}` : p.how;
  $('stage-fill').style.transform = `scaleX(${promptFraction(p)})`;

  // main button
  const btn = $('rec-btn');
  const label = { off: 'rec.mic_on', ready: 'rec.start', recording: 'rec.stop', review: 'rec.start', saving: 'rec.saving' }[st.mode];
  $('rec-btn-label').textContent = t(label);
  btn.disabled = st.mode === 'saving';
  btn.hidden = st.mode === 'review';
  btn.dataset.kind = st.mode === 'off' ? 'mic' : st.mode === 'recording' ? 'stop' : 'rec';
  btn.setAttribute('aria-pressed', String(st.mode === 'recording'));

  // review + extras
  $('rec-review').hidden = st.mode !== 'review';
  $('wave').hidden = st.mode !== 'review';
  $('rec-hint').hidden = amb;
  $('rec-undo').hidden = !(st.lastSaved.length && st.mode === 'ready');
  const retry = $('rec-retry');
  retry.hidden = !st.failed.length;
  retry.textContent = `${t('rec.retry')} (${st.failed.length})`;
  $('rec-mic').disabled = st.mode === 'recording' || st.mode === 'review' || st.mode === 'saving';
  $('rec-session').disabled = st.mode !== 'off' && st.mode !== 'ready';

  renderList();
}

function renderList() {
  const nav = $('rec-list');
  if (!nav.dataset.built) {
    nav.dataset.built = '1';
    for (const s of SECTIONS) {
      const sec = document.createElement('section');
      sec.className = 'rec-section';
      sec.dataset.section = s.id;
      const head = document.createElement('h3');
      head.className = 'field-label';
      const intro = document.createElement('p');
      intro.className = 'field-note';
      intro.textContent = s.intro;
      sec.append(head, intro);
      for (const q of s.prompts) {
        const row = document.createElement('button');
        row.type = 'button';
        row.className = 'rec-row';
        row.dataset.prompt = q.id;
        row.innerHTML = '<span class="rec-row-say"></span><span class="rec-row-count"></span><span class="rec-row-how"></span><i class="rec-row-bar"><b></b></i>';
        row.addEventListener('click', () => selectPrompt(ALL_PROMPTS.find((x) => x.id === q.id)));
        sec.append(row);
      }
      nav.append(sec);
    }
  }
  for (const s of SECTIONS) {
    const sec = nav.querySelector(`[data-section="${s.id}"]`);
    let done = 0;
    let total = 0;
    for (const q of s.prompts) {
      const g = prog({ ...q, section: s });
      const amb = !!s.ambient;
      const frac = amb ? Math.min(1, g.duration_s / q.target) : Math.min(1, g.count / q.target);
      total += amb ? 1 : q.target;
      done += amb ? (frac >= AMBIENT_DONE_FRACTION ? 1 : 0) : Math.min(g.count, q.target);
      const row = sec.querySelector(`[data-prompt="${q.id}"]`);
      row.querySelector('.rec-row-say').textContent = q.say;
      row.querySelector('.rec-row-how').textContent = q.how;
      row.querySelector('.rec-row-count').textContent = amb ? `${fmtTime(g.duration_s)} / ${fmtTime(q.target)}` : `${g.count} / ${q.target}`;
      row.querySelector('b').style.transform = `scaleX(${frac})`;
      row.classList.toggle('is-done', amb ? frac >= AMBIENT_DONE_FRACTION : g.count >= q.target);
      row.classList.toggle('is-current', q.id === st.cur.id);
      row.setAttribute('aria-current', q.id === st.cur.id ? 'true' : 'false');
      row.disabled = st.mode === 'recording' || st.mode === 'saving';
    }
    sec.querySelector('h3').textContent = `${s.title} · ${done} / ${total}`;
  }
}

function renderSessions() {
  const sel = $('rec-session');
  const known = new Set(st.sessions.map((s) => s.session));
  const opts = st.sessions.map((s) => ({ value: s.session, label: t('rec.session_item', { id: s.session, n: s.total }) }));
  if (!known.has(st.session)) opts.unshift({ value: st.session, label: t('rec.session_item', { id: st.session, n: 0 }) });
  opts.push({ value: '__new__', label: t('rec.session_new') });
  sel.replaceChildren(...opts.map((o) => Object.assign(document.createElement('option'), { value: o.value, textContent: o.label })));
  sel.value = st.session;
}

async function renderMics() {
  const sel = $('rec-mic');
  const { all, real } = await AudioController.microphoneOptions();
  const opts = [{ value: '', label: t('settings.mic_default') }];
  real.forEach((d) => opts.push({ value: d.deviceId, label: d.label || t('settings.mic_unnamed', { n: d.index }) }));
  sel.replaceChildren(...opts.map((o) => Object.assign(document.createElement('option'), { value: o.value, textContent: o.label })));
  sel.value = opts.some((o) => o.value === settings.mic) ? settings.mic : '';
  const note = $('rec-mic-note');
  const track = st.stream && st.stream.getAudioTracks()[0];
  const s = track && track.getSettings ? track.getSettings() : null;
  if (!all.length) note.textContent = t('settings.mic_none');
  else if (s) {
    const onoff = (v) => (v ? 'on' : 'off');
    note.textContent = `${st.micLabel} · AEC ${onoff(s.echoCancellation)} · NS ${onoff(s.noiseSuppression)} · AGC ${onoff(s.autoGainControl)}`;
  } else if (!all.some((d) => d.label)) note.textContent = t('settings.mic_nolabels');
  else note.textContent = '';
}

// ---------------------------------------------------------------------------------------------
// events

function primaryAction() {
  if (st.mode === 'off') openMic();
  else if (st.mode === 'ready') startRecording();
  else if (st.mode === 'recording') stopRecording();
}

$('rec-btn').addEventListener('click', primaryAction);
$('rec-save').addEventListener('click', saveTake);
$('rec-redo').addEventListener('click', discardTake);
$('rec-play').addEventListener('click', playTake);
$('rec-undo').addEventListener('click', undoLast);
$('rec-retry').addEventListener('click', retryFailed);
$('rec-autoplay').checked = store('gaja.rec.autoplay') !== '0';
$('rec-autoplay').addEventListener('change', (e) => store('gaja.rec.autoplay', e.target.checked ? '1' : '0'));

$('rec-mic').addEventListener('change', async (e) => {
  settings.mic = e.target.value;
  saveSetting('mic', settings.mic);
  if (st.stream) { closeMic(); await openMic(); } else renderMics();
});

$('rec-session').addEventListener('change', async (e) => {
  st.session = e.target.value === '__new__' ? newSessionId() : e.target.value;
  store('gaja.rec.session', st.session);
  st.lastSaved = [];
  st.progress = {};
  render();
  renderSessions();
  await refreshProgress();
  const first = ALL_PROMPTS.find((p) => !promptDone(p));
  if (first) selectPrompt(first);
});

document.addEventListener('keydown', (e) => {
  if (e.repeat || e.ctrlKey || e.metaKey || e.altKey) return;
  const tag = (e.target.tagName || '').toLowerCase();
  if (tag === 'select' || tag === 'input' || tag === 'textarea') return;
  if (e.code === 'Space') { e.preventDefault(); primaryAction(); }
  else if (e.key === 'Enter' && st.mode === 'review') { e.preventDefault(); saveTake(); }
  else if ((e.key === 'Backspace' || e.key === 'r' || e.key === 'R') && st.mode === 'review') { e.preventDefault(); discardTake(); }
});
// Space activates a focused button on keyup: the keydown handler above already did the work
document.addEventListener('keyup', (e) => { if (e.code === 'Space' && !/^(select|input|textarea)$/i.test(e.target.tagName || '')) e.preventDefault(); });

window.addEventListener('beforeunload', (e) => {
  if (st.mode === 'recording' || st.mode === 'review' || st.mode === 'saving' || st.failed.length) { e.preventDefault(); e.returnValue = t('rec.leave'); }
});
navigator.mediaDevices && navigator.mediaDevices.addEventListener && navigator.mediaDevices.addEventListener('devicechange', renderMics);

// ---------------------------------------------------------------------------------------------
// start

async function init() {
  st.session = store('gaja.rec.session') || newSessionId();
  store('gaja.rec.session', st.session);
  const saved = ALL_PROMPTS.find((p) => p.id === store('gaja.rec.prompt'));
  if (saved) st.cur = saved;
  render();
  renderSessions();
  renderMics();
  await Promise.all([refreshProgress(), refreshSessions()]);
  if (!saved || promptDone(st.cur)) {
    const first = ALL_PROMPTS.find((p) => !promptDone(p));
    if (first) selectPrompt(first);
  }
  if (ALL_PROMPTS.every(promptDone)) setStatus(t('rec.all_done'));
}
init();
