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
// continuous recording (unattended for hours or days)
const CONT_MAX_PENDING = 30; // chunks waiting for upload (~58 MB); beyond this the oldest is dropped and counted
const CONT_BACKOFF_MS = [2000, 5000, 10000, 30000, 60000]; // upload retry delays, the last one repeats forever
const REACQUIRE_MS = 5000; // how often a lost microphone is retried
const STALL_S = 10; // no audio frames for this long = the capture graph is dead (OS sleep, suspended context)

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
const DEV_POLL_MS = 300; // device take status polling: elapsed time + a live level meter
const DEV_MAX_MS = 120000; // protocol cap (record_start max_ms)
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
  // device recording source (protocol v1.10): '' = this browser, else a device id
  dev: { admin: false, devices: [], source: '', active: null, poll: 0, stopAsked: false, marks: 0, stats: null, lastTake: null },
  // continuous recording: pending = upload queue (pending[0] is in flight), dropped = chunks lost to the cap or rejected
  cont: { pending: [], busy: false, sent: 0, bytes: 0, lastOk: 0, dropped: 0, reacq: 0, err: '', full: false, elapsedMs: 0, wake: null },
  deviceId: '', lost: false, lastFrameAt: 0,
  uploading: Promise.resolve(),
  wakeLock: null,
  audioEl: null,
};
const mic = new AudioController(); // only used for the device list and openMicStream

const isDevSource = () => st.dev.source !== '';
const devTakeMs = (p = st.cur) => Math.min(DEV_MAX_MS, Math.round((isAmbient(p) ? AMBIENT_CHUNK_S : MAX_TAKE_S) * 1000));

const isAmbient = (p = st.cur) => !!p.section.ambient;
const isContinuous = (p = st.cur) => !!p.continuous;
const prog = (p = st.cur) => (st.progress[p.section.label] || {})[p.id] || { count: 0, duration_s: 0, next_take: 0 };
const fmtTime = (s) => (s >= 3600
  ? `${Math.floor(s / 3600)}:${String(Math.floor((s % 3600) / 60)).padStart(2, '0')}:${String(Math.floor(s % 60)).padStart(2, '0')}`
  : `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`);
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
    st.dev.stats = data.device || null;
  } catch (err) {
    setStatus(err.message);
  }
  renderDevStat();
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
// device recording source (protocol v1.10, admin API)

async function refreshDevices() {
  try {
    const session = await api('/api/admin/session');
    st.dev.admin = !!session.logged_in;
  } catch (_) {
    st.dev.admin = false;
  }
  st.dev.devices = [];
  if (st.dev.admin) {
    try {
      const { devices } = await api('/api/admin/record/devices');
      st.dev.devices = (devices || []).filter((d) => d.caps_record);
    } catch (_) { /* keep the browser-only selector */ }
  }
  const busy = st.mode === 'recording' || st.mode === 'saving' || st.mode === 'review';
  if (!busy && !st.dev.devices.some((d) => d.id === st.dev.source)) {
    // The device went away: fall back to the browser mic (never mid-take: the poll still needs the id).
    if (st.dev.source) st.mode = st.stream ? 'ready' : 'off';
    st.dev.source = '';
  }
  renderSources();
  render();
}

function renderSources() {
  const sel = $('rec-source');
  const note = $('rec-source-note');
  const opts = [{ value: '', label: t('rec.source_browser') }];
  st.dev.devices.forEach((d) => opts.push({
    value: d.id,
    label: d.name === d.id ? d.id : `${d.name} (${d.id})`,
  }));
  sel.replaceChildren(...opts.map((o) => Object.assign(document.createElement('option'), { value: o.value, textContent: o.label })));
  sel.value = st.dev.source;
  if (!st.dev.admin) {
    note.innerHTML = '';
    note.append(t('rec.source_admin_note'), ' ');
    const link = document.createElement('a');
    link.href = '/admin';
    link.textContent = t('rec.source_admin_link');
    note.append(link);
  } else if (!st.dev.devices.length) {
    note.textContent = t('rec.source_none');
  } else {
    note.textContent = '';
  }
}

async function startDeviceTake() {
  if (st.mode === 'recording' || st.mode === 'saving' || !st.dev.source) return;
  stopPlayback();
  setWarn('');
  const p = st.cur;
  st.mode = 'saving';
  setStatus(t('rec.dev_starting'));
  render();
  try {
    const res = await api('/api/admin/record/takes', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        device_id: st.dev.source, session: st.session, label: p.section.label, prompt_id: p.id, max_ms: devTakeMs(p),
      }),
    });
    if (res.state === 'error') throw new Error(res.error || 'record_error');
    st.dev.active = res;
    st.dev.stopAsked = false;
    st.dev.marks = 0;
    st.t0 = Date.now();
    st.mode = 'recording';
    setStatus(t('rec.dev_recording'));
    startMeter();
    pollDeviceTake();
  } catch (err) {
    st.mode = isDevSource() || st.stream ? 'ready' : 'off';
    setStatus(t('rec.save_failed', { err: err.message }));
  }
  render();
}

function pollDeviceTake() {
  clearInterval(st.dev.poll);
  st.dev.poll = setInterval(async () => {
    if (!st.dev.active || st.mode !== 'recording') { clearInterval(st.dev.poll); return; }
    let take;
    try {
      take = (await api(`/api/admin/record/takes/${encodeURIComponent(st.dev.source)}`)).take;
    } catch (_) { return; /* transient (device reconnecting): keep polling */ }
    if (!take || take.id !== st.dev.active.id) { clearInterval(st.dev.poll); return; }
    st.dev.active = take;
    st.dev.marks = (take.marks || []).length;
    lvl.peak = Math.max(lvl.peak, Math.pow(10, (take.level_dbfs || -100) / 20));
    const secs = (take.ms || 0) / 1000;
    setStatus(isAmbient()
      ? t('rec.ambient_status', { n: st.chunksSent, time: fmtTime(secs) })
      : `${t('rec.dev_recording')} ${secs.toFixed(1)} s`);
    if (take.state === 'done' || take.state === 'error') finishDeviceTake(take);
  }, DEV_POLL_MS);
}

async function stopDeviceTake() {
  if (st.mode !== 'recording' || !st.dev.active) return;
  st.dev.stopAsked = true;
  st.mode = 'saving';
  setStatus(t('rec.saving'));
  render();
  clearInterval(st.dev.poll);
  try {
    const done = await api(`/api/admin/record/takes/${encodeURIComponent(st.dev.source)}/stop`, { method: 'POST' });
    if (done.state === 'done' || done.state === 'error') finishDeviceTake(done);
    else { // the server is still storing the take; the poll finishes it
      st.dev.active = done;
      st.mode = 'recording';
      pollDeviceTake();
    }
  } catch (err) {
    st.mode = 'recording'; // the poll loop finishes the take when the device answers
    pollDeviceTake();
    setStatus(t('rec.save_failed', { err: err.message }));
    render();
  }
}

function finishDeviceTake(take) {
  clearInterval(st.dev.poll);
  const p = st.cur;
  st.dev.active = null;
  st.mode = 'ready';
  if (take.state === 'error') {
    setStatus(t('rec.dev_failed', { err: take.error || 'error' }));
    render();
    return;
  }
  st.chunksSent += 1;
  if (take.result) {
    st.lastSaved.push({ label: p.section.label, prompt_id: p.id, take: take.result.take, prompt: p });
  }
  st.dev.lastTake = take;
  renderDevStat(take);
  if (isContinuous(p) && !st.dev.stopAsked && take.end_reason === 'max') {
    refreshProgress().then(() => startDeviceTake()); // unattended ambient: next chunk right away
    render();
    return;
  }
  setStatus(take.partial ? t('rec.dev_partial') : t('rec.saved'));
  if (promptDone(p)) advance();
  refreshProgress(); // its renderDevStat() re-renders ✓/✗ (st.dev.lastTake) with fresh recall
  refreshSessions();
  render();
}

/** ✓/✗ for a finished wake take plus the running on-device recall for the session. */
function renderDevStat(take = null) {
  const el = $('rec-devstat');
  const stat = st.dev.stats && st.dev.stats.wake;
  const finished = (take || st.dev.lastTake) && (take || st.dev.lastTake).state === 'done';
  if (!finished && !(stat && stat.takes > 0)) { el.hidden = true; return; }
  el.hidden = false;
  el.textContent = '';
  el.dataset.detected = '';
  const last = take || st.dev.lastTake;
  if (finished && st.cur.section.label === 'wake') {
    const ok = !!last.device_detected;
    el.dataset.detected = String(ok);
    el.append(ok ? t('rec.dev_detected') : t('rec.dev_missed'));
    if (last.marks && last.marks.length > 1) el.append(` ×${last.marks.length}`);
  }
  if (stat && stat.takes > 0) {
    const pct = Math.round((stat.recall != null ? stat.recall : stat.detected / stat.takes) * 100);
    const small = document.createElement('small');
    small.textContent = ` · ${t('rec.dev_recall', { d: stat.detected, n: stat.takes, p: pct })}`;
    el.append(small);
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

/** Opens the microphone and wires the capture graph into `st`. Throws; the caller decides what a failure means. */
async function buildGraph(deviceId, strict = false) {
  const stream = await mic.openMicStream(deviceId, { raw: true, strict });
  let ctx = null;
  try {
    ctx = new (window.AudioContext || window.webkitAudioContext)({ latencyHint: 'interactive' });
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
    // the concrete device, so that a later re-acquire after an unplug asks for this one even when "default" was chosen
    st.deviceId = (track && track.getSettings && track.getSettings().deviceId) || deviceId || '';
    st.lastFrameAt = performance.now();
    track.addEventListener('ended', () => { if (st.stream === stream) micLost(); });
    track.addEventListener('mute', () => { if (st.stream === stream && st.mode === 'recording' && isContinuous()) setWarn(t('rec.mic_muted')); });
    track.addEventListener('unmute', () => { if (st.stream === stream) setWarn(''); });
    // an OS that suspends the context (sleep, audio-session interruption): ask for it back; the stall watchdog covers a refusal
    ctx.addEventListener('statechange', () => { if (st.ctx === ctx && ctx.state !== 'running' && ctx.state !== 'closed') ctx.resume().catch(() => {}); });
  } catch (err) {
    stream.getTracks().forEach((tr) => tr.stop());
    if (ctx) ctx.close().catch(() => {});
    throw err;
  }
}

async function openMic() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) { setStatus(t('rec.err_insecure')); return; }
  try {
    await buildGraph(settings.mic);
    if (mic.micFellBack) { settings.mic = ''; saveSetting('mic', ''); }
    st.mode = 'ready';
    setStatus('');
    renderMics();
    startMeter();
  } catch (err) {
    closeMic();
    setStatus(err && err.name === 'NotAllowedError' ? t('rec.err_denied') : t('rec.err_mic'));
  }
  render();
}

function closeMic(keepMode = false) {
  try { st.node && st.node.disconnect(); st.source && st.source.disconnect(); } catch (_) { /* gone */ }
  if (st.stream) st.stream.getTracks().forEach((tr) => tr.stop());
  if (st.ctx) st.ctx.close().catch(() => {});
  Object.assign(st, { stream: null, ctx: null, node: null, source: null, silent: null });
  if (!keepMode && (st.mode === 'ready' || st.mode === 'recording')) st.mode = 'off';
  lvl.peak = lvl.hold = 0;
}

// Continuous recording only: the microphone vanished (unplugged, OS sleep, driver reset). Keep the take going:
// upload what was captured, drop the dead graph and try to open the same device again every few seconds.
let reacqTimer = 0;
function micLost() {
  if (st.mode === 'recording' && isContinuous()) {
    closeMic(true);
    cutAmbientChunk(true);
    st.lost = true;
    setWarn(t('rec.mic_lost'));
    scheduleReacquire(1000);
    renderRun();
  } else {
    closeMic();
    setStatus(t('rec.err_mic'));
    render();
  }
}

function scheduleReacquire(ms = REACQUIRE_MS) {
  clearTimeout(reacqTimer);
  reacqTimer = setTimeout(reacquire, ms);
}

async function reacquire() {
  if (!st.lost || st.mode !== 'recording') return;
  try {
    await buildGraph(st.deviceId, true);
  } catch (_) {
    if (st.lost && st.mode === 'recording') scheduleReacquire();
    return;
  }
  st.lost = false;
  if (st.mode !== 'recording') st.mode = 'ready'; // Stop was pressed while the device was opening
  else { st.cont.reacq += 1; toast(t('rec.mic_back')); }
  setWarn('');
  renderMics();
  startMeter();
  render();
}

function onFrame(frame) {
  st.lastFrameAt = performance.now();
  let peak = 0;
  for (let i = 0; i < frame.length; i++) { const a = Math.abs(frame[i]); if (a > peak) peak = a; }
  lvl.peak = Math.max(lvl.peak, peak / 32768);
  if (st.mode !== 'recording') return;
  if (isAmbient()) {
    st.chunkFrames.push(frame);
    st.chunkN += frame.length;
    st.n += frame.length;
    if (st.chunkN >= AMBIENT_CHUNK_S * SR) cutAmbientChunk();
    if (st.n >= MAX_AMBIENT_S * SR && !isContinuous()) stopRecording();
  } else {
    st.frames.push(frame);
    st.n += frame.length;
    if (st.n >= MAX_TAKE_S * SR) stopRecording();
  }
}

let meterRunning = false;
function startMeter() {
  if (meterRunning) return;
  meterRunning = true;
  requestAnimationFrame(drawMeter);
}

function drawMeter() {
  if (!st.stream && !(isDevSource() && st.mode === 'recording')) { meterRunning = false; $('meter-fill').style.transform = 'scaleX(0)'; return; }
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
  if (st.mode === 'recording' && !isContinuous() && st.stream && !isDevSource()) { // the device and continuous runs have their own status line
    const secs = st.n / SR;
    setStatus(isAmbient()
      ? t('rec.ambient_status', { n: st.chunksSent, time: fmtTime(secs) })
      : `${t('rec.recording')} ${secs.toFixed(1)} s`);
  }
  requestAnimationFrame(drawMeter);
}

async function holdWakeLock() {
  if (st.wakeLock || !navigator.wakeLock) return;
  try {
    const lock = await navigator.wakeLock.request('screen');
    lock.addEventListener('release', () => { if (st.wakeLock === lock) st.wakeLock = null; }); // the browser drops it when the tab is hidden
    st.wakeLock = lock;
  } catch (_) { /* optional */ }
}
function releaseWakeLock() {
  const lock = st.wakeLock;
  st.wakeLock = null;
  if (lock) lock.release().catch(() => {});
}

// ---------------------------------------------------------------------------------------------
// recording flow

function startRecording() {
  if (st.mode !== 'ready') return;
  stopPlayback();
  Object.assign(st, { frames: [], n: 0, t0: Date.now(), chunkFrames: [], chunkN: 0, chunksSent: 0, take: null, lost: false });
  if (isContinuous()) Object.assign(st.cont, { sent: 0, bytes: 0, lastOk: 0, dropped: 0, reacq: 0, err: '', elapsedMs: 0 });
  setWarn('');
  setStatus(t('rec.recording'));
  st.mode = 'recording';
  holdWakeLock();
  render();
}

function stopRecording() {
  if (st.mode !== 'recording') return;
  releaseWakeLock();
  if (isContinuous()) {
    // no 'saving' wait: the owner may stop while the server is down, uploads keep retrying in the background
    clearTimeout(reacqTimer);
    st.lost = false;
    st.cont.elapsedMs = Date.now() - st.t0;
    cutAmbientChunk(true);
    st.mode = st.stream ? 'ready' : 'off';
    setWarn('');
    updateContinuousStatus();
    render();
    return;
  }
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
  if (warn || isContinuous()) setWarn(warn); // a continuous run clears the warning once the room is audible again
  queueChunk({ blob: encodeWav(pcm), prompt: st.cur, seconds: pcm.length / SR });
}

function queueChunk(chunk) {
  if (isContinuous(chunk.prompt)) { queueContinuous(chunk); return; }
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

// Continuous recording: an unbounded-time upload queue. A failing upload is retried with backoff and never gives up
// (the server may be restarting for minutes); memory stays bounded because beyond CONT_MAX_PENDING the oldest chunk goes.
function queueContinuous(chunk) {
  const c = st.cont;
  if (c.full) { st.failed.push(chunk); render(); return; }
  c.pending.push(chunk);
  while (c.pending.length > CONT_MAX_PENDING) {
    c.pending.shift().dropped = true;
    c.dropped += 1;
  }
  pumpContinuous();
  renderRun();
}

async function pumpContinuous() {
  const c = st.cont;
  if (c.busy) return;
  c.busy = true;
  try {
    for (let fails = 0; c.pending.length;) {
      const chunk = c.pending[0];
      try {
        await uploadNext(chunk.blob, chunk.prompt, chunk.seconds);
        if (chunk.dropped) c.dropped -= 1; // it was written after all
        else c.pending.shift();
        Object.assign(c, { sent: c.sent + 1, bytes: c.bytes + chunk.blob.size, lastOk: Date.now(), err: '' });
        fails = 0;
        render();
      } catch (err) {
        c.err = err.message;
        if (err.status === 507) { storageFull(); return; }
        if (err.status && err.status < 500 && err.status !== 409) { // rejected for good: do not block the queue behind it
          if (!chunk.dropped) { c.pending.shift(); c.dropped += 1; } // a chunk the cap already dropped is counted once
          continue;
        }
        fails += 1;
        await new Promise((resolve) => {
          const timer = setTimeout(resolve, CONT_BACKOFF_MS[Math.min(fails, CONT_BACKOFF_MS.length) - 1]);
          c.wake = () => { clearTimeout(timer); resolve(); }; // the browser's "online" event ends the wait early
        });
        c.wake = null;
        renderRun();
      }
    }
  } finally {
    c.busy = false;
  }
  if (st.mode !== 'recording') {
    updateContinuousStatus();
    refreshProgress();
    refreshSessions();
  }
  render();
}

/** The server has no room left (HTTP 507): keep the audio for a manual retry, stop recording and say so. */
function storageFull() {
  const c = st.cont;
  c.full = true;
  st.failed.push(...c.pending.splice(0));
  if (st.mode === 'recording') stopRecording();
  setWarn(t('rec.err_full'));
  setStatus(t('rec.err_full'));
  render();
}

function updateContinuousStatus() {
  const n = st.cont.pending.length;
  setStatus(n ? t('rec.cont_pending', { n }) : st.failed.length ? t('rec.unsent', { n: st.failed.length }) : t('rec.saved'));
}

function retryFailed() {
  st.cont.full = false;
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
  $('stage-how').textContent = !amb ? p.how : `${p.how} ${t(isContinuous(p) ? 'rec.continuous_how' : 'rec.ambient_how')}`;
  $('stage-fill').style.transform = `scaleX(${promptFraction(p)})`;

  // main button
  const btn = $('rec-btn');
  const dev = isDevSource();
  const label = dev
    ? { off: 'rec.start', ready: 'rec.start', recording: 'rec.stop', review: 'rec.start', saving: 'rec.saving' }[st.mode]
    : { off: 'rec.mic_on', ready: 'rec.start', recording: 'rec.stop', review: 'rec.start', saving: 'rec.saving' }[st.mode];
  $('rec-btn-label').textContent = t(label);
  btn.disabled = st.mode === 'saving';
  btn.hidden = st.mode === 'review';
  btn.dataset.kind = st.mode === 'recording' || (dev && st.mode === 'saving') ? 'stop' : dev ? 'rec' : st.mode === 'off' ? 'mic' : 'rec';
  btn.setAttribute('aria-pressed', String(st.mode === 'recording'));

  // review + extras
  $('rec-review').hidden = st.mode !== 'review';
  $('wave').hidden = st.mode !== 'review';
  $('rec-hint').hidden = amb;
  $('rec-undo').hidden = !(st.lastSaved.length && st.mode === 'ready');
  const retry = $('rec-retry');
  retry.hidden = !st.failed.length;
  retry.textContent = `${t('rec.retry')} (${st.failed.length})`;
  $('rec-mic-field').hidden = dev;
  $('rec-mic').disabled = dev || st.mode === 'recording' || st.mode === 'review' || st.mode === 'saving';
  $('rec-source').disabled = st.mode === 'recording' || st.mode === 'review' || st.mode === 'saving';
  $('rec-session').disabled = (st.mode !== 'off' && st.mode !== 'ready') || st.cont.pending.length > 0; // queued chunks belong to this session

  renderRun();
  renderList();
}

/** Status panel of a continuous run: elapsed, uploaded chunks / MB, last upload, pending retries, microphone. */
function renderRun() {
  const c = st.cont;
  const box = $('rec-run');
  const show = isContinuous() && (st.mode === 'recording' || c.pending.length > 0 || c.sent > 0);
  box.hidden = !show;
  if (!show) return;
  const elapsed = (st.mode === 'recording' ? Date.now() - st.t0 : c.elapsedMs) / 1000;
  $('run-elapsed').textContent = `${fmtTime(elapsed)} · ${t('rec.run_recorded', { time: fmtTime(st.n / SR) })}`;
  $('run-sent').textContent = t('rec.run_sent_val', { n: c.sent, mb: (c.bytes / 1048576).toFixed(1) });
  $('run-last').textContent = c.lastOk ? new Date(c.lastOk).toLocaleTimeString() : '—';
  $('run-pending').textContent = String(c.pending.length)
    + (st.failed.length ? t('rec.run_failed', { n: st.failed.length }) : '')
    + (c.dropped ? t('rec.run_lost', { n: c.dropped }) : '')
    + (c.err && (c.pending.length || c.full) ? ` · ${c.err}` : '');
  $('run-mic').textContent = (st.lost ? t('rec.run_mic_lost') : t('rec.run_mic_ok')) + (c.reacq ? t('rec.run_mic_reacq', { n: c.reacq }) : '');
  box.dataset.alert = String(st.lost || c.pending.length > 1 || c.dropped > 0 || c.full);
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
  if (isDevSource()) {
    if (st.mode === 'recording') stopDeviceTake();
    else if (st.mode === 'ready' || st.mode === 'off') startDeviceTake();
    return;
  }
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

$('rec-source').addEventListener('change', (e) => {
  st.dev.source = e.target.value;
  store('gaja.rec.source', st.dev.source);
  if (isDevSource()) {
    if (st.stream) closeMic(); // the browser mic is not needed any more
    clearInterval(st.dev.poll);
    st.dev.active = null;
    st.dev.lastTake = null;
    st.mode = 'ready';
    setStatus(isAmbient() ? t('rec.dev_ambient_note', { s: AMBIENT_CHUNK_S }) : '');
  } else {
    st.mode = st.stream ? 'ready' : 'off';
    setStatus('');
  }
  $('rec-devstat').hidden = true;
  render();
});

$('rec-session').addEventListener('change', async (e) => {
  st.session = e.target.value === '__new__' ? newSessionId() : e.target.value;
  store('gaja.rec.session', st.session);
  st.lastSaved = [];
  st.progress = {};
  st.dev.lastTake = null;
  $('rec-devstat').hidden = true;
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
  if (e.code === 'Space') { e.preventDefault(); if (!(st.mode === 'recording' && isContinuous())) primaryAction(); } // a stray key must not end a days-long run
  else if (e.key === 'Enter' && st.mode === 'review') { e.preventDefault(); saveTake(); }
  else if ((e.key === 'Backspace' || e.key === 'r' || e.key === 'R') && st.mode === 'review') { e.preventDefault(); discardTake(); }
});
// Space activates a focused button on keyup: the keydown handler above already did the work
document.addEventListener('keyup', (e) => { if (e.code === 'Space' && !/^(select|input|textarea)$/i.test(e.target.tagName || '')) e.preventDefault(); });

window.addEventListener('beforeunload', (e) => {
  if (st.mode === 'recording' || st.mode === 'review' || st.mode === 'saving' || st.failed.length || st.cont.pending.length) { e.preventDefault(); e.returnValue = t('rec.leave'); }
});
navigator.mediaDevices && navigator.mediaDevices.addEventListener && navigator.mediaDevices.addEventListener('devicechange', () => {
  renderMics();
  if (st.lost) scheduleReacquire(500); // the unplugged device may just have come back
});

// The browser drops the screen wake lock whenever the tab is hidden: take it again when the tab is shown.
document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'visible' && st.mode === 'recording') holdWakeLock(); });
window.addEventListener('online', () => { if (st.cont.wake) st.cont.wake(); });

// Once a second (throttled, not stopped, in a background tab): the run panel and the capture watchdog.
setInterval(() => {
  if (st.mode === 'recording' && isContinuous() && st.stream && !st.lost) {
    const track = st.stream.getAudioTracks()[0];
    if (!track || track.readyState === 'ended' || performance.now() - st.lastFrameAt > STALL_S * 1000) micLost();
  }
  renderRun();
}, 1000);

// ---------------------------------------------------------------------------------------------
// start

async function init() {
  st.session = store('gaja.rec.session') || newSessionId();
  store('gaja.rec.session', st.session);
  const saved = ALL_PROMPTS.find((p) => p.id === store('gaja.rec.prompt'));
  if (saved) st.cur = saved;
  st.dev.source = store('gaja.rec.source') || '';
  render();
  renderSessions();
  renderMics();
  refreshDevices();
  setInterval(refreshDevices, 15000); // devices come and go; the selector follows
  await Promise.all([refreshProgress(), refreshSessions()]);
  if (!saved || promptDone(st.cur)) {
    const first = ALL_PROMPTS.find((p) => !promptDone(p));
    if (first) selectPrompt(first);
  }
  if (ALL_PROMPTS.every(promptDone)) setStatus(t('rec.all_done'));
}
init();
