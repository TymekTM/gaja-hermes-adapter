// Gaja web client - main controller.
//
// One orb, one transcript, one big control. This file wires the audio engine, the socket and the DOM;
// the visual state is *derived* (see `visualMode`) from what is really happening: socket status, server
// state, VAD, what the speakers are playing and what the mic is doing.

import { AudioController } from './audio.js';
import { WSClient } from './ws.js';
import { Orb } from './orb.js';
import { Chat } from './chat.js';
import { Tasks } from './tasks.js';
import { SettingsSheet } from './settings.js';
import { t, setUiLang, resolveUiLang, applyStatic, emotionLabel, getUiLang } from './i18n.js';

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);

// ---------------------------------------------------------------------------------------------
// settings

const DEFAULTS = { lang: 'pl', tts: '', mode: 'handsfree', bargein: true, theme: 'system', metrics: false, mic: '' };
function loadSettings() {
  try { return { ...DEFAULTS, ...JSON.parse(localStorage.getItem('gaja.settings') || '{}') }; } catch (_) { return { ...DEFAULTS }; }
}
const settings = loadSettings();
if (params.has('metrics')) settings.metrics = params.get('metrics') !== '0';
if (params.has('lang') && ['pl', 'en', 'auto'].includes(params.get('lang'))) settings.lang = params.get('lang');
if (params.has('theme') && ['system', 'light', 'dark'].includes(params.get('theme'))) settings.theme = params.get('theme');
const saveSettings = () => { try { localStorage.setItem('gaja.settings', JSON.stringify(settings)); } catch (_) { /* private mode */ } };

setUiLang(resolveUiLang(settings.lang));

// ---------------------------------------------------------------------------------------------
// state

const S = {
  conn: 'connecting',     // connecting | open | retrying | closed
  everOpen: false,
  server: 'listening',    // last server `state`
  userSpeaking: false,    // server VAD
  playing: false,         // the worklet is really producing sound
  mic: 'off',             // off | live | armed (stream open, gated; push-to-talk)
  held: false,            // push-to-talk key / button is down
  tempMic: false,         // hold opened a mic that tap-mode would not have
  errorUntil: 0,
  emotion: 'neutral',
  noticeKind: '',
  sttDegraded: false,     // server notice / GET /health: speech recognition runs on its fallback engine
  sttEngine: '',
  health: null,           // { stt, degraded, tools } from GET /health
  micUnavailable: false,
  awaitReady: false,

  // audio protocol bookkeeping
  sounding: false,        // a sentence has started playing (can be seen a beat before `playing` / server `speaking`)
  curId: null,            // utterance whose binary frames we are accepting
  curRate: 24000,         // sample rate announced by that utterance's audio_start
  awaiting: new Set(),    // announced / started, `audio_end` not seen yet
  cancelled: new Set(),   // ids that were flushed: any late frame is stale
  queue: new Map(),       // id -> { text, style }
  gate: false,            // after a barge-in, drop everything until the server catches up
  gateTimer: 0,
  sampleRate: 24000,      // default rate (ready.sample_rate)
  ttsName: '',
  sessionId: '',
  engines: null,          // from `ready` (or /health): [{ id, label, note, langs }]
  healthEngines: null,
  holdUntil: 0,           // echo gate: speakers were audible until this time
  echo: '',
};

const stats = window.__gajaStats = {
  framesReceived: 0,      // binary frames accepted for playback
  bytesReceived: 0,
  samplesPlayed: 0,
  underruns: 0,
  droppedStale: 0,        // binary frames discarded (before audio_start / after audio_end / interrupted)
  droppedMarkers: 0,
  flushes: 0,
  utterancesPlayed: 0,
  state: 'idle',
  conn: 'connecting',
  ctx: 'uninitialized',
  buffered: 0,
  framesSent: 0,          // mic frames sent (real + zeroed)
  framesGated: 0,         // ...of which zeroed by the echo gate
  lastRate: 0,            // sample rate of the latest audio_start
  ctxRate: 0,
};
window.__gaja = { S, settings };

const reduceQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
const lightQuery = window.matchMedia('(prefers-color-scheme: light)');
const isCoarse = window.matchMedia('(hover: none)').matches;
const UA = navigator.userAgent || '';
const isIOS = /iPad|iPhone|iPod/.test(UA) || (/Macintosh/.test(UA) && navigator.maxTouchPoints > 1);
const isAndroid = /Android/.test(UA);

// ---------------------------------------------------------------------------------------------
// modules

const audio = new AudioController({
  onAudioChunk: onMicFrame,
  onUnderrun: (count) => { stats.underruns = count; },
  onPlaybackStats: (d) => {
    stats.samplesPlayed = d.samplesPlayed;
    stats.underruns = d.underruns;
    stats.buffered = d.buffered;
  },
  onUtterancePlay: (id) => showSentence(id),
  onUtteranceDone: (id) => {
    stats.utterancesPlayed++;
    chat.mark(id, 'done');
    S.queue.delete(id);
  },
  onPlaying: (value) => {
    S.playing = value;
    if (!value) { S.sounding = false; scheduleSettle(); }
    refresh();
  },
  onMicEnded: () => { closeMic(); toast(t('toast.mic_failed')); },
});
audio.unlockOnGesture(window);

function effectiveTheme() {
  return settings.theme === 'system' ? (lightQuery.matches ? 'light' : 'dark') : settings.theme;
}

const orb = new Orb($('orb'), {
  reducedMotion: reduceQuery.matches,
  getLevels: () => ({ mic: audio.getMicLevel(), out: audio.getOutputLevel() }),
  onAccent: (css) => document.documentElement.style.setProperty('--accent', css),
});
orb.setTheme(effectiveTheme());
orb.start();

const chat = new Chat({
  listEl: $('log'),
  scrollEl: $('chat-scroll'),
  onChange: () => { document.body.dataset.empty = String(chat.empty); requestAnimationFrame(updateFill); },
});
chat.setShowStyle(settings.metrics);

const tasks = new Tasks({
  mountEl: $('hermes'),
  onActiveChange: (active) => { orb.setHermes(active); refresh(); },
  onCancel: (task) => {
    if (!ws.isOpen) { tasks.cancelRefused(); toast(t('hermes.offline')); return; }
    ws.sendCancelTask(task.id); // the card turns "cancelled" when the server's handoff{cancelled} arrives
  },
});

const ws = new WSClient({ onMessage: onSocketMessage, onStatus: onSocketStatus });
Object.assign(window.__gaja, { audio, chat, tasks, ws, orb }); // test / debug hooks
ws.setHello({ lang: settings.lang, tts: settings.tts || null });

reduceQuery.addEventListener('change', () => orb.setReducedMotion(reduceQuery.matches));

const settingsSheet = new SettingsSheet({
  sheet: $('settings'),
  backdrop: $('backdrop'),
  app: $('app'),
  openBtn: $('btn-settings'),
  closeBtn: $('settings-close'),
  grip: $('settings-grip'),
  onOpen: () => { syncSettingsUI(); paintFoot(); renderVoices(); renderMics(); },
});

// ---------------------------------------------------------------------------------------------
// derived visual state

function interruptable() {
  return S.playing || S.sounding || S.server === 'speaking' || S.curId !== null || S.awaiting.size > 0;
}
/** the big button turns into "stop" whenever Gaja is talking or still thinking */
function canInterrupt() {
  return interruptable() || S.server === 'thinking';
}

function visualMode() {
  if (S.conn !== 'open' && (S.everOpen || S.conn === 'retrying')) return 'offline';
  if (Date.now() < S.errorUntil) return 'error';
  if (S.userSpeaking || (S.held && S.mic !== 'off' && audio.getMicLevel() > 0.12)) return 'user';
  if (S.playing || S.sounding || S.server === 'speaking') return 'speaking';
  if (S.server === 'thinking') return 'thinking';
  if (S.mic === 'live' || (S.held && S.mic !== 'off')) return 'listening';
  return 'idle';
}

const EMO_HUE = { happy: 88, excited: 45, calm: 225, sad: 268, serious: 305, laugh: 68, sigh: 250, fast: 20, slow: 195 };

function statusText(mode) {
  switch (mode) {
    case 'offline': return t('state.offline');
    case 'error': return t('state.error');
    case 'user': return t('state.user');
    case 'speaking': return S.emotion !== 'neutral' ? t('state.speaking_emo', { emo: emotionLabel(S.emotion) }) : t('state.speaking');
    case 'thinking': return tasks.active ? t('state.thinking_hermes') : t('state.thinking');
    case 'listening': return t('state.listening');
    default: return t('state.idle');
  }
}

function hintText(mode) {
  if (mode === 'offline') return t('hint.offline');
  if (S.micUnavailable && S.mic === 'off') return t('hint.no_mic');
  if (canInterrupt() && mode !== 'user') return t('hint.interrupt');
  if (mode === 'user') return t('hint.listening');
  if (S.held) return t('hint.ptt_release');
  if (S.mic === 'live') return settings.bargein ? t('hint.listening_interrupt') : t('hint.listening');
  if (settings.mode === 'ptt') return isCoarse ? t('hint.ptt') : t('hint.ptt_desktop');
  if (tasks.active) return t('hint.hermes_bg');
  return isCoarse ? t('hint.tap_start') : t('hint.tap_start_desktop');
}

function swapText(el, text, state) {
  if (state.current === text) return;
  state.current = text;
  clearTimeout(state.timer);
  if (reduceQuery.matches || !el.textContent) { el.textContent = text; return; }
  el.dataset.swap = 'true';
  state.timer = setTimeout(() => { el.textContent = text; el.dataset.swap = 'false'; }, 130);
}
const hintState = { current: '', timer: 0 };
const statusState = { current: '', timer: 0 };
let lastMode = '';

function refresh() {
  const mode = visualMode();
  if (mode !== lastMode) {
    lastMode = mode;
    document.body.dataset.state = mode;
    orb.setMode(mode);
    stats.state = mode;
  }
  stats.conn = S.conn;
  stats.ctx = audio.contextState;
  stats.ctxRate = audio.contextRate;

  // emotion only tints while Gaja is speaking
  const emo = mode === 'speaking' ? S.emotion : 'neutral';
  orb.setEmotion(emo);
  document.body.dataset.emotion = emo;
  const statusEl = $('status');
  if (emo !== 'neutral') { statusEl.dataset.emo = emo; statusEl.style.setProperty('--emo-h', EMO_HUE[emo] ?? 55); } else { delete statusEl.dataset.emo; }

  const stop = canInterrupt() && mode !== 'offline';
  const mic = $('btn-mic');
  mic.classList.toggle('is-interrupt', stop);
  mic.classList.toggle('is-on', !stop && (S.mic === 'live' || S.held));
  mic.classList.toggle('is-held', S.held);
  mic.setAttribute('aria-pressed', String(!stop && S.mic === 'live'));
  mic.setAttribute('aria-label', stop ? t('mic.interrupt') : S.mic === 'live' ? t('mic.off') : settings.mode === 'ptt' ? t('mic.hold') : t('mic.on'));
  document.body.dataset.interrupt = String(stop);
  document.body.dataset.mode = settings.mode;

  const modeBtn = $('btn-mode');
  modeBtn.setAttribute('aria-label', settings.mode === 'ptt' ? t('ctl.mode_ptt') : t('ctl.mode_hf'));
  modeBtn.title = settings.mode === 'ptt' ? t('mode.ptt_short') : t('mode.hf_short');

  swapText($('status-text'), statusText(mode), statusState);
  swapText($('hint'), hintText(mode), hintState);
}

// ---------------------------------------------------------------------------------------------
// transcript glue

// How much of the phone screen the conversation needs: a short exchange leaves the orb more room, a long one
// gives the text more. Only ever grows (until the chat is cleared), so the orb never wobbles back and forth.
const FILL = ['low', 'mid', 'high'];
function updateFill() {
  const kids = $('log').children;
  const need = kids.length ? kids[kids.length - 1].offsetTop + kids[kids.length - 1].offsetHeight - kids[0].offsetTop : 0;
  const ratio = need / Math.max(1, $('body').clientHeight);
  const level = need === 0 ? 0 : ratio < 0.2 ? 0 : ratio < 0.38 ? 1 : 2;
  const cur = FILL.indexOf(document.body.dataset.fill);
  if (need === 0 || level > cur) document.body.dataset.fill = FILL[level];
}

let settleTimer = 0;
function scheduleSettle() {
  clearTimeout(settleTimer);
  settleTimer = setTimeout(() => {
    if (S.playing || S.awaiting.size || S.curId !== null || S.server !== 'listening') return;
    chat.settle();
    S.emotion = 'neutral';
    refresh();
  }, 1400);
}

function showSentence(id) {
  const u = S.queue.get(id);
  if (!u) return;
  clearTimeout(settleTimer);
  chat.mark(id, 'playing');
  S.sounding = true;
  S.emotion = (u.style && u.style.emotion) || 'neutral';
  refresh();
}

let ghostTimer = 0;
function armGhostTimeout() {
  clearTimeout(ghostTimer);
  ghostTimer = setTimeout(() => chat.dropGhost(), 10000);
}

// ---------------------------------------------------------------------------------------------
// barge-in

function openGate() {
  S.gate = true;
  clearTimeout(S.gateTimer);
  S.gateTimer = setTimeout(() => { S.gate = false; }, 1500);
}
function closeGate() {
  S.gate = false;
  clearTimeout(S.gateTimer);
}

/** Drop everything that is queued or sounding, right now. */
function flushPlayback({ gate = false } = {}) {
  const had = interruptable();
  audio.flush();
  if (S.curId !== null) S.cancelled.add(S.curId);
  S.awaiting.forEach((id) => S.cancelled.add(id));
  S.curId = null;
  S.awaiting.clear();
  S.queue.clear();
  if (S.cancelled.size > 64) S.cancelled = new Set([...S.cancelled].slice(-32));
  S.playing = false;
  S.sounding = false;
  if (gate) openGate();
  if (had) {
    stats.flushes++;
    clearTimeout(settleTimer);
    chat.cutAll();
    S.emotion = 'neutral';
  }
  refresh();
}

function interrupt() {
  if (!canInterrupt()) return;
  flushPlayback({ gate: true });
  chat.cutAll();
  ws.sendInterrupt();
}

// ---------------------------------------------------------------------------------------------
// microphone frames -> server, with the echo gate
//
// Why a client-side gate: with the loudspeaker on, the mic hears Gaja. Browser AEC (we ask for
// echoCancellation + noiseSuppression + autoGainControl) removes most of it, but the residue (~-20..-30 dB)
// is still enough to make the server VAD "hear a user" and cancel Gaja's own sentence. So, in hands-free,
// while Gaja is audible we send *silence* (zero frames - the server's VAD stream stays continuous) unless
// the mic is clearly louder than what we are playing: >= 3 consecutive 40 ms frames above
// max(0.03, 0.22 x output RMS). Then the gate opens (with ~320 ms of pre-roll so the first syllable
// survives) and stays open 700 ms after the last loud frame - the server sees real speech, VAD fires and
// the normal barge-in happens. Headphones make this trivial; with "Przerywanie głosem" off the gate stays
// shut while Gaja talks (half duplex) and only a tap interrupts. Push-to-talk is never gated: pressing
// the button already flushed playback.

const echo = { hot: 0, openUntil: 0, pre: [] };

function frameRms(buf) {
  const a = new Int16Array(buf, 0, buf.byteLength >> 1);
  let sum = 0;
  for (let i = 0; i < a.length; i++) { const v = a[i] / 32768; sum += v * v; }
  return Math.sqrt(sum / Math.max(1, a.length));
}

function sendFrame(buf, gated) {
  stats.framesSent++;
  if (gated) stats.framesGated++;
  ws.sendBinary(buf);
}

function speakerAudible() {
  // `server === 'speaking'` bridges the gaps between sentences (the next one is still being synthesised)
  return S.playing || S.curId !== null || S.awaiting.size > 0 || S.server === 'speaking';
}

function onMicFrame(buf) {
  if (!ws.isOpen) return;
  const now = performance.now();
  if (speakerAudible()) S.holdUntil = now + 250;
  const active = now < S.holdUntil;
  if (S.held || !active) {
    echo.hot = 0;
    echo.pre.length = 0;
    echo.openUntil = 0;
    sendFrame(buf, false);
    return;
  }
  if (!settings.bargein) { sendFrame(new ArrayBuffer(buf.byteLength), true); return; }

  const rms = frameRms(buf);
  const loud = rms > Math.max(0.03, 0.22 * audio.getOutputRms());
  echo.hot = loud ? echo.hot + 1 : 0;
  if (now < echo.openUntil) {
    if (loud) echo.openUntil = now + 700;
    sendFrame(buf, false);
    return;
  }
  echo.pre.push(buf);
  if (echo.pre.length > 8) echo.pre.shift();
  if (echo.hot >= 3) {
    echo.openUntil = now + 700;
    for (const f of echo.pre) sendFrame(f, false);
    echo.pre.length = 0;
    return;
  }
  sendFrame(new ArrayBuffer(buf.byteLength), true);
}

// ---------------------------------------------------------------------------------------------
// socket

function onSocketStatus({ status, attempt, retryInMs }) {
  S.conn = status === 'closed' || status === 'retrying' ? (status === 'closed' && !ws.wantOpen ? 'closed' : 'retrying') : status;
  if (status === 'open') {
    const wasRetry = S.everOpen && connShownOffline;
    S.everOpen = true;
    S.awaitReady = true;
    showConn('open', { recovered: wasRetry });
  } else if (status === 'retrying') {
    S.userSpeaking = false;
    S.server = 'listening';
    flushPlayback();
    chat.dropGhost();
    tasks.failRunning(t('hermes.lost'));
    showConn('retrying', { attempt, retryInMs });
  } else if (status === 'connecting') {
    showConn('connecting', { attempt });
  } else if (status === 'closed') {
    S.userSpeaking = false;
  }
  refresh();
}

let connShownOffline = false;
let connTimer = 0;
let connTick = 0;
function showConn(kind, opts = {}) {
  const el = $('conn');
  const text = $('conn-text');
  const retry = $('conn-retry');
  clearTimeout(connTimer);
  clearInterval(connTick);
  el.dataset.kind = kind;
  if (kind === 'open') {
    connShownOffline = false;
    retry.hidden = true;
    if (opts.recovered) {
      el.hidden = false;
      text.textContent = t('conn.back');
      connTimer = setTimeout(() => { el.hidden = true; }, 1800);
    } else {
      el.hidden = true;
    }
  } else if (kind === 'connecting') {
    retry.hidden = true;
    if (S.everOpen || opts.attempt > 0) {
      el.hidden = false;
      text.textContent = t('conn.connecting');
    } else {
      el.hidden = true;
      connTimer = setTimeout(() => { if (S.conn === 'connecting') { el.hidden = false; text.textContent = t('conn.connecting'); } }, 900);
    }
  } else if (kind === 'retrying') {
    connShownOffline = true;
    el.hidden = false;
    retry.hidden = false;
    const until = Date.now() + (opts.retryInMs || 0);
    const paint = () => {
      const s = Math.max(0, Math.ceil((until - Date.now()) / 1000));
      text.textContent = t('conn.retry_in', { s });
    };
    paint();
    connTick = setInterval(paint, 250);
  }
}
$('conn-retry').addEventListener('click', () => ws.retryNow());

/** `engines` / `voices` in ready or /health: strings or objects -> [{ id, label, note, langs }] */
function parseEngines(list) {
  if (!Array.isArray(list) || !list.length) return null;
  const out = [];
  for (const e of list) {
    if (typeof e === 'string') out.push({ id: e });
    else if (e && (e.id || e.name || e.engine)) {
      out.push({ id: String(e.id || e.name || e.engine), label: e.label || e.title, note: e.note || e.description, langs: Array.isArray(e.langs) ? e.langs : undefined });
    }
  }
  return out.length ? out : null;
}

/** announce (new) hello settings to the running session: the server re-sends `ready`, history is kept */
function applyHello() {
  ws.setHello({ lang: settings.lang, tts: settings.tts || null });
  if (!ws.isOpen) return;
  S.awaitReady = true;
  clearTimeout(S.helloTimer);
  S.helloTimer = setTimeout(() => { if (S.awaitReady) ws.reconnect(); }, 2500);
  ws.sendJson({ type: 'hello', ...ws.hello });
}

function onSocketMessage({ binary, data }) {
  if (binary) return onAudioFrame(data);
  switch (data.type) {
    case 'ready': {
      S.awaitReady = false;
      clearTimeout(S.helloTimer);
      S.sessionId = data.session_id || '';
      S.ttsName = data.tts || '';
      if (data.sample_rate) { S.sampleRate = data.sample_rate; audio.setPlaybackSampleRate(data.sample_rate); }
      const list = parseEngines(data.engines) || parseEngines(data.voices);
      if (list) S.engines = list;
      closeGate();
      renderVoices();
      paintFoot();
      break;
    }

    case 'vad':
      if (data.state === 'speech_start') {
        S.userSpeaking = true;
        if (canInterrupt()) flushPlayback({ gate: true }); // barge-in: the server cancel is on its way
        chat.userListening();
        clearTimeout(ghostTimer);
      } else {
        S.userSpeaking = false;
        chat.userTranscribing();
        armGhostTimeout();
      }
      break;

    case 'transcript':
      clearTimeout(ghostTimer);
      if (data.text && data.text.trim()) {
        if (S.echo && S.echo === data.text.trim()) S.echo = '';
        else chat.addUser(data.text.trim());
      } else {
        chat.dropGhost();
      }
      break;

    case 'assistant_delta': // not sent by the current server (the bubble is built from `utterance`); kept for older ones
      if (!S.gate && data.text) { clearTimeout(ghostTimer); chat.delta(data.text); }
      break;

    case 'state':
      onServerState(data);
      break;

    case 'utterance':
      if (S.gate || S.cancelled.has(data.id)) { stats.droppedMarkers++; break; }
      clearTimeout(ghostTimer);
      S.queue.set(data.id, { text: data.text || '', style: data.style || {} });
      S.awaiting.add(data.id);
      chat.sentence(data.id, data.text || '', data.style || {});
      // a suspended context never reaches the worklet: keep the text readable and say why it is silent
      if (audio.contextState !== 'running') { chat.mark(data.id, 'done'); toast(t('toast.audio_locked'), { kind: 'info' }); }
      break;

    case 'audio_start':
      if (S.gate || S.cancelled.has(data.id)) { stats.droppedMarkers++; S.curId = null; break; }
      S.curId = data.id;
      S.curRate = data.sample_rate || S.sampleRate;
      stats.lastRate = S.curRate;
      S.awaiting.add(data.id);
      audio.startUtterance(data.id, S.curRate);
      break;

    case 'audio_end':
      if (S.curId === data.id) {
        audio.endUtterance(data.id);
        S.curId = null;
      } else {
        stats.droppedMarkers++;
      }
      S.awaiting.delete(data.id);
      break;

    case 'handoff':
      onHandoff(data);
      break;

    case 'tool':
      chat.addTool(data);
      break;

    case 'notice':
      onNotice(data);
      break;

    case 'metrics':
      onMetrics(data);
      break;

    case 'error':
      onServerError(data.message || t('toast.server_error'));
      break;

    default:
      break; // pong, ...
  }
  refresh();
}

function onAudioFrame(buf) {
  if (S.gate || S.curId === null || buf.byteLength < 2) {
    stats.droppedStale++;
    return;
  }
  stats.framesReceived++;
  stats.bytesReceived += buf.byteLength;
  audio.enqueue(buf, S.curId, S.curRate);
}

function onServerState(data) {
  const next = data.state;
  const cutMidDelivery = next === 'listening' && (S.curId !== null || S.awaiting.size > 0);
  if (data.reason === 'barge_in' || cutMidDelivery) flushPlayback();
  // barge-in only stops Gaja's speech: Hermes tasks keep running (their cards end with handoff done/error/cancelled)
  S.server = next;
  if (next === 'thinking') { chat.dropGhost(); clearTimeout(ghostTimer); }
  closeGate();
  if (next === 'listening') scheduleSettle();
}

function onServerError(message) {
  // a cancel that raced with the task finishing: not an error worth a red toast
  if (/no running Hermes task/i.test(message)) {
    tasks.cancelRefused();
    toast(t('hermes.already_done'), { kind: 'info' });
    return;
  }
  // a stale saved voice the server no longer knows: fall back to automatic and say hello again
  if (S.awaitReady && settings.tts) {
    settings.tts = '';
    saveSettings();
    syncSettingsUI();
    renderVoices();
    applyHello();
    return;
  }
  S.errorUntil = Date.now() + 5000;
  toast(message);
  chat.addNote('error', message);
  setTimeout(refresh, 5100);
}

// ---------------------------------------------------------------------------------------------
// Hermes

function onHandoff(evt) {
  const task = tasks.handle(evt);
  if (!task) return; // e.g. `cancelled` for a task this page never saw
  if (evt.state === 'started') chat.addNote('hermes', t('hermes.log_started', { task: task.task }));
  else if (evt.state === 'cancelled') chat.addNote('hermes', t('hermes.log_cancelled', { task: task.task }));
  else if (evt.state === 'done') chat.addNote('hermes', clip(evt.text) || t('hermes.log_done'));
  else if (evt.state === 'error') chat.addNote('error', clip(evt.text) || t('hermes.log_error'));
}
const clip = (s, n = 280) => ((s || '').trim().length > n ? `${s.trim().slice(0, n - 1)}…` : (s || '').trim());

// ---------------------------------------------------------------------------------------------
// server notices ({code: stt_degraded | stt_recovered, message, engine}), sent only when the state changes

function onNotice(evt) {
  const key = `notice.${evt.code}`;
  const message = (evt.code && t(key) !== key ? t(key) : evt.message) || '';
  if (evt.code === 'stt_degraded' || evt.code === 'stt_recovered') setSttDegraded(evt.code === 'stt_degraded', evt.engine);
  if (message) toast(message, { kind: 'info', ms: evt.code === 'stt_degraded' ? 7000 : 4200 });
}

/** a small standing pill in the top bar while speech recognition runs in fallback mode */
function setSttDegraded(on, engine = '') {
  S.sttDegraded = on;
  S.sttEngine = engine || S.sttEngine;
  const pill = $('sttnote');
  pill.hidden = !on;
  $('sttnote-text').textContent = t('notice.stt_degraded_short');
  pill.title = t('notice.stt_degraded');
  paintFoot();
}

// ---------------------------------------------------------------------------------------------
// metrics strip (dev)

const METRICS = [
  ['e2e_first_audio_ms', 'metrics.e2e'], ['stt_ms', 'metrics.stt'], ['router_ms', 'metrics.router'], ['llm_ttft_ms', 'metrics.ttft'],
  ['first_sentence_ms', 'metrics.first'], ['tts_first_ms', 'metrics.tts'],
];
const history = Object.fromEntries(METRICS.map(([k]) => [k, []]));
const p50 = (a) => { const s = [...a].sort((x, y) => x - y); return s[Math.floor((s.length - 1) / 2)]; };

function onMetrics(m) {
  if (m.kind === 'result') return; // announcing a finished Hermes task: no STT/router, not a reply latency
  METRICS.forEach(([k]) => {
    if (typeof m[k] === 'number' && m[k] > 0) { history[k].push(m[k]); if (history[k].length > 30) history[k].shift(); }
  });
  paintMetrics();
}

function paintMetrics() {
  const el = $('metrics');
  el.textContent = '';
  let any = false;
  METRICS.forEach(([k, label]) => {
    const h = history[k];
    if (!h.length) return;
    any = true;
    const last = h[h.length - 1];
    const span = document.createElement('span');
    span.className = `m${k === 'e2e_first_audio_ms' && last > 1500 ? ' bad' : ''}`;
    const name = document.createElement('em');
    name.textContent = t(label);
    const val = document.createElement('b');
    val.textContent = `${Math.round(last)} ms`;
    const med = document.createElement('em');
    med.textContent = `${t('metrics.p50')} ${Math.round(p50(h))}`;
    span.append(name, val, med);
    el.appendChild(span);
  });
  if (!any) {
    const w = document.createElement('span');
    w.className = 'wait';
    w.textContent = t('metrics.wait');
    el.appendChild(w);
  }
}

function applyMetricsVisibility() {
  $('metrics').hidden = !settings.metrics;
  chat.setShowStyle(settings.metrics);
  if (settings.metrics) paintMetrics();
}

// ---------------------------------------------------------------------------------------------
// toast + notices

let toastTimer = 0;
function toast(message, { kind = 'error', ms = 4200 } = {}) {
  const el = $('toast');
  clearTimeout(toastTimer);
  el.hidden = true;
  void el.offsetWidth; // restart the entrance animation
  el.textContent = message;
  el.classList.toggle('info', kind === 'info');
  el.setAttribute('role', kind === 'info' ? 'status' : 'alert'); // info toasts are announced politely
  el.hidden = false;
  toastTimer = setTimeout(() => { el.hidden = true; }, ms);
}

function clearNotice() {
  $('notice').hidden = true;
  $('notice').textContent = '';
  S.noticeKind = '';
  document.body.dataset.notice = '';
  refresh();
}

function showNotice(kind, { title, body, command, foot, actions = [] }) {
  const n = $('notice');
  n.textContent = '';
  const h = document.createElement('h3');
  h.textContent = title;
  const p = document.createElement('p');
  p.textContent = body;
  n.append(h, p);
  if (command) {
    const c = document.createElement('code');
    c.textContent = command;
    n.appendChild(c);
  }
  if (foot) {
    const f = document.createElement('p');
    f.textContent = foot;
    n.appendChild(f);
  }
  const row = document.createElement('div');
  row.className = 'actions';
  actions.forEach(({ label, primary, onClick }) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = `btn small${primary ? ' primary' : ''}`;
    b.textContent = label;
    b.addEventListener('click', onClick);
    row.appendChild(b);
  });
  n.appendChild(row);
  n.hidden = false;
  S.noticeKind = kind;
  document.body.dataset.notice = kind;
  refresh();
}

const typeInstead = (primary = true) => ({ label: t('notice.type_instead'), primary, onClick: () => { clearNotice(); setTypebar(true); } });
const closeAction = { get label() { return t('notice.close'); }, onClick: () => clearNotice() };

function insecureNotice() {
  S.micUnavailable = true;
  showNotice('insecure', {
    title: t('notice.insecure.title'),
    body: t('notice.insecure.body'),
    command: 'tailscale serve --bg --https=8443 http://127.0.0.1:8790',
    foot: t('notice.insecure.foot'),
    actions: [typeInstead(), { label: closeAction.label, onClick: closeAction.onClick }],
  });
}

function deniedNotice() {
  showNotice('denied', {
    title: t('notice.denied.title'),
    body: t(`notice.denied.body.${isIOS ? 'ios' : isAndroid ? 'android' : 'desktop'}`),
    actions: [
      { label: t('notice.retry'), primary: true, onClick: () => { clearNotice(); toggleHandsFree(); } },
      typeInstead(false),
    ],
  });
}

function watchMicPermission() {
  if (!navigator.permissions || !navigator.permissions.query) return;
  navigator.permissions.query({ name: 'microphone' }).then((st) => {
    st.addEventListener('change', () => {
      if (st.state === 'granted' && S.noticeKind === 'denied') { clearNotice(); toast(t('toast.mic_granted'), { kind: 'info' }); }
    });
  }).catch(() => { /* Safari: not supported */ });
}

// ---------------------------------------------------------------------------------------------
// microphone

let micBusy = null;
let wakeLock = null;

async function holdScreenAwake(on) {
  try {
    if (on && 'wakeLock' in navigator && !wakeLock) {
      wakeLock = await navigator.wakeLock.request('screen');
      wakeLock.addEventListener('release', () => { wakeLock = null; });
    } else if (!on && wakeLock) {
      await wakeLock.release();
      wakeLock = null;
    }
  } catch (_) { /* not allowed (battery saver, no gesture) */ }
}

async function openMic({ muted }) {
  if (S.mic !== 'off') return true;
  if (micBusy) return micBusy;
  micBusy = (async () => {
    try {
      if (params.has('insecure')) { const e = new Error('insecure'); e.name = 'InsecureContext'; throw e; }
      if (params.has('micdenied')) { const e = new Error('denied'); e.name = 'NotAllowedError'; throw e; }
      await audio.startMicrophone(settings.mic);
      if (audio.micFellBack) {
        // the remembered device is gone: forget it so the selector shows the system default again
        settings.mic = '';
        saveSettings();
        toast(t('toast.mic_fallback'), { kind: 'info', ms: 3200 });
      }
      audio.setMuted(muted);
      renderMics();
      S.mic = muted ? 'armed' : 'live';
      S.micUnavailable = false;
      clearNotice();
      if (!muted) holdScreenAwake(true);
      return true;
    } catch (err) {
      console.info('microphone unavailable:', err && err.name);
      if (err && err.name === 'InsecureContext') insecureNotice();
      else if (err && (err.name === 'NotAllowedError' || err.name === 'SecurityError')) deniedNotice();
      else if (err && (err.name === 'NotFoundError' || err.name === 'OverconstrainedError')) {
        S.micUnavailable = true;
        showNotice('nomic', { title: t('notice.nomic.title'), body: t('notice.nomic.body'), actions: [typeInstead(), { label: closeAction.label, onClick: closeAction.onClick }] });
      } else if (err && (err.name === 'NotReadableError' || err.name === 'AbortError')) {
        showNotice('busy', { title: t('notice.busy.title'), body: t('notice.busy.body'), actions: [{ label: t('notice.retry'), primary: true, onClick: () => { clearNotice(); toggleHandsFree(); } }, typeInstead(false)] });
      } else toast(t('toast.mic_failed'));
      return false;
    } finally {
      micBusy = null;
      refresh();
    }
  })();
  return micBusy;
}

function closeMic() {
  if (S.mic === 'off') return;
  audio.stopMicrophone();
  S.mic = 'off';
  S.held = false;
  S.tempMic = false;
  holdScreenAwake(false);
  ws.sendEndOfInput();
  refresh();
}

async function toggleHandsFree() {
  if (canInterrupt()) { interrupt(); return; }
  if (settings.mode === 'ptt') {
    toast(isCoarse ? t('hint.ptt') : t('hint.ptt_desktop'), { kind: 'info', ms: 2200 });
    return;
  }
  if (S.mic === 'off') await openMic({ muted: false });
  else closeMic();
}

async function pttStart() {
  if (S.held) return;
  S.held = true;
  if (canInterrupt()) interrupt();
  if (S.mic === 'off') {
    S.tempMic = settings.mode === 'handsfree';
    const ok = await openMic({ muted: false });
    if (!ok) { S.held = false; S.tempMic = false; refresh(); return; }
    if (settings.mode === 'ptt') audio.setMuted(false);
  } else if (S.mic === 'armed') {
    audio.setMuted(false);
  }
  if (!S.held) pttEnd(true); // released while the permission prompt was open
  refresh();
}

function pttEnd(force = false) {
  if (!S.held && !force) return;
  S.held = false;
  if (S.tempMic) {
    closeMic();
  } else if (S.mic === 'armed') {
    audio.setMuted(true);
    ws.sendEndOfInput();
  }
  S.tempMic = false;
  refresh();
}

// The big button.
//  hands-free mode: tap = mic on/off (or interrupt Gaja), press-and-hold (>=320 ms) = talk while held
//  push-to-talk mode: press = talk, release = send
const micBtn = $('btn-mic');
let pressTimer = 0;
let pressHeld = false;
let pressDown = false;
const HOLD_MS = 320;

micBtn.addEventListener('pointerdown', (e) => {
  if (e.button > 0) return;
  pressDown = true;
  pressHeld = false;
  try { micBtn.setPointerCapture(e.pointerId); } catch (_) { /* synthetic */ }
  if (canInterrupt()) return; // a tap interrupts; no hold in that state
  if (settings.mode === 'ptt') { pressHeld = true; pttStart(); return; }
  pressTimer = setTimeout(() => { pressHeld = true; pttStart(); }, HOLD_MS);
});
const endPress = (e) => {
  if (!pressDown) return;
  pressDown = false;
  clearTimeout(pressTimer);
  if (pressHeld) { pressHeld = false; pttEnd(); return; }
  if (e.type === 'pointerup') toggleHandsFree();
};
micBtn.addEventListener('pointerup', endPress);
micBtn.addEventListener('pointercancel', endPress);
micBtn.addEventListener('contextmenu', (e) => e.preventDefault());
micBtn.addEventListener('click', (e) => { if (e.detail === 0) toggleHandsFree(); }); // keyboard Enter

$('orb-hit').style.pointerEvents = 'auto';
$('orb-hit').addEventListener('click', () => { if (canInterrupt()) interrupt(); });

$('btn-mode').addEventListener('click', () => {
  settings.mode = settings.mode === 'ptt' ? 'handsfree' : 'ptt';
  saveSettings();
  applyMode();
  syncSettingsUI();
  toast(settings.mode === 'ptt' ? t('mode.ptt_short') : t('mode.hf_short'), { kind: 'info', ms: 1600 });
  refresh();
});

function applyMode() {
  if (settings.mode === 'ptt' && S.mic === 'live') { audio.setMuted(true); S.mic = 'armed'; holdScreenAwake(false); }
  else if (settings.mode === 'handsfree' && S.mic === 'armed') { audio.setMuted(false); S.mic = 'live'; holdScreenAwake(true); }
}

// ---------------------------------------------------------------------------------------------
// typing

const typebar = $('typebar');
const input = $('text-input');
const kbBtn = $('btn-keyboard');
const sendBtn = $('btn-send');

function setTypebar(open) {
  document.body.dataset.input = open ? 'open' : 'closed';
  kbBtn.setAttribute('aria-pressed', String(open));
  if (open) input.focus({ preventScroll: true });
  else if (document.activeElement === input) input.blur();
}
kbBtn.addEventListener('click', () => setTypebar(document.body.dataset.input !== 'open'));

function sendTyped(text) {
  text = (text || '').trim();
  if (!text) return;
  audio.unlock();                 // we are inside a user gesture: typed turns must be audible without ever touching the mic
  audio.resume();
  if (!ws.isOpen) { toast(t('input.offline')); return; }
  flushPlayback();                // typing over Gaja is a barge-in too
  closeGate();
  S.echo = text;
  chat.addUser(text);
  ws.sendText(text);
  input.value = '';
  sendBtn.disabled = true;
  refresh();
}

typebar.addEventListener('submit', (e) => { e.preventDefault(); sendTyped(input.value); });
input.addEventListener('input', () => { sendBtn.disabled = !input.value.trim(); });
sendBtn.disabled = true;

function renderSuggestions() {
  const box = $('suggest');
  box.textContent = '';
  [1, 2, 3].forEach((n) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = t(`suggest.${n}`);
    b.addEventListener('click', () => sendTyped(b.textContent));
    box.appendChild(b);
  });
}

$('btn-clear').addEventListener('click', () => { chat.clear(); S.emotion = 'neutral'; refresh(); });

// ---------------------------------------------------------------------------------------------
// keyboard

function isTyping(el) { return el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable); }

window.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    if (settingsSheet.close()) return;
    if (document.body.dataset.input === 'open' && document.activeElement === input && !window.matchMedia('(min-width: 900px)').matches) { setTypebar(false); return; }
    if (S.noticeKind) { clearNotice(); return; }
    if (canInterrupt()) interrupt();
    return;
  }
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (settingsSheet.isOpen) return;
  if (e.key === '/' && !isTyping(e.target)) {
    e.preventDefault();
    setTypebar(true);
    return;
  }
  if ((e.code === 'Space' || e.key === ' ') && !isTyping(e.target)) {
    const onButton = e.target.tagName === 'BUTTON' && e.target !== micBtn;
    if (onButton) return;
    e.preventDefault();
    if (!e.repeat) pttStart();
  }
});
window.addEventListener('keyup', (e) => {
  if ((e.code === 'Space' || e.key === ' ') && !isTyping(e.target) && S.held) {
    e.preventDefault();
    pttEnd();
  }
});
window.addEventListener('blur', () => { if (S.held) pttEnd(); });

// ---------------------------------------------------------------------------------------------
// settings

const KNOWN_LANGS = { piper: ['pl'], 'piper-darkman': ['pl'], paradee: ['en'] };

function engineView(e) {
  const nameKey = `voice.${e.id}`;
  const label = e.label || (t(nameKey) !== nameKey ? t(nameKey) : e.id.charAt(0).toUpperCase() + e.id.slice(1));
  const noteKey = `${nameKey}.note`;
  const note = e.note || (t(noteKey) !== noteKey ? t(noteKey) : '');
  return { id: e.id, label, note, langs: e.langs || KNOWN_LANGS[e.id] || [] };
}

let voicesSig = '';
function renderVoices() {
  const known = S.engines || S.healthEngines;
  const basic = !known;
  const list = (known || [{ id: 'piper' }, { id: 'paradee' }]).map(engineView);
  const auto = { id: '', label: t('settings.voice_auto'), note: t('settings.voice_auto_hint'), langs: [] };
  const sig = JSON.stringify([getUiLang(), settings.tts, basic, list]);
  $('voice-note').textContent = basic ? (S.everOpen ? t('settings.voice_basic') : t('settings.voice_loading')) : t('settings.voice_note_lang');
  if (sig === voicesSig) return;
  voicesSig = sig;

  const box = $('voices');
  box.textContent = '';
  [auto, ...list].forEach((v) => {
    const label = document.createElement('label');
    label.className = 'vopt';
    const radio = document.createElement('input');
    radio.type = 'radio';
    radio.name = 'tts';
    radio.value = v.id;
    radio.checked = (settings.tts || '') === v.id;
    radio.addEventListener('change', () => {
      if (!radio.checked) return;
      settings.tts = v.id;
      saveSettings();
      flushPlayback();
      applyHello();
    });
    const vbox = document.createElement('span');
    vbox.className = 'vbox';
    const text = document.createElement('span');
    const b = document.createElement('b');
    b.textContent = v.label;
    text.appendChild(b);
    if (v.note) {
      const sm = document.createElement('small');
      sm.textContent = v.note;
      text.appendChild(sm);
    }
    const langs = document.createElement('span');
    langs.className = v.langs.length ? 'vlang' : '';
    langs.textContent = v.langs.map((l) => l.toUpperCase()).join(' ');
    const mark = document.createElement('span');
    mark.className = 'vmark';
    vbox.append(text, langs, mark);
    label.append(radio, vbox);
    box.appendChild(label);
  });
}

async function fetchHealth() {
  try {
    const res = await fetch(new URL('health', location.href), { cache: 'no-store' });
    if (!res.ok) return;
    const h = await res.json();
    S.healthEngines = parseEngines(h.engines);
    S.health = {
      stt: h.stt_engine ? `${h.stt_engine}${h.stt_device ? `/${h.stt_device}` : ''}` : '',
      degraded: !!h.stt_degraded,
      tools: !!(h.router && h.router.tools),
    };
    if (h.stt_degraded) setSttDegraded(true, h.stt_engine); // the session's own `notice` may arrive later or never
    renderVoices();
    paintFoot();
  } catch (_) { /* the socket's ready event may still deliver the list */ }
}

function paintFoot() {
  const rate = (hz) => `${(hz / 1000).toLocaleString(getUiLang() === 'en' ? 'en-GB' : 'pl-PL')} kHz`;
  const lines = [];
  const head = [S.ttsName && t('settings.voice_label', { v: S.ttsName }), S.sampleRate && S.ttsName && rate(S.sampleRate), S.sessionId && t('settings.session', { id: S.sessionId.slice(0, 8) })].filter(Boolean);
  lines.push(head.length ? head.join(' · ') : t('settings.not_connected'));
  if (audio.ctx) lines.push(`audio: ${audio.contextState} · ${rate(audio.contextRate)} · underruns ${stats.underruns}`);
  if (S.health && S.health.stt) lines.push(`stt: ${S.health.stt}${S.sttDegraded ? ' (fallback)' : ''} · tools ${S.health.tools ? 'on' : 'off'}`);
  const ms = audio.micSettings;
  if (ms) lines.push(`mic: AEC ${ms.echoCancellation ? 'on' : 'off'} · NS ${ms.noiseSuppression ? 'on' : 'off'} · AGC ${ms.autoGainControl ? 'on' : 'off'}`);
  $('settings-foot').textContent = lines.join('\n');
}

function syncSettingsUI() {
  document.querySelectorAll('input[name="lang"]').forEach((r) => { r.checked = r.value === settings.lang; });
  document.querySelectorAll('input[name="mode"]').forEach((r) => { r.checked = r.value === settings.mode; });
  document.querySelectorAll('input[name="theme"]').forEach((r) => { r.checked = r.value === settings.theme; });
  $('set-bargein').checked = !!settings.bargein;
  $('set-metrics').checked = !!settings.metrics;
  $('mode-note').textContent = settings.mode === 'ptt' ? t('settings.mode_ptt_note') : t('settings.mode_hf_note');
}

function applyTheme() {
  const root = document.documentElement;
  if (settings.theme === 'light' || settings.theme === 'dark') root.dataset.theme = settings.theme;
  else delete root.dataset.theme;
  orb.setTheme(effectiveTheme());
  // browser chrome colour
  const colors = { dark: '#14110f', light: '#f6f2ec' };
  document.querySelectorAll('meta[name="theme-color"]').forEach((m) => {
    if (!m.dataset.media) m.dataset.media = m.getAttribute('media') || '';
    if (settings.theme === 'system') { m.setAttribute('media', m.dataset.media); m.content = m.dataset.media.includes('light') ? colors.light : colors.dark; } else { m.removeAttribute('media'); m.content = colors[settings.theme]; }
  });
}
lightQuery.addEventListener('change', () => { if (settings.theme === 'system') orb.setTheme(effectiveTheme()); });

function retranslate() {
  setUiLang(resolveUiLang(settings.lang));
  applyStatic();
  $('log').dataset.emptyText = t('chat.empty');
  renderSuggestions();
  voicesSig = '';
  renderVoices();
  syncSettingsUI();
  paintFoot();
  tasks.retranslate();
  if (settings.metrics) paintMetrics();
  hintState.current = '';
  statusState.current = '';
  refresh();
}

function bindSettings() {
  document.querySelectorAll('input[name="lang"]').forEach((r) => {
    r.addEventListener('change', () => {
      if (!r.checked || settings.lang === r.value) return;
      settings.lang = r.value;
      saveSettings();
      retranslate();
      flushPlayback();
      applyHello();
    });
  });
  document.querySelectorAll('input[name="mode"]').forEach((r) => {
    r.addEventListener('change', () => {
      if (!r.checked) return;
      settings.mode = r.value;
      saveSettings();
      applyMode();
      syncSettingsUI();
      refresh();
    });
  });
  document.querySelectorAll('input[name="theme"]').forEach((r) => {
    r.addEventListener('change', () => { if (r.checked) { settings.theme = r.value; saveSettings(); applyTheme(); } });
  });
  $('set-bargein').addEventListener('change', (e) => { settings.bargein = e.target.checked; saveSettings(); refresh(); });
  $('set-metrics').addEventListener('change', (e) => { settings.metrics = e.target.checked; saveSettings(); applyMetricsVisibility(); });
  $('set-mic').addEventListener('change', (e) => { switchMic(e.target.value); });
  if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
    navigator.mediaDevices.addEventListener('devicechange', () => { renderMics(); });
  }
}

// ---------------------------------------------------------------------------------------------
// microphone selector: '' = whatever the OS calls default; a remembered deviceId otherwise

async function renderMics() {
  const sel = $('set-mic');
  const mics = await AudioController.listMicrophones();
  // Chrome adds pseudo-devices ('default', 'communications') that mirror real ones; the first option covers them
  const real = mics.filter((d) => d.deviceId && d.deviceId !== 'default' && d.deviceId !== 'communications');
  const opts = [{ value: '', label: t('settings.mic_default') }];
  real.forEach((d, i) => opts.push({ value: d.deviceId, label: d.label || t('settings.mic_unnamed', { n: i + 1 }) }));
  const sig = JSON.stringify(opts) + settings.mic;
  if (sel.dataset.sig !== sig) {
    sel.dataset.sig = sig;
    sel.replaceChildren(...opts.map((o) => { const el = document.createElement('option'); el.value = o.value; el.textContent = o.label; return el; }));
  }
  const known = opts.some((o) => o.value === settings.mic);
  sel.value = known ? settings.mic : '';
  const note = $('mic-note');
  if (!mics.length) note.textContent = t('settings.mic_none');
  else if (audio.micLabel) note.textContent = t('settings.mic_using', { label: audio.micLabel });
  else if (!mics.some((d) => d.label)) note.textContent = t('settings.mic_nolabels');
  else note.textContent = '';
}

async function switchMic(deviceId) {
  settings.mic = deviceId;
  saveSettings();
  // give the new device a fresh "no signal" grace period
  micWatch.since = 0; micWatch.warned = false;
  if (S.noticeKind === 'nosignal') clearNotice();
  if (S.mic === 'off') { renderMics(); return; }
  // reopen on the new device, keeping the push-to-talk gate where it was
  const muted = S.mic === 'armed';
  audio.stopMicrophone();
  S.mic = 'off';
  await openMic({ muted });
}

// ---------------------------------------------------------------------------------------------
// live levels -> CSS (the mic ring, the user's equaliser); cheap, runs only while something listens

let lvl = 0;
const rootStyle = document.documentElement.style;
function levelLoop() {
  if (!document.hidden) {
    const target = S.mic !== 'off' ? audio.getMicLevel() : 0;
    lvl += (target - lvl) * 0.35;
    micBtn.style.setProperty('--lvl', lvl.toFixed(3));
    if (chat.ghost) rootStyle.setProperty('--mic-lvl', lvl.toFixed(3));
    stats.ctx = audio.contextState;
  }
  requestAnimationFrame(levelLoop);
}
requestAnimationFrame(levelLoop);

// "no signal" watchdog: an open mic that delivers (near) digital silence usually means the wrong input
// device is selected, or the hardware/OS mute is on. Real rooms are never this quiet (rms ~1e-3 and up).
const SILENT_RMS = 0.0003;
const SILENT_AFTER_MS = 5000;
const micWatch = { since: 0, warned: false };
setInterval(() => {
  if (S.mic === 'off') { micWatch.since = 0; micWatch.warned = false; return; }
  const rms = audio.getMicRms();
  if (rms > SILENT_RMS) {
    micWatch.since = 0;
    if (micWatch.warned) { micWatch.warned = false; if (S.noticeKind === 'nosignal') clearNotice(); }
    return;
  }
  if (!micWatch.since) { micWatch.since = performance.now(); return; }
  if (micWatch.warned || performance.now() - micWatch.since < SILENT_AFTER_MS) return;
  micWatch.warned = true;
  if (S.noticeKind) return; // don't cover a more important notice
  const device = audio.micLabel ? t('notice.nosignal.device', { label: audio.micLabel }) : '';
  showNotice('nosignal', {
    title: t('notice.nosignal.title'),
    body: t('notice.nosignal.body') + device,
    actions: [
      { label: t('notice.nosignal.open'), primary: true, onClick: () => { clearNotice(); settingsSheet.open(); requestAnimationFrame(() => $('set-mic').focus()); } },
      { label: closeAction.label, onClick: closeAction.onClick },
    ],
  });
}, 250);

setInterval(refresh, 400); // picks up time-based changes (error window, mic level threshold)

// ---------------------------------------------------------------------------------------------
// phone plumbing: keyboard-aware height, reconnect when the page comes back

const vv = window.visualViewport;
if (vv) {
  const fit = () => {
    const covered = window.innerHeight - vv.height;
    if (covered > 80) { rootStyle.setProperty('--app-h', `${Math.round(vv.height)}px`); window.scrollTo(0, 0); } else rootStyle.removeProperty('--app-h');
  };
  vv.addEventListener('resize', fit);
  vv.addEventListener('scroll', () => { if (document.activeElement === input) window.scrollTo(0, 0); });
}

function wake() {
  if (!ws.isOpen && ws.wantOpen) ws.retryNow();
}
document.addEventListener('visibilitychange', () => {
  if (document.hidden) return;
  wake();
  audio.resume();
  if (S.mic === 'live') holdScreenAwake(true);
});
window.addEventListener('online', wake);
window.addEventListener('pageshow', (e) => { if (e.persisted) wake(); });

// ---------------------------------------------------------------------------------------------
// go

applyStatic();
$('log').dataset.emptyText = t('chat.empty');
document.body.dataset.empty = String(chat.empty);
applyTheme();
renderSuggestions();
bindSettings();
syncSettingsUI();
applyMetricsVisibility();
renderVoices();
paintFoot();
refresh();
watchMicPermission();
fetchHealth();

if (params.has('insecure') || !(window.isSecureContext && navigator.mediaDevices)) {
  // say it up front instead of failing on the first tap
  insecureNotice();
  document.body.dataset.input = 'open'; // no focus: that would pop the keyboard open on a phone
  kbBtn.setAttribute('aria-pressed', 'true');
}

ws.connect();
