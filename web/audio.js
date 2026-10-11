// AudioController
//
// - The AudioContext is created and resumed *synchronously* inside the first user gesture of any kind
//   (see `unlockOnGesture`) - iOS Safari only honours resume() when it is called from the gesture's
//   own call stack, not after an `await`. It does not depend on the microphone, so typed-text
//   replies are audible even if the mic was never started.
// - Frames / utterance markers that arrive while the worklet graph is still initialising are queued
//   and replayed in order, never dropped.
// - Playback goes worklet -> AnalyserNode -> speakers, so the orb can follow the real output level.
//   The worklet resamples whatever `audio_start.sample_rate` says (Piper 22.05 k, Paradee 24 k,
//   Supertonic 44.1 k, ...) to the context rate, per utterance.
// - Microphone: echoCancellation / noiseSuppression / autoGainControl, 16 kHz PCM16 frames of
//   ~40 ms (resampling happens inside capture-processor.js). A second AnalyserNode on the mic
//   feeds the visual level.

export class AudioController {
  constructor(options = {}) {
    this.onAudioChunk = options.onAudioChunk || (() => {});
    this.onUnderrun = options.onUnderrun || (() => {});
    this.onPlaybackStats = options.onPlaybackStats || (() => {});
    this.onUtterancePlay = options.onUtterancePlay || (() => {});
    this.onUtteranceDone = options.onUtteranceDone || (() => {});
    this.onPlaying = options.onPlaying || (() => {});
    this.onContextState = options.onContextState || (() => {});
    this.onMicEnded = options.onMicEnded || (() => {});

    this.ctx = null;
    this.initPromise = null;
    this.pending = []; // markers/frames posted before the playback worklet exists
    this.playbackNode = null;
    this.outAnalyser = null;
    this.outData = null;

    this.micStream = null;
    this.micSource = null;
    this.micAnalyser = null;
    this.micData = null;
    this.captureNode = null;
    this.silentNode = null;
    this.micSettings = null;
    this.isRecording = false;
    this.isMuted = false;
    this.muteRequested = false;

    this.sampleRate = 24000;
  }

  get contextState() {
    return this.ctx ? this.ctx.state : 'uninitialized';
  }

  get contextRate() {
    return this.ctx ? this.ctx.sampleRate : 0;
  }

  /** Synchronous: safe to call from inside a gesture handler. */
  ensureContext() {
    if (this.ctx) return this.ctx;
    const Ctx = window.AudioContext || window.webkitAudioContext;
    this.ctx = new Ctx({ latencyHint: 'interactive' });
    this.ctx.addEventListener('statechange', () => this.onContextState(this.ctx.state));
    this.setAudioSession('playback');
    return this.ctx;
  }

  /** iOS 16.4+/Safari: make Web Audio play even with the ring/silent switch on, and prefer the speaker. */
  setAudioSession(type) {
    try {
      if (navigator.audioSession && navigator.audioSession.type !== type) navigator.audioSession.type = type;
    } catch (_) { /* not supported */ }
  }

  async init() {
    if (this.initPromise) return this.initPromise;
    this.ensureContext();
    this.initPromise = (async () => {
      await Promise.all([
        this.ctx.audioWorklet.addModule('capture-processor.js'),
        this.ctx.audioWorklet.addModule('playback-processor.js'),
      ]);

      this.playbackNode = new AudioWorkletNode(this.ctx, 'playback-processor', {
        numberOfInputs: 0,
        outputChannelCount: [1],
      });
      this.playbackNode.port.onmessage = (event) => this.onPlaybackMessage(event.data);

      this.outAnalyser = this.ctx.createAnalyser();
      this.outAnalyser.fftSize = 1024;
      this.outAnalyser.smoothingTimeConstant = 0.4;
      this.outData = new Float32Array(this.outAnalyser.fftSize);
      this.playbackNode.connect(this.outAnalyser);
      this.outAnalyser.connect(this.ctx.destination);

      this.playbackNode.port.postMessage({ type: 'config', sampleRate: this.sampleRate });

      // replay everything that arrived during init, in order
      const queued = this.pending;
      this.pending = [];
      for (const m of queued) this.post(m.msg, m.transfer);
    })();
    this.initPromise.catch(() => { this.initPromise = null; });
    return this.initPromise;
  }

  onPlaybackMessage(d) {
    if (!d) return;
    switch (d.type) {
      case 'underrun':
        this.onUnderrun(d.count);
        break;
      case 'stats':
        this.onPlaybackStats(d);
        break;
      case 'utterance_play':
        this.onUtterancePlay(d.id);
        break;
      case 'utterance_done':
        this.onUtteranceDone(d.id);
        break;
      case 'playing':
        this.onPlaying(d.value);
        break;
    }
  }

  async resume() {
    if (this.ctx && this.ctx.state !== 'running') {
      try {
        await this.ctx.resume();
      } catch (err) {
        console.debug('AudioContext resume failed', err);
      }
    }
  }

  async ensureReady() {
    this.unlock();
    await this.init();
    await this.resume();
  }

  /** Call from inside a user gesture: create + resume the context right now, play one silent sample (iOS). */
  unlock() {
    const ctx = this.ensureContext();
    if (ctx.state !== 'running') {
      try {
        const p = ctx.resume();
        if (p && p.catch) p.catch(() => {});
        const buf = ctx.createBuffer(1, 1, ctx.sampleRate);
        const src = ctx.createBufferSource();
        src.buffer = buf;
        src.connect(ctx.destination);
        src.start(0);
      } catch (err) {
        console.debug('audio unlock failed', err);
      }
    }
    this.init().catch((e) => console.warn('audio init failed', e));
  }

  /**
   * Browsers only let a page start audio from inside a user gesture. Listen for gestures of any kind
   * and (re)unlock whenever the context is not running - also after iOS 'interrupted' (a call, Siri).
   */
  unlockOnGesture(target = window) {
    const events = ['pointerdown', 'touchstart', 'pointerup', 'touchend', 'click', 'keydown'];
    const handler = () => { if (!this.ctx || this.ctx.state !== 'running') this.unlock(); };
    events.forEach((e) => target.addEventListener(e, handler, { capture: true, passive: true }));
    document.addEventListener('visibilitychange', () => { if (!document.hidden) this.resume(); });
    return handler;
  }

  post(msg, transfer) {
    if (!this.playbackNode) {
      this.pending.push({ msg, transfer });
      // frames may arrive before any gesture: start building the graph (it stays suspended
      // until the first gesture resumes it)
      this.init().catch((e) => console.warn('audio init failed', e));
      return;
    }
    this.playbackNode.port.postMessage(msg, transfer || []);
  }

  setPlaybackSampleRate(rate) {
    if (!rate) return;
    this.sampleRate = rate;
    this.post({ type: 'config', sampleRate: rate });
  }

  startUtterance(id, rate) {
    if (rate) this.sampleRate = rate;
    this.post({ type: 'start_utterance', id, sampleRate: this.sampleRate });
  }

  enqueue(pcm16, id, rate) {
    let buf = pcm16;
    if (buf.byteLength % 2) buf = buf.slice(0, buf.byteLength - 1); // never let a stray byte throw in the worklet
    this.post({ type: 'audio', pcm16: buf, utteranceId: id, sampleRate: rate || this.sampleRate }, [buf]);
  }

  endUtterance(id) {
    this.post({ type: 'audio_end', id });
  }

  /** Barge-in: drop everything that is queued or sounding, immediately. */
  flush() {
    this.pending = this.pending.filter((m) => m.msg.type === 'config');
    if (this.playbackNode) this.playbackNode.port.postMessage({ type: 'flush' });
  }

  // ---- levels (0..1), read from the render loop -------------------------------------------

  static rms(analyser, buf) {
    analyser.getFloatTimeDomainData(buf);
    let sum = 0;
    for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
    return Math.sqrt(sum / buf.length);
  }

  /** raw RMS (0..1) of what the speakers are playing */
  getOutputRms() {
    if (!this.outAnalyser) return 0;
    return AudioController.rms(this.outAnalyser, this.outData);
  }

  /** perceptual 0..1 for visuals */
  getOutputLevel() {
    return Math.min(1, this.getOutputRms() * 3.2);
  }

  /** raw input RMS, also while push-to-talk is gated (for the "no signal" watchdog) */
  getMicRms() {
    if (!this.micAnalyser || !this.isRecording) return 0;
    return AudioController.rms(this.micAnalyser, this.micData);
  }

  getMicLevel() {
    if (!this.micAnalyser || !this.isRecording || this.isMuted) return 0;
    return Math.min(1, AudioController.rms(this.micAnalyser, this.micData) * 6);
  }

  // ---- microphone --------------------------------------------------------------------------

  /** Audio inputs the browser exposes; labels are empty until mic permission has been granted. */
  static async listMicrophones() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return [];
    try {
      const all = await navigator.mediaDevices.enumerateDevices();
      return all.filter((d) => d.kind === 'audioinput');
    } catch (_) {
      return [];
    }
  }

  /**
   * Opens the chosen device, or the system default when deviceId is empty. A device that is gone
   * (or constraints the hardware rejects) falls back to the default, then to a bare {audio:true}.
   * Sets this.micFellBack when the requested device could not be used.
   * opts.raw = true turns echo cancellation / noise suppression / AGC OFF (the wake-word recorder
   * wants the untreated microphone signal).
   */
  async openMicStream(deviceId, opts = {}) {
    const on = !opts.raw;
    const processing = { echoCancellation: on, noiseSuppression: on, autoGainControl: on, channelCount: 1 };
    const attempts = [];
    if (deviceId) attempts.push({ audio: { ...processing, deviceId: { exact: deviceId } }, video: false });
    attempts.push({ audio: processing, video: false });
    attempts.push({ audio: opts.raw ? { echoCancellation: false, noiseSuppression: false, autoGainControl: false } : true, video: false });
    let lastErr = null;
    for (let i = 0; i < attempts.length; i++) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia(attempts[i]);
        this.micFellBack = !!deviceId && i > 0;
        return stream;
      } catch (err) {
        lastErr = err;
        // only a missing device / impossible constraints are worth retrying; a denial is final
        if (!err || !['NotFoundError', 'OverconstrainedError', 'NotReadableError', 'AbortError'].includes(err.name)) throw err;
      }
    }
    throw lastErr;
  }

  /** Inputs for a device picker: Chrome's 'default' / 'communications' pseudo-devices are dropped (the caller adds its own "system default"). */
  static async microphoneOptions() {
    const mics = await AudioController.listMicrophones();
    const real = mics.filter((d) => d.deviceId && d.deviceId !== 'default' && d.deviceId !== 'communications');
    return { all: mics, real: real.map((d, i) => ({ deviceId: d.deviceId, label: d.label, index: i + 1 })) };
  }

  async startMicrophone(deviceId = '') {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      const err = new Error('insecure-context');
      err.name = 'InsecureContext';
      throw err;
    }
    this.unlock();
    // getUserMedia first (it shows the permission prompt inside the gesture), the graph after
    const stream = await this.openMicStream(deviceId);
    if (this.micStream) { stream.getTracks().forEach((t) => t.stop()); return; }
    this.micStream = stream;
    try {
      await this.init();
      await this.resume();
      this.setAudioSession('play-and-record');
      const track = stream.getAudioTracks()[0];
      this.micSettings = track && track.getSettings ? track.getSettings() : null;
      this.micLabel = track ? track.label : '';
      if (track) track.addEventListener('ended', () => { if (this.micStream === stream) this.onMicEnded(); });

      this.micSource = this.ctx.createMediaStreamSource(stream);

      this.micAnalyser = this.ctx.createAnalyser();
      this.micAnalyser.fftSize = 512;
      this.micData = new Float32Array(this.micAnalyser.fftSize);
      this.micSource.connect(this.micAnalyser);

      this.captureNode = new AudioWorkletNode(this.ctx, 'mic-capture-processor', {
        numberOfOutputs: 1,
        outputChannelCount: [1],
      });
      this.captureNode.port.onmessage = (event) => {
        const d = event.data;
        if (!d || !this.isRecording) return;
        if (d.type === 'audio' && !this.isMuted) this.onAudioChunk(d.pcm16);
        // the worklet has emitted its partial frame: now it is safe to close the gate
        else if (d.type === 'flushed' && this.muteRequested) this.isMuted = true;
      };
      this.micSource.connect(this.captureNode);
      // keep the capture node pulled by the graph without making any sound
      const silent = this.ctx.createGain();
      silent.gain.value = 0;
      this.captureNode.connect(silent);
      silent.connect(this.ctx.destination);
      this.silentNode = silent;

      this.isRecording = true;
    } catch (err) {
      this.stopMicrophone();
      throw err;
    }
  }

  stopMicrophone() {
    this.isRecording = false;
    for (const key of ['captureNode', 'micSource', 'micAnalyser', 'silentNode']) {
      if (this[key]) {
        try { this[key].disconnect(); } catch (_) { /* already gone */ }
        this[key] = null;
      }
    }
    if (this.micStream) {
      this.micStream.getTracks().forEach((t) => t.stop());
      this.micStream = null;
    }
    this.micSettings = null;
    this.micLabel = '';
    this.setAudioSession('playback');
  }

  /** Push-to-talk gate. Muting first flushes the partial frame so the tail of the phrase is sent. */
  setMuted(muted) {
    this.muteRequested = muted;
    if (!muted) {
      this.isMuted = false;
    } else if (this.captureNode) {
      this.captureNode.port.postMessage({ type: 'flush' });
    } else {
      this.isMuted = true;
    }
  }
}
