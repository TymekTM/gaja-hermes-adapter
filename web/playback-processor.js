// AudioWorkletProcessor for playback.
//
// - Receives PCM16 chunks at the sample rate declared by `audio_start` and converts them to the
//   AudioContext rate with *continuous* linear resampling (the fractional phase and the last input
//   sample carry over between chunks, so chunk borders never click).
// - One growable ring buffer (starts at 60 s, doubles on demand, never drops audio). TTS runs ~15x
//   faster than real time, so a long answer arrives as a burst that must be held.
// - ~80 ms pre-buffer whenever playback starts from silence; underruns are counted when a stream
//   that is still being delivered runs dry.
// - Utterance segments: every `start_utterance` remembers where it begins in the stream, so the
//   processor can tell the page *when a sentence actually starts sounding* (`utterance_play`) and
//   when its last sample was played (`utterance_done`). Captions and the orb follow the speaker,
//   not the network.
// - `flush` empties everything instantly (barge-in).

class PlaybackProcessor extends AudioWorkletProcessor {
  constructor() {
    super();

    this.capacity = Math.round(sampleRate * 60);
    this.ring = new Float32Array(this.capacity);
    this.readIdx = 0;
    this.writeIdx = 0;
    this.available = 0;

    // absolute counters (samples at context rate) used to place utterance boundaries
    this.writeAbs = 0;
    this.readAbs = 0;
    this.segments = []; // { id, startAbs, endAbs|null, started }

    // resampler state (reset at each utterance)
    this.sourceRate = sampleRate;
    this.phase = 0.0;
    this.lastSample = 0.0;
    this.hasLast = false;

    this.prebufferSamples = Math.round(sampleRate * 0.08);
    this.prebuffering = true;
    this.streaming = false; // an utterance is still being delivered
    this.currentId = null;

    this.underruns = 0;
    this.samplesPlayed = 0;
    this.blocks = 0;
    this.isPlaying = false;

    this.port.onmessage = (event) => {
      const d = event.data;
      if (!d) return;
      switch (d.type) {
        case 'config':
          if (d.sampleRate > 0) this.sourceRate = d.sampleRate;
          break;
        case 'start_utterance':
          this.startUtterance(d.id, d.sampleRate);
          break;
        case 'audio':
          this.enqueue(d.pcm16, d.sampleRate);
          break;
        case 'audio_end':
          this.endUtterance(d.id);
          break;
        case 'flush':
          this.flush();
          break;
        case 'get_stats':
          this.reportStats();
          break;
      }
    };
  }

  startUtterance(id, rate) {
    if (rate > 0) this.sourceRate = rate;
    this.segments.push({ id, startAbs: this.writeAbs, endAbs: null, started: false });
    this.currentId = id;
    this.streaming = true;
    // only wait for the pre-buffer when nothing is already sounding
    if (this.available === 0) this.prebuffering = true;
    this.phase = 0.0;
    this.hasLast = false;
  }

  endUtterance(id) {
    const seg = this.segments.find((s) => s.id === id);
    if (!seg) return; // flushed / unknown: ignore
    seg.endAbs = this.writeAbs;
    if (this.currentId === id) {
      this.streaming = false;
      this.prebuffering = false; // short sentences must not wait for a pre-buffer that never fills
    }
  }

  flush() {
    this.readIdx = 0;
    this.writeIdx = 0;
    this.available = 0;
    this.readAbs = this.writeAbs;
    this.segments = [];
    this.phase = 0.0;
    this.lastSample = 0.0;
    this.hasLast = false;
    this.prebuffering = true;
    this.streaming = false;
    this.currentId = null;
    this.setPlaying(false);
    this.port.postMessage({ type: 'flushed' });
    this.reportStats();
  }

  grow(extra) {
    if (this.available + extra <= this.capacity) return;
    let cap = this.capacity * 2;
    while (this.available + extra > cap) cap *= 2;
    const next = new Float32Array(cap);
    for (let i = 0; i < this.available; i++) {
      next[i] = this.ring[(this.readIdx + i) % this.capacity];
    }
    this.ring = next;
    this.readIdx = 0;
    this.writeIdx = this.available;
    this.capacity = cap;
  }

  push(sample) {
    this.ring[this.writeIdx] = sample;
    this.writeIdx = (this.writeIdx + 1) % this.capacity;
    this.available++;
    this.writeAbs++;
  }

  enqueue(buffer, chunkRate) {
    const rate = chunkRate || this.sourceRate;
    const pcm = new Int16Array(buffer);
    const n = pcm.length;
    if (n === 0) return;

    if (rate === sampleRate) {
      this.grow(n);
      for (let i = 0; i < n; i++) this.push(pcm[i] / 32768);
      return;
    }

    // Continuous linear resampling. Input index -1 is the last sample of the previous chunk.
    const step = rate / sampleRate;
    this.grow(Math.ceil((n + 2) / step) + 32);
    if (!this.hasLast) {
      this.lastSample = pcm[0] / 32768;
      this.hasLast = true;
      this.phase = 1.0; // start exactly on sample 0
    }
    let pos = -1.0 + this.phase;
    while (pos < n - 1) {
      let s;
      if (pos < 0) {
        s = this.lastSample + (pos + 1) * (pcm[0] / 32768 - this.lastSample);
      } else {
        const i0 = Math.floor(pos);
        const a = pcm[i0] / 32768;
        const b = pcm[i0 + 1] / 32768;
        s = a + (pos - i0) * (b - a);
      }
      this.push(s);
      pos += step;
    }
    this.lastSample = pcm[n - 1] / 32768;
    this.phase = pos - (n - 1);
  }

  setPlaying(value) {
    if (value !== this.isPlaying) {
      this.isPlaying = value;
      this.port.postMessage({ type: 'playing', value });
      if (!value) this.reportStats();
    }
  }

  reportStats() {
    this.port.postMessage({
      type: 'stats',
      samplesPlayed: this.samplesPlayed,
      underruns: this.underruns,
      buffered: this.available,
    });
  }

  // announce sentence starts / ends that the read head has reached
  advanceSegments() {
    for (;;) {
      const seg = this.segments[0];
      if (!seg) return;
      if (!seg.started) {
        if (this.readAbs >= seg.startAbs && (this.available > 0 || seg.endAbs !== null)) {
          seg.started = true;
          this.port.postMessage({ type: 'utterance_play', id: seg.id });
        } else {
          return;
        }
      }
      if (seg.endAbs !== null && this.readAbs >= seg.endAbs) {
        this.segments.shift();
        this.port.postMessage({ type: 'utterance_done', id: seg.id });
        continue;
      }
      return;
    }
  }

  process(inputs, outputs) {
    const output = outputs[0];
    if (!output || output.length === 0) return true;
    const ch0 = output[0];
    const len = ch0.length;

    if (this.prebuffering && this.available >= this.prebufferSamples) this.prebuffering = false;

    if (this.prebuffering || this.available === 0) {
      ch0.fill(0);
      for (let c = 1; c < output.length; c++) output[c].fill(0);
      if (this.available === 0) {
        this.advanceSegments(); // zero-length segments
        this.setPlaying(false);
      }
      return true;
    }

    this.advanceSegments();
    let starved = false;
    for (let i = 0; i < len; i++) {
      if (this.available > 0) {
        ch0[i] = this.ring[this.readIdx];
        this.readIdx = (this.readIdx + 1) % this.capacity;
        this.available--;
        this.readAbs++;
        this.samplesPlayed++;
      } else {
        ch0[i] = 0;
        starved = true;
      }
    }
    for (let c = 1; c < output.length; c++) output[c].set(ch0);

    this.advanceSegments();
    this.setPlaying(true);

    if (starved && this.streaming) {
      this.underruns++;
      this.prebuffering = true;
      this.port.postMessage({ type: 'underrun', count: this.underruns });
      this.reportStats();
    } else if (this.available === 0) {
      this.setPlaying(false);
    }

    if (++this.blocks % 20 === 0) this.reportStats();
    return true;
  }
}

registerProcessor('playback-processor', PlaybackProcessor);
