// AudioWorkletProcessor for microphone capture.
//
//   input (any channel count, AudioContext rate)
//     -> mono downmix
//     -> 2nd-order low-pass at 7 kHz (anti-aliasing before decimation)
//     -> continuous linear resampling to 16 kHz
//     -> PCM16 LE frames of 40 ms (640 samples = 1280 bytes)
//
// Frames are transferred to the main thread, which forwards them to the WebSocket.

class MicCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.target = 16000;
    this.chunk = 640; // 40 ms
    this.out = new Int16Array(this.chunk);
    this.outIdx = 0;

    this.ratio = sampleRate / this.target;
    this.pos = 0.0; // read position relative to the first sample of the current block
    this.prev = 0.0; // last (filtered) sample of the previous block = index -1

    // RBJ biquad low-pass, Butterworth Q
    this.useFilter = sampleRate > this.target * 1.25;
    if (this.useFilter) {
      const fc = Math.min(7000, sampleRate * 0.45);
      const w0 = (2 * Math.PI * fc) / sampleRate;
      const alpha = Math.sin(w0) / (2 * Math.SQRT1_2);
      const cos = Math.cos(w0);
      const a0 = 1 + alpha;
      this.b0 = (1 - cos) / 2 / a0;
      this.b1 = (1 - cos) / a0;
      this.b2 = this.b0;
      this.a1 = (-2 * cos) / a0;
      this.a2 = (1 - alpha) / a0;
      this.x1 = this.x2 = this.y1 = this.y2 = 0;
    }

    this.port.onmessage = (event) => {
      if (event.data && event.data.type === 'flush') this.flush();
    };
  }

  flush() {
    if (this.outIdx > 0) {
      const slice = this.out.slice(0, this.outIdx);
      this.port.postMessage({ type: 'audio', pcm16: slice.buffer }, [slice.buffer]);
      this.outIdx = 0;
    }
    this.port.postMessage({ type: 'flushed' });
  }

  emit(value) {
    const c = value < -1 ? -1 : value > 1 ? 1 : value;
    this.out[this.outIdx++] = c < 0 ? Math.round(c * 0x8000) : Math.round(c * 0x7fff);
    if (this.outIdx >= this.chunk) {
      const copy = new Int16Array(this.out);
      this.port.postMessage({ type: 'audio', pcm16: copy.buffer }, [copy.buffer]);
      this.outIdx = 0;
    }
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0 || input[0].length === 0) return true;
    const n = input[0].length;
    const channels = input.length;

    const mono = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      let sum = 0;
      for (let c = 0; c < channels; c++) sum += input[c][i];
      mono[i] = sum / channels;
    }

    if (this.useFilter) {
      for (let i = 0; i < n; i++) {
        const x = mono[i];
        const y = this.b0 * x + this.b1 * this.x1 + this.b2 * this.x2 - this.a1 * this.y1 - this.a2 * this.y2;
        this.x2 = this.x1; this.x1 = x;
        this.y2 = this.y1; this.y1 = y;
        mono[i] = y;
      }
    }

    // positions in [-1, n-1): index -1 is the last sample of the previous block
    while (this.pos < n - 1) {
      let s;
      if (this.pos < 0) {
        s = this.prev + (this.pos + 1) * (mono[0] - this.prev);
      } else {
        const i0 = Math.floor(this.pos);
        s = mono[i0] + (this.pos - i0) * (mono[i0 + 1] - mono[i0]);
      }
      this.emit(s);
      this.pos += this.ratio;
    }
    this.pos -= n;
    this.prev = mono[n - 1];
    return true;
  }
}

registerProcessor('mic-capture-processor', MicCaptureProcessor);
