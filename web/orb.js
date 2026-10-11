// The presence: a soft, organic orb drawn with a WebGL fragment shader (Canvas 2D fallback).
//
// It is the state indicator and it reacts to REAL audio: the mic level while the user talks, the
// output level while Gaja speaks. Everything that moves is smoothed here; the app only says
//   orb.setMode('idle'|'listening'|'user'|'thinking'|'speaking'|'offline'|'error')
//   orb.setEmotion('happy'|'calm'|…)
//   orb.setHermes(true|false)      // a satellite detaches and orbits / merges back
// and supplies levels through the `getLevels()` callback once per frame.

// ---- colour -----------------------------------------------------------------------------------

function oklchToSrgb(L, C, hDeg) {
  const h = (hDeg * Math.PI) / 180;
  const a = C * Math.cos(h);
  const b = C * Math.sin(h);
  const l_ = L + 0.3963377774 * a + 0.2158037573 * b;
  const m_ = L - 0.1055613458 * a - 0.0638541728 * b;
  const s_ = L - 0.0894841775 * a - 1.291485548 * b;
  const l = l_ ** 3, m = m_ ** 3, s = s_ ** 3;
  const lin = [
    4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s,
  ];
  return lin.map((v) => {
    const c = Math.min(1, Math.max(0, v));
    return c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055;
  });
}

// [L, C, H] triples for the three layers of the orb (base, body, highlight).
const PALETTE = {
  idle:      [[0.80, 0.095, 58], [0.68, 0.125, 28], [0.89, 0.065, 88]],
  listening: [[0.81, 0.100, 55], [0.69, 0.130, 26], [0.90, 0.070, 90]],
  user:      [[0.85, 0.085, 172], [0.72, 0.100, 192], [0.94, 0.055, 150]],
  thinking:  [[0.74, 0.095, 236], [0.60, 0.110, 252], [0.87, 0.060, 220]],
  speaking:  [[0.85, 0.120, 82], [0.74, 0.140, 48], [0.93, 0.065, 98]],
  offline:   [[0.62, 0.020, 60], [0.52, 0.020, 50], [0.72, 0.015, 70]],
  error:     [[0.72, 0.130, 20], [0.58, 0.150, 12], [0.84, 0.080, 40]],
};

// emotion -> subtle modifiers; never cartoonish
const EMOTION = {
  neutral: { hue: 0, chroma: 1, energy: 1, speed: 1, scale: 1, jitter: 0, light: 0 },
  happy:   { hue: 16, chroma: 1.12, energy: 1.15, speed: 1.2, scale: 1.02, jitter: 0.05, light: 0.02 },
  excited: { hue: 30, chroma: 1.28, energy: 1.45, speed: 1.65, scale: 1.04, jitter: 0.3, light: 0.03 },
  calm:    { hue: -28, chroma: 0.82, energy: 0.65, speed: 0.65, scale: 1.0, jitter: 0, light: 0 },
  sad:     { hue: -75, chroma: 0.55, energy: 0.55, speed: 0.5, scale: 0.93, jitter: 0, light: -0.07 },
  serious: { hue: -14, chroma: 0.5, energy: 0.7, speed: 0.8, scale: 0.97, jitter: 0, light: -0.04 },
  fast:    { hue: 6, chroma: 1.05, energy: 1.1, speed: 1.55, scale: 1, jitter: 0.05, light: 0 },
  slow:    { hue: -12, chroma: 0.9, energy: 0.8, speed: 0.55, scale: 1, jitter: 0, light: 0 },
  laugh:   { hue: 20, chroma: 1.2, energy: 1.35, speed: 1.5, scale: 1.03, jitter: 0.55, light: 0.03 },
  sigh:    { hue: -40, chroma: 0.7, energy: 0.5, speed: 0.45, scale: 0.95, jitter: 0, light: -0.03 },
};

// per mode base values; `in` = how much the mic level drives, `out` = output level
const MODES = {
  idle:      { energy: 0.12, speed: 0.55, scale: 1.0,  swirl: 0, glow: 0.55, sat: 0, in: 0.0, out: 0.0 },
  listening: { energy: 0.16, speed: 0.7,  scale: 1.0,  swirl: 0, glow: 0.65, sat: 0, in: 0.6, out: 0.0 },
  user:      { energy: 0.30, speed: 1.1,  scale: 1.04, swirl: 0, glow: 0.9,  sat: 0, in: 0.95, out: 0.0 },
  thinking:  { energy: 0.14, speed: 0.5,  scale: 0.93, swirl: 1, glow: 0.6,  sat: 0, in: 0.0, out: 0.0 },
  speaking:  { energy: 0.24, speed: 0.9,  scale: 1.0,  swirl: 0, glow: 0.85, sat: 0, in: 0.0, out: 1.0 },
  offline:   { energy: 0.03, speed: 0.06, scale: 0.95, swirl: 0, glow: 0.12, sat: 1, in: 0.0, out: 0.0 },
  error:     { energy: 0.08, speed: 0.3,  scale: 0.97, swirl: 0, glow: 0.4,  sat: 0.2, in: 0.0, out: 0.0 },
};

const damp = (cur, target, lambda, dt) => cur + (target - cur) * (1 - Math.exp(-lambda * dt));
const lerp = (a, b, t) => a + (b - a) * t;

// ---- shaders ----------------------------------------------------------------------------------

const VERT = `attribute vec2 aPos; void main(){ gl_Position = vec4(aPos, 0.0, 1.0); }`;

const FRAG = `
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
uniform vec2 uRes;
uniform float uT, uT2, uEnergy, uLevel, uScale, uSwirl, uGlow, uSat, uJit;
uniform vec3 uC1, uC2, uC3, uSatC;
uniform vec2 uRip[3];
uniform vec4 uSatA;
uniform vec2 uLight;

float hash(vec2 p){ p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }

float wob(float a, float t, float ph){
  return 0.50 * sin(a * 2.0 + t * 0.90 + ph)
       + 0.32 * sin(a * 3.0 - t * 1.25 + ph * 1.7)
       + 0.18 * sin(a * 5.0 + t * 1.85 + ph * 0.6)
       + uJit * 0.22 * sin(a * 9.0 + t * 6.5 + ph * 2.0);
}

vec4 over(vec4 dst, vec4 src){ return src + dst * (1.0 - src.a); }

void main(){
  vec2 p = (gl_FragCoord.xy - 0.5 * uRes) / (0.5 * min(uRes.x, uRes.y));
  float r = length(p);
  float a = atan(p.y, p.x);
  float R0 = 0.56 * uScale;
  float amp = 0.030 + 0.115 * uEnergy + 0.10 * uLevel;

  // three overlapping lobes with different phases make the silhouette feel alive
  float s0 = wob(a, uT, 0.0) * amp;
  float s1 = wob(a, uT * 1.17, 2.1) * amp * 1.25;
  float s2 = wob(a, uT * 0.83, 4.2) * amp * 0.8;
  float d0 = r - R0 * (1.0 + s0);
  float d1 = r - R0 * (0.95 + s1);
  float d2 = r - R0 * (0.88 + s2);
  float f0 = 0.016, f1 = 0.040, f2 = 0.085;
  float a0 = smoothstep(f0, -f0, d0);
  float a1 = smoothstep(f1, -f1, d1);
  float a2 = smoothstep(f2, -f2, d2);

  // interior flow, rotated by the thinking swirl
  float sw = uSwirl * (1.5 - r * 1.8) * (1.0 + 0.4 * sin(uT2 * 0.4));
  float cs = cos(sw), sn = sin(sw);
  vec2 q = mat2(cs, -sn, sn, cs) * p;
  float f = 0.5 + 0.5 * sin(q.x * 4.0 + uT2 * 0.8 + sin(q.y * 3.0 - uT2 * 0.6));
  float g = 0.5 + 0.5 * sin(q.y * 3.6 - uT2 * 0.7 + sin(q.x * 2.4 + uT2 * 0.5));
  float m = smoothstep(0.12, 0.92, f * 0.62 + g * 0.5);
  vec3 body = mix(uC2, uC1, m);
  float spiral = 0.5 + 0.5 * sin(a * 3.0 + r * 10.0 + uT2 * 2.4);
  body += uC3 * spiral * uSwirl * smoothstep(R0, 0.0, r) * smoothstep(0.02, 0.30, r) * 0.30;
  float hl = smoothstep(0.85, 0.0, length(p - uLight * R0) / R0);
  body += uC3 * hl * (0.30 + 0.25 * uLevel);
  float rn = r / R0;
  float rim = smoothstep(0.55, 1.05, rn);
  float side = dot(p / max(r, 0.001), normalize(uLight + 0.0001)); // facing the light
  body *= (1.0 - 0.20 * rim) * (0.80 + 0.20 * smoothstep(-1.0, 0.8, side) * smoothstep(0.1, 0.9, rn) + 0.20 * (1.0 - smoothstep(0.1, 0.9, rn)));

  vec4 acc = vec4(0.0);
  // outer glow
  float dist = max(r - R0 * (1.0 + amp), 0.0);
  float glow = exp(-dist * 4.6) * (0.20 + 0.50 * uGlow) * (0.8 + 0.5 * uLevel) * smoothstep(0.98, 0.62, r);
  vec3 gcol = mix(uC1, uC2, 0.5);
  acc = over(acc, vec4(gcol * glow * 0.42, glow * 0.40));
  // ripples that leave the orb
  for (int k = 0; k < 3; k++){
    float prog = uRip[k].x, str = uRip[k].y;
    float rr = R0 * (1.04 + prog * 0.62);
    float w = 0.016 + 0.034 * prog;
    float ring = exp(-pow((r - rr) / w, 2.0)) * str * smoothstep(0.0, 0.18, prog) * (1.0 - prog) * (1.0 - prog);
    acc = over(acc, vec4(uC3 * ring * 0.34, ring * 0.26));
  }
  // layers
  acc = over(acc, vec4(uC2 * 0.78 * a1 * 0.55, a1 * 0.55));
  acc = over(acc, vec4(body * a0 * 0.95, a0 * 0.95));
  acc = over(acc, vec4(uC3 * a2 * 0.16, a2 * 0.16));

  // Hermes satellite with a short trail
  if (uSatA.w > 0.001){
    for (int j = 0; j < 6; j++){
      float fj = float(j);
      float ang = uSatA.x - fj * 0.12;
      vec2 sp = vec2(cos(ang), sin(ang)) * uSatA.y;
      float size = uSatA.z * (1.0 - fj * 0.1);
      float dd = length(p - sp);
      float core = smoothstep(size, size * 0.2, dd);
      float halo = exp(-dd / (size * 1.8)) * 0.35;
      float al = uSatA.w * (1.0 - fj / 6.0) * (1.0 - fj / 6.0);
      float v = clamp(core + halo, 0.0, 1.0) * al;
      acc = over(acc, vec4(uSatC * v, v * 0.9));
    }
  }

  // fade everything out before the canvas border, so a satellite halo near the edge never shows
  // up as a hard vertical/horizontal cut (premultiplied alpha: scale rgb and a together)
  vec2 uv = gl_FragCoord.xy / uRes;
  float edge = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
  acc *= smoothstep(0.0, 0.14, edge);

  float lum = dot(acc.rgb, vec3(0.30, 0.59, 0.11));
  vec3 rgb = mix(acc.rgb, vec3(lum), uSat * 0.88);
  rgb += (hash(gl_FragCoord.xy) - 0.5) * (1.6 / 255.0) * acc.a;
  gl_FragColor = vec4(min(rgb, vec3(1.0)), clamp(acc.a, 0.0, 1.0));
}
`;

// ---- renderer ---------------------------------------------------------------------------------

export class Orb {
  constructor(canvas, { getLevels, reducedMotion = false, onAccent } = {}) {
    this.canvas = canvas;
    this.getLevels = getLevels || (() => ({ mic: 0, out: 0 }));
    this.reduced = reducedMotion;
    this.onAccent = onAccent || (() => {});

    this.mode = 'idle';
    this.emotion = 'neutral';
    this.hermes = false;

    // smoothed render state
    this.s = {
      energy: 0.12, speed: 0.5, scale: 1, swirl: 0, glow: 0.5, sat: 0, jitter: 0,
      mic: 0, out: 0, level: 0, hueShift: 0, chromaMul: 1, lightShift: 0,
      colors: PALETTE.idle.map((c) => [...c]),
      themeK: 0,
    };
    this.theme = 'dark';
    this.tShape = Math.random() * 20;
    this.tFlow = Math.random() * 20;
    this.clock = 0;
    this.slowOut = 0;
    this.ripples = [
      { prog: 1, str: 0 }, { prog: 1, str: 0 }, { prog: 1, str: 0 },
    ];
    this.lastRipple = -10;
    this.sat = { mix: 0, angle: 1.2, wobble: 0, bump: 0 };

    this.raf = 0;
    this.last = 0;
    this.running = false;
    this.slowFrames = 0;
    this.dprCap = 1.75;
    this.accentAt = 0;
    this.gl = null;

    this.initContext();
    this.resize();
    this.onVisibility = () => (document.hidden ? this.pause() : this.start());
    document.addEventListener('visibilitychange', this.onVisibility);
    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(canvas);
  }

  // ---- public ----

  setMode(mode) {
    if (!MODES[mode] || mode === this.mode) return;
    this.mode = mode;
    this.kick();
  }

  setEmotion(emotion) {
    const e = EMOTION[emotion] ? emotion : 'neutral';
    if (e === this.emotion) return;
    this.emotion = e;
    this.kick();
  }

  setHermes(active) {
    if (active === this.hermes) return;
    this.hermes = active;
    if (!active) this.sat.bump = 1; // swallowed back: a small pulse when it merges
    this.kick();
  }

  /** 'light' darkens + saturates the palette so the orb keeps its body on a light page */
  setTheme(theme) {
    this.theme = theme === 'light' ? 'light' : 'dark';
    if (!this.themeApplied && this.s) this.s.themeK = this.theme === 'light' ? 1 : 0; // first call: no fade from the wrong theme
    this.themeApplied = true;
    this.kick();
  }

  setReducedMotion(v) {
    this.reduced = v;
    this.kick();
  }

  start() {
    if (this.running) return;
    this.running = true;
    this.last = performance.now();
    this.raf = requestAnimationFrame((t) => this.frame(t));
  }

  pause() {
    this.running = false;
    cancelAnimationFrame(this.raf);
  }

  kick() {
    if (!this.running && !document.hidden) this.start();
  }

  // ---- context ----

  initContext() {
    const canvas = this.canvas;
    const gl = canvas.getContext('webgl', {
      alpha: true, premultipliedAlpha: true, antialias: false, depth: false, stencil: false,
      powerPreference: 'low-power',
    });
    if (!gl) {
      this.ctx2d = canvas.getContext('2d');
      this.gl = null;
      return;
    }
    const compile = (type, src) => {
      const sh = gl.createShader(type);
      gl.shaderSource(sh, src);
      gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
        throw new Error(gl.getShaderInfoLog(sh) || 'shader compile failed');
      }
      return sh;
    };
    try {
      const prog = gl.createProgram();
      gl.attachShader(prog, compile(gl.VERTEX_SHADER, VERT));
      gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FRAG));
      gl.linkProgram(prog);
      if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error('link failed');
      gl.useProgram(prog);
      const buf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
      const loc = gl.getAttribLocation(prog, 'aPos');
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
      this.u = {};
      ['uRes', 'uT', 'uT2', 'uEnergy', 'uLevel', 'uScale', 'uSwirl', 'uGlow', 'uSat', 'uJit',
        'uC1', 'uC2', 'uC3', 'uSatC', 'uSatA', 'uLight'].forEach((n) => { this.u[n] = gl.getUniformLocation(prog, n); });
      this.u.uRip = [0, 1, 2].map((i) => gl.getUniformLocation(prog, `uRip[${i}]`));
      gl.clearColor(0, 0, 0, 0);
      this.gl = gl;
      canvas.addEventListener('webglcontextlost', (e) => { e.preventDefault(); this.contextLost = true; });
      canvas.addEventListener('webglcontextrestored', () => {
        this.contextLost = false;
        this.initContext();
        this.resize();
      });
    } catch (err) {
      console.warn('Orb: WebGL unavailable, using 2D fallback', err);
      this.gl = null;
      this.ctx2d = canvas.getContext('2d');
    }
  }

  resize() {
    const rect = this.canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, this.dprCap);
    const w = Math.max(2, Math.round(rect.width * dpr));
    const h = Math.max(2, Math.round(rect.height * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
      if (this.gl) this.gl.viewport(0, 0, w, h);
    }
    this.kick();
  }

  // ---- simulation ----

  step(dt, now) {
    const s = this.s;
    const base = MODES[this.mode];
    const emo = EMOTION[this.emotion];
    const { mic, out } = this.getLevels();

    // asymmetric smoothing: fast attack, slow release (feels like breath, not a meter)
    const follow = (cur, target) => damp(cur, target, target > cur ? 22 : 5, dt);
    s.mic = follow(s.mic, mic);
    s.out = follow(s.out, out);
    this.slowOut = damp(this.slowOut, out, 3, dt);

    const breath = 0.5 + 0.5 * Math.sin((this.clock * Math.PI * 2) / 5.6);
    const driven = base.in * s.mic + base.out * s.out;
    const reducedK = this.reduced ? 0.35 : 1;

    const targetEnergy = (base.energy * emo.energy + 0.9 * driven * emo.energy) * reducedK;
    const targetSpeed = base.speed * emo.speed * (this.reduced ? 0 : 1);
    const targetScale = base.scale * emo.scale * (1 + 0.025 * breath * (this.mode === 'offline' ? 0 : 1)) + 0.08 * driven * reducedK;
    const lam = 3.2;
    s.energy = damp(s.energy, targetEnergy, lam, dt);
    s.speed = damp(s.speed, targetSpeed, lam, dt);
    s.scale = damp(s.scale, targetScale, 9, dt);
    s.swirl = damp(s.swirl, this.reduced ? 0 : base.swirl, 2.2, dt);
    s.glow = damp(s.glow, base.glow * (0.85 + 0.3 * breath) + 0.3 * driven, 4, dt);
    s.sat = damp(s.sat, base.sat, 2.5, dt);
    s.jitter = damp(s.jitter, emo.jitter * (this.mode === 'speaking' ? 1 : 0.3), 4, dt);
    s.level = driven;
    s.hueShift = damp(s.hueShift, this.mode === 'offline' ? 0 : emo.hue, 2.4, dt);
    s.chromaMul = damp(s.chromaMul, this.mode === 'offline' ? 1 : emo.chroma, 2.4, dt);
    s.lightShift = damp(s.lightShift, emo.light, 2.4, dt);
    s.themeK = damp(s.themeK, this.theme === 'light' ? 1 : 0, 6, dt);

    this.tShape += dt * s.speed * 1.0;
    this.tFlow += dt * (0.25 + s.speed * 0.8);

    // palette
    const pal = PALETTE[this.mode];
    for (let i = 0; i < 3; i++) {
      const c = s.colors[i];
      c[0] = damp(c[0], pal[i][0] + s.lightShift, 3, dt);
      // blend in the a/b plane: opposite hues pass through a soft neutral instead of sweeping
      // across the whole wheel (peach -> blue would otherwise flash green on the way)
      if (c.a === undefined) { c.a = c[1] * Math.cos((c[2] * Math.PI) / 180); c.b = c[1] * Math.sin((c[2] * Math.PI) / 180); }
      const hr = (pal[i][2] * Math.PI) / 180;
      c.a = damp(c.a, pal[i][1] * Math.cos(hr), 3, dt);
      c.b = damp(c.b, pal[i][1] * Math.sin(hr), 3, dt);
      c[1] = Math.hypot(c.a, c.b);
      c[2] = ((Math.atan2(c.b, c.a) * 180) / Math.PI + 360) % 360;
    }

    // ripples: mic onsets while listening, syllable onsets while speaking
    for (const r of this.ripples) {
      if (r.prog < 1) r.prog = Math.min(1, r.prog + dt * (this.reduced ? 0 : 0.85));
    }
    if (!this.reduced) {
      const spawn = (strength) => {
        const r = this.ripples.find((x) => x.prog >= 1) || this.ripples[0];
        r.prog = 0;
        r.str = Math.min(1, 0.35 + strength);
        this.lastRipple = this.clock;
      };
      if ((this.mode === 'listening' || this.mode === 'user') && s.mic > 0.1 && this.clock - this.lastRipple > 0.5) {
        spawn(s.mic);
      } else if (this.mode === 'speaking' && s.out - this.slowOut > 0.1 && this.clock - this.lastRipple > 0.28) {
        spawn(s.out);
      }
    }

    // Hermes satellite: detach -> orbit -> merge back
    const sat = this.sat;
    sat.mix = damp(sat.mix, this.hermes ? 1 : 0, this.hermes ? 2.6 : 3.2, dt);
    if (sat.mix < 0.002 && !this.hermes) sat.mix = 0;
    if (!this.reduced) sat.angle += dt * (0.55 + 0.25 * sat.mix);
    sat.wobble = Math.sin(this.clock * 0.9) * 0.05;
    sat.bump = damp(sat.bump, 0, 3, dt);
    if (sat.bump > 0.01) s.scale += 0.04 * sat.bump;

    // accent colour for the surrounding UI (throttled)
    if (now - this.accentAt > 120) {
      this.accentAt = now;
      const [L, C, H] = s.colors[0];
      // on a light page the accent doubles as text and button fill: push it darker for contrast
      this.onAccent(`oklch(${(L - 0.27 * s.themeK).toFixed(3)} ${(C * s.chromaMul * (1 - 0.85 * s.sat) * (1 + 0.25 * s.themeK)).toFixed(3)} ${(H + s.hueShift).toFixed(1)})`, this.mode);
    }
  }

  frame(now) {
    if (!this.running) return;
    // rAF timestamps can precede the performance.now() taken in start(): a negative dt would make damp() blow up
    const dt = Math.min(0.05, Math.max(0.001, (now - this.last) / 1000 || 0.016));
    this.last = Math.max(now, this.last);
    this.clock += dt;
    if (!this.contextLost) {
      this.step(dt, now);
      this.draw();
    }

    // adaptive resolution: if we cannot hold ~50 fps for a second, shrink the render scale
    if (dt > 0.024) this.slowFrames += 1; else this.slowFrames = Math.max(0, this.slowFrames - 1);
    if (this.slowFrames > 45 && this.dprCap > 1) {
      this.dprCap = Math.max(1, this.dprCap - 0.25);
      this.slowFrames = 0;
      this.resize();
    }

    if (this.reduced) {
      // low-rate loop: opacity/colour transitions only
      setTimeout(() => { if (this.running) this.raf = requestAnimationFrame((t) => this.frame(t)); }, 90);
    } else {
      this.raf = requestAnimationFrame((t) => this.frame(t));
    }
  }

  // ---- drawing ----

  layerRgb() {
    const s = this.s;
    return s.colors.map(([L, C, H]) => oklchToSrgb(L - 0.12 * s.themeK, C * s.chromaMul * (1 + 0.14 * s.themeK), H + s.hueShift));
  }

  satParams() {
    const s = this.s;
    const sat = this.sat;
    const R0 = 0.56 * s.scale;
    const dist = lerp(R0 * 0.35, R0 * (1.32 + sat.wobble), sat.mix);
    return {
      angle: sat.angle,
      dist,
      size: lerp(0.02, 0.058, sat.mix),
      alpha: Math.min(1, sat.mix * 1.6),
    };
  }

  draw() {
    const [c1, c2, c3] = this.layerRgb();
    const s = this.s;
    const sp = this.satParams();
    const satRgb = oklchToSrgb(0.86, 0.1, 195);
    const lx = Math.cos(this.clock * 0.13) * 0.35 - 0.12;
    const ly = Math.sin(this.clock * 0.11) * 0.28 + 0.28;

    if (this.gl) {
      const gl = this.gl;
      const u = this.u;
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.uniform2f(u.uRes, this.canvas.width, this.canvas.height);
      gl.uniform1f(u.uT, this.tShape);
      gl.uniform1f(u.uT2, this.tFlow);
      gl.uniform1f(u.uEnergy, s.energy);
      gl.uniform1f(u.uLevel, s.level);
      gl.uniform1f(u.uScale, s.scale);
      gl.uniform1f(u.uSwirl, s.swirl);
      gl.uniform1f(u.uGlow, s.glow);
      gl.uniform1f(u.uSat, s.sat);
      gl.uniform1f(u.uJit, s.jitter);
      gl.uniform3fv(u.uC1, c1);
      gl.uniform3fv(u.uC2, c2);
      gl.uniform3fv(u.uC3, c3);
      gl.uniform3fv(u.uSatC, satRgb);
      gl.uniform4f(u.uSatA, sp.angle, sp.dist, sp.size, sp.alpha);
      gl.uniform2f(u.uLight, lx, ly);
      this.ripples.forEach((r, i) => gl.uniform2f(u.uRip[i], r.prog, r.prog >= 1 ? 0 : r.str));
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      return;
    }
    this.draw2d(c1, c2, c3, sp, satRgb);
  }

  draw2d(c1, c2, c3, sp, satRgb) {
    const ctx = this.ctx2d;
    if (!ctx) return;
    const s = this.s;
    const W = this.canvas.width;
    const H = this.canvas.height;
    const half = Math.min(W, H) / 2;
    const css = (c, a = 1) => `rgba(${Math.round(c[0] * 255)},${Math.round(c[1] * 255)},${Math.round(c[2] * 255)},${a})`;
    const gray = (c) => {
      const l = c[0] * 0.3 + c[1] * 0.59 + c[2] * 0.11;
      return c.map((v) => lerp(v, l, s.sat * 0.88));
    };
    const [a1, a2, a3] = [gray(c1), gray(c2), gray(c3)];
    ctx.clearRect(0, 0, W, H);
    ctx.save();
    ctx.translate(W / 2, H / 2);
    const R0 = 0.56 * s.scale * half;
    const amp = 0.03 + 0.115 * s.energy + 0.1 * s.level;
    const wob = (a, t, ph) => 0.5 * Math.sin(a * 2 + t * 0.9 + ph) + 0.32 * Math.sin(a * 3 - t * 1.25 + ph * 1.7)
      + 0.18 * Math.sin(a * 5 + t * 1.85 + ph * 0.6) + s.jitter * 0.22 * Math.sin(a * 9 + t * 6.5 + ph * 2);

    // glow
    const glow = ctx.createRadialGradient(0, 0, R0 * 0.8, 0, 0, R0 * 1.9);
    glow.addColorStop(0, css(a1, 0.35 * (0.3 + s.glow)));
    glow.addColorStop(1, css(a2, 0));
    ctx.fillStyle = glow;
    ctx.fillRect(-W / 2, -H / 2, W, H);

    // ripples
    ctx.lineWidth = Math.max(1, half * 0.012);
    for (const r of this.ripples) {
      if (r.prog >= 1) continue;
      ctx.strokeStyle = css(a3, 0.45 * r.str * (1 - r.prog) ** 2);
      ctx.beginPath();
      ctx.arc(0, 0, R0 * (1.04 + r.prog * 0.62), 0, Math.PI * 2);
      ctx.stroke();
    }

    const layer = (k, t, ph, sc, fill, alpha) => {
      ctx.beginPath();
      for (let i = 0; i <= 120; i++) {
        const a = (i / 120) * Math.PI * 2;
        const rr = R0 * (sc + wob(a, t, ph) * amp * k);
        const x = Math.cos(a) * rr, y = Math.sin(a) * rr;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.globalAlpha = alpha;
      ctx.fillStyle = fill;
      ctx.fill();
    };
    layer(1.25, this.tShape * 1.17, 2.1, 0.95, css(a2), 0.55);
    const body = ctx.createRadialGradient(-R0 * 0.25, -R0 * 0.3, R0 * 0.05, 0, 0, R0 * 1.1);
    body.addColorStop(0, css(a3));
    body.addColorStop(0.45, css(a1));
    body.addColorStop(1, css(a2));
    layer(1, this.tShape, 0, 1, body, 0.95);
    ctx.globalAlpha = 1;

    if (sp.alpha > 0.01) {
      const sc = gray(satRgb);
      for (let j = 0; j < 6; j++) {
        const ang = sp.angle - j * 0.12;
        ctx.fillStyle = css(sc, sp.alpha * (1 - j / 6) ** 2);
        ctx.beginPath();
        ctx.arc(Math.cos(ang) * sp.dist * half, Math.sin(ang) * sp.dist * half, sp.size * half * (1 - j * 0.1), 0, Math.PI * 2);
        ctx.fill();
      }
    }
    ctx.restore();
  }
}
