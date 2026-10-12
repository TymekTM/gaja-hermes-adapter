// Conversation transcript: bubbles that stream.
//
//   you:   a ghost bubble while the user is speaking / being transcribed -> the final transcript
//   gaja:  one bubble per turn. Text arrives twice in the protocol:
//            assistant_delta  raw LLM text as it streams            -> the dim "tail" with a caret
//            utterance        the sentence that is about to be spoken -> a sentence span
//          and the page's audio engine says when a sentence really *sounds* (queued -> playing -> done),
//          so the highlight follows the speaker, not the network. Servers that never send
//          assistant_delta still work: sentences alone make the bubble.
//   note:  Hermes / error lines (compact, centred)
//   tool:  a small chip for a local tool Gaja ran ("🌤 pogoda · Warszawa"); the spoken answer follows as a bubble
//
// The current server sends no assistant_delta (the bubble is built from `utterance` alone); delta() stays for
// older servers and the mock's `?deltas=1`.

import { t, emotionLabel, getUiLang } from './i18n.js';

const compact = (s) => (s || '').toLowerCase().replace(/[^\p{L}\p{N}]+/gu, '');

const TOOL_GLYPHS = {
  get_time: '🕐', get_weather: '🌤', calculate: '🧮', convert_units: '⚖️', exchange_rate: '💱',
  wiki_summary: '📖', server_status: '🖥', next_holiday: '📅',
};
const TOOL_LABELS = Object.fromEntries(Object.keys(TOOL_GLYPHS).map((k) => [k, true]));

/** the one argument worth showing next to the tool name (arguments come from the model: show defensively) */
function toolDetail(name, a) {
  const str = (v) => (v == null ? '' : String(v).trim());
  const clip = (v, n = 40) => (v.length > n ? `${v.slice(0, n - 1)}…` : v);
  switch (name) {
    case 'get_weather': return clip([str(a.location), ['now', 'teraz'].includes(str(a.when).toLowerCase()) ? '' : str(a.when)].filter(Boolean).join(', '));
    case 'get_time': return clip(str(a.timezone));
    case 'calculate': return clip(str(a.expression));
    case 'convert_units': return clip([str(a.value), str(a.from_unit), a.to_unit ? `→ ${str(a.to_unit)}` : ''].filter(Boolean).join(' '));
    case 'exchange_rate': return clip([str(a.amount), str(a.currency), a.to ? `→ ${str(a.to)}` : ''].filter(Boolean).join(' '));
    case 'wiki_summary': return clip(str(a.topic));
    default: return '';
  }
}

export class Chat {
  constructor({ listEl, scrollEl, onChange, onAssistantTurn }) {
    this.list = listEl;
    this.scroller = scrollEl;
    this.onChange = onChange || (() => {});
    this.onAssistantTurn = onAssistantTurn || (() => {});
    this.turn = null;        // current assistant bubble state
    this.ghost = null;       // the user's "listening…" bubble
    this.sentences = new Map(); // utterance id -> { el, bubble }
    this.stick = true;       // follow the newest text unless the user scrolled away
    this.showStyle = false;  // metrics mode: rate / pitch of each sentence

    this.scroller.addEventListener('scroll', () => {
      const el = this.scroller;
      this.stick = el.scrollHeight - el.scrollTop - el.clientHeight < 48;
    }, { passive: true });
  }

  get empty() { return this.list.children.length === 0 && !this.ghost; }

  // ---- helpers ---------------------------------------------------------------------------

  timeStr() {
    return new Intl.DateTimeFormat(getUiLang() === 'en' ? 'en-GB' : 'pl-PL', { hour: '2-digit', minute: '2-digit', hour12: false }).format(new Date());
  }

  makeItem(kind, who) {
    const li = document.createElement('li');
    li.className = `msg ${kind}`;
    const meta = document.createElement('div');
    meta.className = 'msg-meta';
    const w = document.createElement('span');
    w.className = 'who';
    w.textContent = who;
    const time = document.createElement('time');
    time.textContent = this.timeStr();
    meta.append(w, time);
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    li.append(meta, bubble);
    return { li, meta, bubble };
  }

  append(li) {
    this.list.appendChild(li);
    while (this.list.children.length > 150) this.list.firstChild.remove();
    this.scrollDown(true);
    this.onChange();
  }

  scrollDown(force = false) {
    if (!force && !this.stick) return;
    const el = this.scroller;
    requestAnimationFrame(() => { el.scrollTop = el.scrollHeight; });
  }

  // ---- user ----------------------------------------------------------------------------------

  /** the user started speaking: a live bubble with a little equaliser */
  userListening() {
    this.closeTurn();
    if (this.ghost) return;
    const { li, bubble } = this.makeItem('you ghost', t('chat.you'));
    bubble.innerHTML = '<span class="eq" aria-hidden="true"><i></i><i></i><i></i><i></i><i></i></span><span class="ghost-text"></span>';
    bubble.querySelector('.ghost-text').textContent = t('chat.listening_ghost');
    li.querySelector('time').remove();
    this.ghost = li;
    this.append(li);
  }

  userTranscribing() {
    if (!this.ghost) return;
    this.ghost.querySelector('.ghost-text').textContent = t('chat.transcribing');
    this.ghost.classList.add('busy');
  }

  dropGhost() {
    if (!this.ghost) return;
    this.ghost.remove();
    this.ghost = null;
    this.onChange();
  }

  addUser(text) {
    this.closeTurn();
    this.dropGhost();
    const { li, bubble } = this.makeItem('you', t('chat.you'));
    bubble.textContent = text;
    this.append(li);
  }

  // ---- assistant -----------------------------------------------------------------------------

  ensureTurn() {
    if (this.turn) return this.turn;
    this.dropGhost();
    const { li, meta, bubble } = this.makeItem('gaja', t('chat.gaja'));
    const tail = document.createElement('span');
    tail.className = 'tail';
    bubble.appendChild(tail);
    this.turn = { li, meta, bubble, tail, tailText: '', spoken: 0 };
    this.append(li);
    return this.turn;
  }

  closeTurn() {
    const tn = this.turn;
    if (!tn) return;
    this.finalizeTail(tn);
    const sent = [...tn.bubble.querySelectorAll('.sent')].map((el) => el.textContent.trim()).filter(Boolean).join(' ');
    const text = sent || tn.tail.textContent.trim();
    if (text) this.onAssistantTurn(text);
    this.turn = null;
  }

  /** end of a turn: text that streamed but was never voiced stays only if the bubble would be empty */
  finalizeTail(tn) {
    const left = tn.tailText.trim();
    tn.tailText = '';
    if (left && tn.spoken === 0) {
      tn.tail.textContent = left;
      tn.tail.classList.add('plain');
    } else {
      tn.tail.textContent = '';
    }
    tn.tail.classList.remove('streaming');
    if (!tn.bubble.querySelector('.sent') && !left) tn.li.remove();
  }

  delta(text) {
    if (!text) return;
    const tn = this.ensureTurn();
    tn.tailText += text;
    tn.tail.textContent = tn.tailText.replace(/^\s+/, '');
    tn.tail.classList.add('streaming');
    this.scrollDown();
  }

  /** consume the part of the streamed text that this sentence accounts for */
  consumeTail(tn, sentence) {
    const sc = compact(sentence);
    if (!sc || !tn.tailText) return;
    let acc = '';
    const buf = tn.tailText;
    // cut point = the first position where the compacted prefix contains the sentence
    for (let i = 0; i < buf.length; i++) {
      acc += compact(buf[i]);
      if (acc.length >= sc.length && acc.endsWith(sc)) {
        let end = i + 1;
        while (end < buf.length && !/[\p{L}\p{N}]/u.test(buf[end])) end++; // the sentence's own trailing punctuation / spaces
        tn.tailText = buf.slice(end);
        return;
      }
    }
  }

  sentence(id, text, style = {}) {
    const tn = this.ensureTurn();
    this.consumeTail(tn, text);
    tn.tail.textContent = tn.tailText.replace(/^\s+/, '');
    if (!tn.tailText.trim()) tn.tail.classList.remove('streaming');

    const emo = style.emotion || 'neutral';
    const span = document.createElement('span');
    span.className = 'sent';
    span.dataset.state = 'queued';
    span.dataset.emo = emo;
    if (emo !== 'neutral') span.title = emotionLabel(emo);
    if (emo !== 'neutral') {
      const dot = document.createElement('i');
      dot.className = 'emo-dot';
      dot.setAttribute('aria-hidden', 'true');
      span.appendChild(dot);
    }
    span.appendChild(document.createTextNode(text));
    if (this.showStyle) span.appendChild(this.styleBadge(style));
    span.append(' ');
    tn.bubble.insertBefore(span, tn.tail);
    tn.spoken += 1;
    this.sentences.set(id, { span, style });
    if (this.sentences.size > 80) this.sentences.delete(this.sentences.keys().next().value);
    this.scrollDown();
  }

  styleBadge(style) {
    const b = document.createElement('small');
    b.className = 'style-badge';
    const bits = [];
    if (style.rate && Math.abs(style.rate - 1) > 0.005) bits.push(`×${Number(style.rate).toFixed(2)}`);
    if (style.pitch_semitones) bits.push(`${style.pitch_semitones > 0 ? '+' : ''}${style.pitch_semitones} st`);
    if (style.energy && Math.abs(style.energy - 1) > 0.005) bits.push(`E${Number(style.energy).toFixed(1)}`);
    if (style.events && style.events.length) bits.push(style.events.join(','));
    b.textContent = bits.join(' · ');
    return b;
  }

  setShowStyle(on) {
    this.showStyle = on;
    document.body.dataset.styleBadges = on ? 'on' : 'off';
  }

  mark(id, state) {
    const rec = this.sentences.get(id);
    if (!rec) return;
    rec.span.dataset.state = state;
    if (state === 'playing') {
      if (this.stick) rec.span.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    }
  }

  /** playback finished: nothing is left "queued" or "playing" (e.g. a sentence whose audio never came) */
  settle() {
    for (const rec of this.sentences.values()) {
      const st = rec.span.dataset.state;
      if (st === 'queued' || st === 'playing') rec.span.dataset.state = 'done';
    }
    this.closeTurn();
  }

  /** barge-in / error: everything not yet finished is cut off */
  cutAll() {
    let cut = false;
    for (const rec of this.sentences.values()) {
      const st = rec.span.dataset.state;
      if (st === 'queued' || st === 'playing') { rec.span.dataset.state = 'cut'; cut = true; }
    }
    const tn = this.turn;
    if (tn) {
      if (tn.tailText.trim() || cut) {
        tn.li.classList.add('cut');
        if (!tn.meta.querySelector('.tag')) {
          const tag = document.createElement('span');
          tag.className = 'tag';
          tag.textContent = t('chat.cut');
          tn.meta.appendChild(tag);
        }
      }
      tn.tailText = '';
      tn.tail.textContent = '';
      tn.tail.classList.remove('streaming');
    } else if (cut) {
      // the cut sentences belong to a bubble that was already closed: tag the latest assistant item
      const items = this.list.querySelectorAll('.msg.gaja');
      const last = items[items.length - 1];
      if (last && !last.querySelector('.tag')) {
        last.classList.add('cut');
        const tag = document.createElement('span');
        tag.className = 'tag';
        tag.textContent = t('chat.cut');
        last.querySelector('.msg-meta').appendChild(tag);
      }
    }
    this.closeTurn();
  }

  // ---- notes ----------------------------------------------------------------------------------

  addNote(kind, text) {
    this.closeTurn();
    const li = document.createElement('li');
    li.className = `msg note ${kind}`;
    const body = document.createElement('div');
    body.className = 'note-body';
    body.textContent = text;
    li.appendChild(body);
    this.append(li);
  }

  // ---- tool chips -----------------------------------------------------------------------------

  /** `tool` event {name, arguments, ok, text}: one compact chip, e.g. "🌤 pogoda · Warszawa" */
  addTool(evt) {
    const name = String(evt.name || '');
    const label = TOOL_LABELS[name] ? t(`tool.${name}`) : name || t('tool.failed');
    const detail = toolDetail(name, evt.arguments || {});
    this.closeTurn();
    const li = document.createElement('li');
    li.className = 'msg note tool';
    const chip = document.createElement('div');
    chip.className = `tool-chip${evt.ok === false ? ' fail' : ''}`;
    chip.dataset.tool = name;
    if (evt.text) chip.title = evt.text;
    const glyph = document.createElement('span');
    glyph.className = 'tool-glyph';
    glyph.setAttribute('aria-hidden', 'true');
    glyph.textContent = evt.ok === false ? '⚠️' : TOOL_GLYPHS[name] || '🛠';
    const text = document.createElement('span');
    const parts = [label, detail];
    if (evt.ok === false) parts.push(t('tool.failed'));
    text.textContent = parts.filter(Boolean).join(' · ');
    chip.append(glyph, text);
    li.appendChild(chip);
    this.append(li);
  }

  clear() {
    this.list.textContent = '';
    this.turn = null;
    this.ghost = null;
    this.sentences.clear();
    this.onChange();
  }
}
