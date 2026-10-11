// Hermes hand-offs as cards pinned above the control dock.
//
//   collapsed:  [glyph] Hermes pracuje · 12 s            v
//               <latest progress line>
//   expanded:   task title + a timeline of everything Hermes reported (started / progress / result)
//
// While working the card shows an indeterminate bar, a ticking timer and an "Przerwij" button that asks the
// server to cancel that one task (`cancel_task{id}`); done / error / cancelled freeze the timer. Cards are keyed
// by the server's `handoff.id` (several tasks can run at once). Finished cards stay until dismissed (or 90 s),
// the result is also logged in the chat.

import { t } from './i18n.js';

export function formatElapsed(ms) {
  const s = Math.max(0, Math.round(ms / 1000));
  if (s < 60) return `${s} s`;
  return `${Math.floor(s / 60)} min ${String(s % 60).padStart(2, '0')} s`;
}

const ICON = (id) => `<svg class="ico" aria-hidden="true"><use href="icons.svg#i-${id}"/></svg>`;
const el = (tag, cls, text) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
};

export class Tasks {
  constructor({ mountEl, onActiveChange, onCancel }) {
    this.mount = mountEl;
    this.onActiveChange = onActiveChange || (() => {});
    this.onCancel = onCancel || (() => {});
    this.items = new Map(); // key -> task
    this.cards = new Map(); // key -> { root, parts }
    this.ticker = 0;
    this.lastActive = false;
  }

  get active() {
    for (const task of this.items.values()) if (task.state === 'working') return true;
    return false;
  }

  get workingTask() {
    for (const task of this.items.values()) if (task.state === 'working') return task;
    return null;
  }

  handle(evt) {
    // `id` is stable for one task; the task text is only a fallback for servers that do not send it
    const key = evt.id || evt.task || '(task)';
    let task = this.items.get(key);
    const now = Date.now();
    if (evt.state === 'cancelled') {
      // sent by the server after cancel_task / interrupt{scope:all} / the spoken "anuluj zadanie"; no `done` follows
      if (!task || task.state !== 'working') return task || null;
      task.state = 'cancelled';
      task.cancelling = false;
      task.end = now;
      task.steps.push({ kind: 'cancelled', text: '', at: now });
      this.scheduleExpire(task, 12000);
      this.render();
      return task;
    }
    if (!task || (!evt.id && evt.state === 'started' && task.state !== 'working')) {
      task = { key, id: evt.id || null, task: evt.task || t('hermes.default_task'), state: 'working', t0: now, end: 0, steps: [], open: false, cancelling: false };
      this.items.set(key, task);
    }
    let text = (evt.text || '').trim();
    // the server puts its routing reason into `started.text` ("router", "rules", ...): a label, not something to read
    if (evt.state === 'started' && /^[a-z_-]{1,24}$/.test(text)) text = '';
    if (evt.state === 'started') {
      task.steps.push({ kind: 'started', text, at: now });
    } else if (evt.state === 'progress') {
      const raw = evt.text || '';
      const prev = task.steps[task.steps.length - 1];
      // Hermes streams its answer token by token ("\n\nD", "ziś piątek, …"): glue such fragments back into
      // one line instead of showing every chunk as a step. A capitalised or newline-led chunk starts a new line.
      const fragment = prev && prev.kind === 'progress' && prev.raw !== undefined && now - prev.at < 2000 && raw && !/^[\n\r]/.test(raw) && !/^[A-ZĄĆĘŁŃÓŚŹŻ]/.test(raw);
      if (fragment) {
        prev.raw += raw;
        prev.text = prev.raw.trim();
        prev.at = now;
      } else if (text) {
        task.steps.push({ kind: 'progress', text, raw, at: now });
      }
    } else if (evt.state === 'done') {
      task.state = 'done';
      task.end = now;
      // the streamed answer and the result are the same words: show them once
      const last = task.steps[task.steps.length - 1];
      if (last && last.kind === 'progress' && last.text === text) task.steps.pop();
      task.steps.push({ kind: 'done', text, at: now });
      this.scheduleExpire(task);
    } else if (evt.state === 'error') {
      task.state = 'error';
      task.end = now;
      task.steps.push({ kind: 'error', text: text || t('hermes.default_error'), at: now });
      task.open = true; // an error deserves attention
      this.scheduleExpire(task);
    }
    this.render();
    return task;
  }

  scheduleExpire(task, ms = 90000) {
    clearTimeout(task.expire);
    task.expire = setTimeout(() => this.dismiss(task.key), ms);
  }

  /** the connection dropped: the server forgot about running work */
  failRunning(message) {
    let changed = false;
    for (const task of this.items.values()) {
      if (task.state === 'working') {
        task.state = 'error';
        task.end = Date.now();
        task.steps.push({ kind: 'error', text: message, at: task.end });
        task.open = true;
        this.scheduleExpire(task);
        changed = true;
      }
    }
    if (changed) this.render();
  }

  /** the Przerwij button: ask the server; the card flips to "cancelled" when `handoff{cancelled}` arrives */
  requestCancel(key) {
    const task = this.items.get(key);
    if (!task || task.state !== 'working' || task.cancelling) return;
    task.cancelling = true;
    this.onCancel(task);
    this.render();
  }

  /** the server refused (e.g. "no running Hermes task": it finished a moment ago): re-enable the buttons */
  cancelRefused() {
    let changed = false;
    for (const task of this.items.values()) if (task.cancelling) { task.cancelling = false; changed = true; }
    if (changed) this.render();
  }

  dismiss(key) {
    const task = this.items.get(key);
    if (task) clearTimeout(task.expire);
    this.items.delete(key);
    this.render();
  }

  toggle(key) {
    const task = this.items.get(key);
    if (!task) return;
    task.open = !task.open;
    this.render();
  }

  title(task) {
    return task.state === 'working' ? t('hermes.working') : task.state === 'done' ? t('hermes.done') : task.state === 'cancelled' ? t('hermes.cancelled') : t('hermes.error');
  }

  /** the one line shown while collapsed */
  summary(task) {
    for (let i = task.steps.length - 1; i >= 0; i--) {
      const s = task.steps[i];
      if (s.text) return s.text;
    }
    return task.task;
  }

  render() {
    for (const [key, card] of this.cards) {
      if (!this.items.has(key)) { card.root.remove(); this.cards.delete(key); }
    }
    for (const task of this.items.values()) {
      let card = this.cards.get(task.key);
      if (!card) { card = this.build(task); this.cards.set(task.key, card); this.mount.appendChild(card.root); }
      this.paint(task, card);
    }
    this.mount.hidden = this.items.size === 0;

    const active = this.active;
    if (active && !this.ticker) this.ticker = setInterval(() => this.tick(), 1000);
    if (!active && this.ticker) { clearInterval(this.ticker); this.ticker = 0; }
    if (active !== this.lastActive) {
      this.lastActive = active;
      this.onActiveChange(active);
    }
  }

  build(task) {
    const root = el('section', 'hcard');
    root.dataset.key = task.key;
    if (task.id) root.dataset.id = task.id;
    const row = el('div', 'hrow');
    const head = el('button', 'hcard-head');
    head.type = 'button';
    head.addEventListener('click', () => this.toggle(task.key));
    const glyph = el('span', 'hglyph');
    glyph.setAttribute('aria-hidden', 'true');
    const text = el('span', 'htext');
    const title = el('b', 'htitle');
    const sub = el('span', 'hsub');
    text.append(title, sub);
    const elapsed = el('span', 'helapsed');
    const chev = el('span', 'hchev');
    chev.innerHTML = ICON('chevron');
    head.append(glyph, text, elapsed, chev);
    const stop = el('button', 'hcancel');
    stop.type = 'button';
    stop.innerHTML = `${ICON('stop')}<span class="hcancel-text"></span>`;
    stop.addEventListener('click', () => this.requestCancel(task.key));
    row.append(head, stop);
    const bar = el('div', 'hbar');
    bar.setAttribute('aria-hidden', 'true');
    bar.appendChild(el('i'));
    const body = el('div', 'hbody');
    const inner = el('div', 'hbody-inner');
    const taskLine = el('p', 'htask');
    const steps = el('ol', 'hsteps');
    const foot = el('div', 'hfoot');
    const close = el('button', 'btn small', t('hermes.dismiss'));
    close.type = 'button';
    close.addEventListener('click', () => this.dismiss(task.key));
    foot.appendChild(close);
    inner.append(taskLine, steps, foot);
    body.appendChild(inner);
    root.append(row, bar, body);
    return { root, head, stop, stopText: stop.querySelector('.hcancel-text'), glyph, title, sub, elapsed, taskLine, steps, close, body };
  }

  paint(task, card) {
    const { root, head } = card;
    root.dataset.state = task.state;
    root.dataset.open = String(task.open);
    head.setAttribute('aria-expanded', String(task.open));
    head.setAttribute('aria-label', `${this.title(task)}. ${task.open ? t('hermes.collapse') : t('hermes.expand')}`);
    card.glyph.innerHTML = task.state === 'done' ? ICON('check') : task.state === 'error' ? ICON('warn') : task.state === 'cancelled' ? ICON('stop') : '<span class="orbit"></span>';
    card.title.textContent = this.title(task);
    card.sub.textContent = this.summary(task);
    card.elapsed.textContent = formatElapsed((task.end || Date.now()) - task.t0);
    card.taskLine.textContent = task.task;
    card.close.textContent = t('hermes.dismiss');
    card.body.inert = !task.open;
    card.stop.hidden = task.state !== 'working';
    card.stop.disabled = task.cancelling;
    card.stopText.textContent = task.cancelling ? t('hermes.cancelling') : t('hermes.cancel');
    card.stop.setAttribute('aria-label', `${task.cancelling ? t('hermes.cancelling') : t('hermes.cancel_task')}: ${task.task}`);

    // timeline (rebuilt: it is short)
    card.steps.textContent = '';
    for (const s of task.steps) {
      const li = el('li');
      li.dataset.kind = s.kind;
      li.appendChild(el('span', 'sdot'));
      const col = el('div', 'scol');
      col.appendChild(el('b', null, t(`hermes.step_${s.kind}`)));
      if (s.text) col.appendChild(el('p', null, s.text));
      const time = el('time', null, `+${formatElapsed(s.at - task.t0)}`);
      li.append(col, time);
      card.steps.appendChild(li);
    }
    if (task.state === 'working') {
      const li = el('li');
      li.dataset.kind = 'live';
      li.append(el('span', 'sdot'), el('div', 'scol', t('hermes.working_sub')));
      card.steps.appendChild(li);
    }
  }

  tick() {
    for (const [key, card] of this.cards) {
      const task = this.items.get(key);
      if (task && task.state === 'working') card.elapsed.textContent = formatElapsed(Date.now() - task.t0);
    }
  }

  /** the UI language changed: repaint every card */
  retranslate() {
    for (const task of this.items.values()) {
      const card = this.cards.get(task.key);
      if (card) this.paint(task, card);
    }
  }
}
