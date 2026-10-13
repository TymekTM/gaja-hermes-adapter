// Small DOM helpers shared by the admin panel: element builder, icons, toasts, confirm dialog.

const SVG_NS = 'http://www.w3.org/2000/svg';

/** h('div.class#id', {attr: value, onclick: fn, dataset: {...}}, ...children) */
export function h(tag, attrs, ...children) {
  const m = /^([a-z0-9-]*)((?:[.#][\w-]+)*)$/i.exec(tag);
  const el = document.createElement(m && m[1] ? m[1] : 'div');
  if (m && m[2]) {
    for (const part of m[2].match(/[.#][\w-]+/g)) {
      if (part[0] === '.') el.classList.add(part.slice(1));
      else el.id = part.slice(1);
    }
  }
  if (attrs && (typeof attrs !== 'object' || attrs instanceof Node || Array.isArray(attrs))) {
    children.unshift(attrs);
    attrs = null;
  }
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === false || value == null) continue;
    if (key.startsWith('on') && typeof value === 'function') el.addEventListener(key.slice(2), value);
    else if (key === 'class') el.className += (el.className ? ' ' : '') + value;
    else if (key === 'dataset') Object.assign(el.dataset, value);
    else if (['value', 'checked', 'disabled', 'hidden', 'readOnly'].includes(key)) el[key] = value;
    else el.setAttribute(key, value === true ? '' : String(value));
  }
  append(el, children);
  return el;
}

export function append(el, children) {
  for (const child of children.flat(Infinity)) {
    if (child == null || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}

export function clear(el) {
  el.textContent = '';
  return el;
}

/** <svg><use> from the admin icon sprite (web/admin-icons.svg), drawn with currentColor. */
export function icon(name, cls = '') {
  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('class', `a-ico ${cls}`.trim());
  svg.setAttribute('aria-hidden', 'true');
  const use = document.createElementNS(SVG_NS, 'use');
  use.setAttribute('href', `admin-icons.svg#i-${name}`);
  svg.append(use);
  return svg;
}

// ---------------------------------------------------------------- toasts

export function toast(message, kind = 'ok', ms = 4200) {
  const host = document.getElementById('a-toasts');
  if (!host) return;
  const el = h(`div.a-toast.${kind}`, icon(kind === 'ok' ? 'check' : 'warn'), h('span', message));
  host.append(el);
  requestAnimationFrame(() => el.classList.add('in'));
  setTimeout(() => {
    el.classList.remove('in');
    setTimeout(() => el.remove(), 260);
  }, ms);
}

// ---------------------------------------------------------------- confirm dialog

/**
 * Modal confirmation (native <dialog>: focus trap and Esc for free).
 * confirm({title, body: string|Node, confirmText, cancelText, danger}) -> Promise<boolean>
 */
export function confirm({ title, body, confirmText = 'Potwierdź', cancelText = 'Anuluj', danger = false }) {
  return new Promise((resolve) => {
    const dialog = h('dialog.a-dialog', { 'aria-labelledby': 'a-dialog-title' });
    const ok = h(`button.btn.${danger ? 'a-danger' : 'primary'}`, { type: 'button' }, confirmText);
    const cancel = h('button.btn', { type: 'button' }, cancelText);
    append(dialog, [
      h('h2#a-dialog-title', title),
      body ? h('div.a-dialog-body', typeof body === 'string' ? h('p', body) : body) : null,
      h('div.a-dialog-actions', cancel, ok),
    ]);
    let result = false;
    ok.addEventListener('click', () => { result = true; dialog.close(); });
    cancel.addEventListener('click', () => dialog.close());
    dialog.addEventListener('close', () => { dialog.remove(); resolve(result); });
    dialog.addEventListener('click', (e) => { if (e.target === dialog) dialog.close(); });
    document.body.append(dialog);
    dialog.showModal();
    (danger ? cancel : ok).focus();
  });
}

// ---------------------------------------------------------------- page pieces

export function pageHead(title, lede, ...actions) {
  return h('header.a-page-head',
    h('div', h('h1', title), lede ? h('p.a-lede', lede) : null),
    actions.length ? h('div.a-page-actions', actions) : null);
}

export function emptyState(iconName, text, hint) {
  return h('div.a-empty', icon(iconName), h('p', text), hint ? h('p.a-muted', hint) : null);
}

export function errorState(message, retry) {
  return h('div.a-empty.is-error', icon('warn'), h('p', message),
    retry ? h('button.btn', { type: 'button', onclick: retry }, 'Spróbuj ponownie') : null);
}

// ---------------------------------------------------------------- formatting

const pad = (n) => String(n).padStart(2, '0');

export function formatClock(date) {
  return `${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

/** "5 min temu" / "za 12 min"; far dates fall back to an absolute date. */
export function relative(ts) {
  const diff = (ts - Date.now()) / 1000;
  const abs = Math.abs(diff);
  let text;
  if (abs < 45) return 'teraz';
  if (abs < 3600) text = `${Math.round(abs / 60)} min`;
  else if (abs < 86400) text = `${Math.floor(abs / 3600)} godz. ${Math.round((abs % 3600) / 60)} min`;
  else return new Date(ts).toLocaleString('pl-PL', { dateStyle: 'medium', timeStyle: 'short' });
  return diff < 0 ? `${text} temu` : `za ${text}`;
}

export function fmtMs(value) {
  return value == null ? '–' : `${Math.round(value)} ms`;
}

/** Lowercase, no diacritics: for search. */
export function normalize(text) {
  return String(text ?? '')
    .toLowerCase()
    .replace(/ł/g, 'l')
    .normalize('NFD')
    .replace(/[̀-ͯ]/g, '');
}
