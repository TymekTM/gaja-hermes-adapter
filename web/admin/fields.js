// Form controls for the settings editor, one per config field kind.
//
// createControl(field, ctx) -> { el, inputId, get(), set(value), disabled }
//   get() -> { value } for a valid input, or { error: 'text' }.
//   ctx.labelId: id of the element that labels the control; ctx.onInput(): called after every user edit.

import { h, icon } from './ui.js';
import { choiceLabel } from './labels.js';

let uid = 0;
const nextId = (prefix) => `${prefix}-${++uid}`;

export function deepEqual(a, b) {
  if (a === b) return true;
  if (a == null || b == null || typeof a !== 'object' || typeof b !== 'object') return false;
  if (Array.isArray(a) !== Array.isArray(b)) return false;
  const ka = Object.keys(a);
  const kb = Object.keys(b);
  return ka.length === kb.length && ka.every((k) => deepEqual(a[k], b[k]));
}

/** A short human text for a value (defaults, confirmations). */
export function describeValue(field, value) {
  if (value === null || value === undefined) return 'brak';
  switch (field.kind) {
    case 'bool':
      return value ? 'włączone' : 'wyłączone';
    case 'list':
      return value.length ? value.join(', ') : '(pusta lista)';
    case 'map': {
      const entries = Object.entries(value);
      return entries.length ? entries.map(([k, v]) => `${k} = ${v}`).join(', ') : '(pusta mapa)';
    }
    case 'choice':
      return choiceLabel(field.path, value);
    default:
      return value === '' ? '(puste)' : String(value);
  }
}

export function createControl(field, ctx) {
  const builders = { bool: boolControl, int: numberControl, float: numberControl, choice: choiceControl, list: listControl, map: mapControl };
  const build = builders[field.kind] || textControl;
  const control = build(field, ctx);
  control.set(field.value);
  return control;
}

function boolControl(field, ctx) {
  const input = h('input', { type: 'checkbox', id: nextId('f'), 'aria-labelledby': ctx.labelId });
  const el = h('label.switch', input, h('span.switch-track'));
  input.addEventListener('change', () => ctx.onInput());
  return {
    el,
    inputId: input.id,
    get: () => ({ value: input.checked }),
    set: (v) => { input.checked = !!v; },
    setDisabled: (d) => { input.disabled = d; },
  };
}

function numberControl(field, ctx) {
  const isInt = field.kind === 'int';
  const input = h('input.a-input.a-num', {
    type: 'number',
    id: nextId('f'),
    step: isInt ? '1' : 'any',
    inputmode: isInt ? 'numeric' : 'decimal',
    placeholder: field.optional ? 'brak' : null,
    'aria-labelledby': ctx.labelId,
  });
  input.addEventListener('input', () => ctx.onInput());
  return {
    el: input,
    inputId: input.id,
    get() {
      const raw = input.value.trim();
      if (raw === '') {
        if (input.validity.badInput) return { error: 'To nie jest liczba.' };
        return field.optional ? { value: null } : { error: 'Podaj liczbę.' };
      }
      const n = Number(raw);
      if (!Number.isFinite(n)) return { error: 'To nie jest liczba.' };
      if (isInt && !Number.isInteger(n)) return { error: 'Wymagana liczba całkowita.' };
      return { value: n };
    },
    set: (v) => { input.value = v == null ? '' : String(v); },
    setDisabled: (d) => { input.disabled = d; },
  };
}

function textControl(field, ctx) {
  const input = h('input.a-input', {
    type: 'text',
    id: nextId('f'),
    class: field.kind === 'path' ? 'a-mono' : null,
    placeholder: field.optional ? 'brak' : null,
    autocomplete: 'off',
    autocapitalize: 'off',
    spellcheck: 'false',
    'aria-labelledby': ctx.labelId,
  });
  input.addEventListener('input', () => ctx.onInput());
  return {
    el: input,
    inputId: input.id,
    get() {
      if (input.value === '' && field.optional) return { value: null };
      return { value: input.value };
    },
    set: (v) => { input.value = v == null ? '' : String(v); },
    setDisabled: (d) => { input.disabled = d; },
  };
}

function choiceControl(field, ctx) {
  const select = h('select.a-input.a-select', { id: nextId('f'), 'aria-labelledby': ctx.labelId });
  if (field.optional) select.append(h('option', { value: '' }, '— brak —'));
  for (const choice of field.choices || []) select.append(h('option', { value: String(choice) }, choiceLabel(field.path, choice)));
  select.addEventListener('change', () => ctx.onInput());
  const wrap = h('div.a-select-wrap', select, icon('chevron', 'a-select-chev'));
  return {
    el: wrap,
    inputId: select.id,
    get() {
      if (select.value === '') return { value: field.optional ? null : '' };
      const match = (field.choices || []).find((c) => String(c) === select.value);
      return { value: match !== undefined ? match : select.value };
    },
    set(v) {
      select.value = v == null ? '' : String(v);
      // A value outside the known choices (hand-edited file): show it instead of silently picking another.
      if (v != null && select.value !== String(v)) {
        select.append(h('option', { value: String(v) }, String(v)));
        select.value = String(v);
      }
    },
    setDisabled: (d) => { select.disabled = d; },
  };
}

function listControl(field, ctx) {
  const area = h('textarea.a-input.a-area.a-mono', {
    id: nextId('f'),
    rows: 3,
    autocomplete: 'off',
    autocapitalize: 'off',
    spellcheck: 'false',
    placeholder: 'jedna pozycja w wierszu',
    'aria-labelledby': ctx.labelId,
  });
  const fit = () => {
    area.rows = Math.max(2, Math.min(10, area.value.split('\n').length));
  };
  area.addEventListener('input', () => { fit(); ctx.onInput(); });
  return {
    el: area,
    inputId: area.id,
    get: () => ({ value: area.value.split('\n').map((s) => s.trim()).filter(Boolean) }),
    set: (v) => { area.value = (v || []).join('\n'); fit(); },
    setDisabled: (d) => { area.disabled = d; },
  };
}

function mapControl(field, ctx) {
  const rows = h('div.a-map-rows');
  const addBtn = h('button.btn.a-sm.a-map-add', { type: 'button' }, icon('plus'), 'Dodaj wiersz');
  const el = h('div.a-map', { role: 'group', 'aria-labelledby': ctx.labelId }, rows, addBtn);
  let disabled = false;

  function addRow(key = '', value = '', focus = false) {
    const k = h('input.a-input.a-mono', { type: 'text', placeholder: 'klucz', 'aria-label': 'Klucz', autocomplete: 'off', autocapitalize: 'off', spellcheck: 'false' });
    const v = h('input.a-input', { type: 'text', placeholder: 'wartość', 'aria-label': 'Wartość', autocomplete: 'off' });
    k.value = key;
    v.value = value;
    const del = h('button.icon-btn.small.a-map-del', { type: 'button', 'aria-label': 'Usuń wiersz', title: 'Usuń wiersz' }, icon('close'));
    const row = h('div.a-map-row', k, h('span.a-map-eq', { 'aria-hidden': 'true' }, '='), v, del);
    for (const input of [k, v]) { input.disabled = disabled; input.addEventListener('input', () => ctx.onInput()); }
    del.disabled = disabled;
    del.addEventListener('click', () => { row.remove(); ctx.onInput(); });
    rows.append(row);
    if (focus) k.focus();
  }

  addBtn.addEventListener('click', () => { addRow('', '', true); ctx.onInput(); });
  return {
    el,
    inputId: null,
    get() {
      const out = {};
      for (const row of rows.children) {
        const [k, v] = row.querySelectorAll('input');
        const key = k.value.trim();
        if (!key) {
          if (v.value.trim()) return { error: 'Wiersz z wartością musi mieć klucz.' };
          continue;
        }
        if (key in out) return { error: `Klucz „${key}” występuje dwa razy.` };
        out[key] = v.value;
      }
      return { value: out };
    },
    set(v) {
      rows.textContent = '';
      for (const [key, value] of Object.entries(v || {})) addRow(key, String(value));
    },
    setDisabled(d) {
      disabled = d;
      for (const node of el.querySelectorAll('input, button')) node.disabled = d;
    },
  };
}
