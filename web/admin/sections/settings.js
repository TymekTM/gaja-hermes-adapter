// Ustawienia: a form generated from GET /api/admin/config, saved with PUT /api/admin/config.

import { h, icon, clear, toast, confirm, normalize, pageHead, errorState } from '../ui.js';
import { createControl, deepEqual, describeValue } from '../fields.js';
import { SECTION_ORDER, sectionTitle, fieldLabel } from '../labels.js';
import { restartAdapter } from '../restart.js';

// Survives re-renders (saving reloads the form): which section cards are open and the search text.
const openSections = new Set();
let lastQuery = '';

/** "1 validation error for Settings\nvad.min_silence_ms\n  Input should be..." -> { path: message }. */
export function parseValidation(text) {
  const byPath = {};
  let current = null;
  for (const raw of String(text).split('\n')) {
    if (!raw.trim() || /^\d+ validation errors? for /.test(raw)) continue;
    if (/^\s*For further information/.test(raw)) continue;
    if (!/^\s/.test(raw)) {
      current = raw.trim();
      continue;
    }
    if (current && !(current in byPath)) byPath[current] = raw.trim().replace(/\s*\[type=.*$/, '');
  }
  return byPath;
}

export default {
  id: 'ustawienia',
  title: 'Ustawienia',
  icon: 'sliders',
  render(container, api) {
    const view = new SettingsView(container, api);
    view.load();
    return () => view.dispose();
  },
};

class SettingsView {
  constructor(container, api) {
    this.container = container;
    this.api = api;
    this.rows = new Map(); // path -> row state
    this.cards = new Map(); // section -> { details, ... }
    this.config = null;
    this.saving = false;
    this.serverErrors = new Map();
    this.barError = null;
    this.beforeUnload = (e) => {
      if (this.changes().size) { e.preventDefault(); e.returnValue = ''; }
    };
    window.addEventListener('beforeunload', this.beforeUnload);
    api.setLeaveGuard(async () => {
      if (!this.changes().size) return true;
      return confirm({
        title: 'Odrzucić niezapisane zmiany?',
        body: `Masz ${this.changes().size} niezapisanych zmian w ustawieniach. Wyjście z tej sekcji je odrzuci.`,
        confirmText: 'Odrzuć i wyjdź',
        danger: true,
      });
    });
  }

  dispose() {
    window.removeEventListener('beforeunload', this.beforeUnload);
    this.api.setLeaveGuard(null);
  }

  // ------------------------------------------------------------ data

  async load({ keepScroll = false } = {}) {
    const scroller = this.container.closest('.a-main');
    const top = keepScroll && scroller ? scroller.scrollTop : 0;
    try {
      this.config = await this.api.get('/api/admin/config');
    } catch (err) {
      if (err.status === 401) return;
      clear(this.container).append(errorState(`Nie udało się wczytać ustawień: ${err.message}`, () => this.load()));
      return;
    }
    this.build();
    if (scroller) scroller.scrollTop = top;
  }

  // ------------------------------------------------------------ view

  build() {
    this.rows.clear();
    this.cards.clear();
    this.serverErrors.clear();
    this.barError = null;
    const { fields, sections } = this.config;

    this.search = h('input.a-input.a-search-input', { type: 'search', placeholder: 'Szukaj ustawienia, np. cisza, port, tts…', 'aria-label': 'Szukaj ustawień', autocomplete: 'off', value: lastQuery });
    this.search.addEventListener('input', () => { lastQuery = this.search.value; this.applyFilter(); });
    this.expandBtn = h('button.btn.a-sm', { type: 'button', onclick: () => this.toggleAll() }, 'Rozwiń wszystko');
    this.banner = h('div.a-banner-slot');
    this.list = h('div.a-cards');
    this.empty = h('div.a-empty', { hidden: true }, icon('search'), h('p', 'Żadne ustawienie nie pasuje do wyszukiwania.'));
    this.saveCount = h('strong.a-tabular');
    this.saveMsg = h('div.a-savebar-msg');
    this.discardBtn = h('button.btn', { type: 'button', onclick: () => this.discard() }, 'Odrzuć');
    this.saveBtn = h('button.btn.primary', { type: 'button', onclick: () => this.save() }, 'Zapisz');
    this.saveBar = h('div.a-savebar', { hidden: true, role: 'region', 'aria-label': 'Zapis zmian' },
      h('div.a-savebar-info', h('div.a-savebar-count', this.saveCount), this.saveMsg),
      h('div.a-savebar-actions', this.discardBtn, this.saveBtn));

    const bySection = new Map();
    for (const field of fields) {
      if (!bySection.has(field.section)) bySection.set(field.section, []);
      bySection.get(field.section).push(field);
    }
    const order = (s) => { const i = SECTION_ORDER.indexOf(s); return i < 0 ? 999 : i; };
    const names = [...bySection.keys()].sort((a, b) => order(a) - order(b) || a.localeCompare(b));
    for (const name of names) this.list.append(this.buildCard(name, bySection.get(name), sections[name] || ''));

    clear(this.container).append(
      pageHead('Ustawienia', 'Zmiany trafiają do config.local.toml. Część ustawień działa dopiero po restarcie adaptera.'),
      h('div.a-toolbar',
        h('label.a-search', icon('search'), this.search),
        this.expandBtn),
      this.banner,
      this.list,
      this.empty,
      this.saveBar,
    );
    this.renderBanner();
    this.applyFilter();
    this.refreshAll();
  }

  buildCard(name, fields, help) {
    const modified = fields.filter((f) => f.overridden).length;
    const details = h('details.a-card', { dataset: { section: name } });
    details.open = openSections.has(name);
    const badges = h('span.a-card-badges');
    const summary = h('summary.a-card-summary',
      h('span.a-card-title', h('strong', sectionTitle(name)), h('code.a-path', name)),
      badges,
      h('span.a-card-count', `${fields.length}`),
      icon('chevron', 'a-card-chev'));
    details.addEventListener('toggle', () => {
      if (this.filtering) return;
      if (details.open) openSections.add(name); else openSections.delete(name);
      this.syncExpandBtn();
    });
    const body = h('div.a-card-body', help ? h('p.a-card-help', help) : null);
    for (const field of fields) body.append(this.buildRow(field));
    details.append(summary, body);
    this.cards.set(name, { details, badges, fields, modified });
    return details;
  }

  buildRow(field) {
    const labelId = `lbl-${field.path.replace(/\./g, '-')}`;
    const row = { field, mode: 'edit', error: null };
    const label = h('label.a-field-label', { id: labelId }, fieldLabel(field.path));
    const badges = h('span.a-field-badges');
    const errorEl = h('p.a-field-error', { role: 'alert', hidden: true });
    const meta = h('div.a-field-meta');
    const control = createControl(field, { labelId, onInput: () => this.onEdit(field.path) });
    if (field.read_only) control.setDisabled(true);
    if (control.inputId) label.htmlFor = control.inputId;

    const el = h('div.a-field',
      { dataset: { path: field.path, kind: field.kind } },
      h('div.a-field-text',
        h('div.a-field-title', label, badges),
        h('code.a-path', field.path),
        field.help ? h('p.a-field-help', field.help) : null),
      h('div.a-field-ctl', control.el, meta, errorEl));
    el.classList.toggle('is-ro', field.read_only);
    Object.assign(row, { el, control, badges, meta, errorEl, labelText: normalize(`${fieldLabel(field.path)} ${field.path} ${field.help} ${sectionTitle(field.section)}`) });
    this.rows.set(field.path, row);
    return el;
  }

  // ------------------------------------------------------------ change tracking

  onEdit(path) {
    const row = this.rows.get(path);
    if (row.mode === 'reset') row.mode = 'edit';
    this.serverErrors.delete(path);
    this.barError = null;
    this.refreshRow(row);
    this.refreshBar();
  }

  /** Map path -> new value (null = back to default). Invalid inputs are excluded and flagged. */
  changes() {
    const out = new Map();
    for (const row of this.rows.values()) {
      const c = this.rowChange(row);
      if (c.change) out.set(row.field.path, c.value);
    }
    return out;
  }

  rowChange(row) {
    const { field } = row;
    if (field.read_only) return {};
    if (row.mode === 'reset') return { change: true, value: null };
    const got = row.control.get();
    if (got.error) return { error: got.error };
    if (got.value === null) {
      // Clearing an optional field drops the override; nothing to do when it has none.
      return field.overridden ? { change: true, value: null } : {};
    }
    return deepEqual(got.value, field.value) ? {} : { change: true, value: got.value };
  }

  refreshAll() {
    for (const row of this.rows.values()) this.refreshRow(row);
    this.refreshBar();
  }

  refreshRow(row) {
    const { field } = row;
    const c = this.rowChange(row);
    const err = c.error || this.serverErrors.get(field.path) || null;
    row.el.classList.toggle('is-dirty', !!c.change);
    row.el.classList.toggle('is-error', !!err);
    row.errorEl.hidden = !err;
    row.errorEl.textContent = err || '';
    row.invalid = !!c.error;

    clear(row.badges);
    if (field.read_only) row.badges.append(h('span.a-badge', icon('lock'), 'tylko do odczytu'));
    if (field.overridden && row.mode !== 'reset') row.badges.append(h('span.a-badge.accent', 'zmienione lokalnie'));
    if (field.restart_pending) row.badges.append(h('span.a-badge.warn', { title: `Działa: ${describeValue(field, field.running)}` }, 'czeka na restart'));
    if (field.danger) row.badges.append(h('span.a-badge.danger', 'ostrożnie'));

    clear(row.meta);
    if (field.read_only) return;
    row.meta.append(h('span.a-default', 'domyślnie: ', h('b', describeValue(field, field.default))));
    if (row.mode === 'reset') {
      row.meta.append(h('span.a-reset-note', 'po zapisaniu wróci wartość domyślna'),
        h('button.a-link', { type: 'button', onclick: () => this.undoReset(row) }, 'cofnij'));
    } else if (field.overridden) {
      row.meta.append(h('button.a-link', { type: 'button', onclick: () => this.reset(row) }, icon('undo'), 'przywróć domyślne'));
    }
  }

  reset(row) {
    row.mode = 'reset';
    row.control.set(row.field.default);
    this.serverErrors.delete(row.field.path);
    this.refreshRow(row);
    this.refreshBar();
  }

  undoReset(row) {
    row.mode = 'edit';
    row.control.set(row.field.value);
    this.refreshRow(row);
    this.refreshBar();
  }

  refreshBar() {
    const changes = this.changes();
    const invalid = [...this.rows.values()].filter((r) => r.invalid).length;
    const n = changes.size;
    this.saveBar.hidden = n === 0 && !this.barError && invalid === 0;
    this.saveCount.textContent = n ? plural(n, 'niezapisana zmiana', 'niezapisane zmiany', 'niezapisanych zmian') : 'Brak zmian do zapisu';
    clear(this.saveMsg);
    if (invalid) this.saveMsg.append(h('span.a-bar-error', `Popraw ${plural(invalid, 'pole', 'pola', 'pól')} z błędem, żeby zapisać.`));
    if (this.barError) {
      this.saveMsg.append(h('span.a-bar-error', this.barError.summary));
      if (this.barError.raw) this.saveMsg.append(h('details.a-bar-raw', h('summary', 'szczegóły'), h('pre', this.barError.raw)));
    }
    this.saveBtn.disabled = this.saving || n === 0 || invalid > 0;
    this.discardBtn.disabled = this.saving || n === 0;
    this.saveBtn.textContent = this.saving ? 'Zapisuję…' : 'Zapisz';
    for (const card of this.cards.values()) {
      const dirty = card.fields.filter((f) => changes.has(f.path)).length;
      clear(card.badges);
      if (dirty) card.badges.append(h('span.a-badge.accent', `${dirty} do zapisu`));
      const pending = card.fields.filter((f) => f.restart_pending).length;
      if (pending) card.badges.append(h('span.a-badge.warn', 'restart'));
      const modified = card.fields.filter((f) => f.overridden).length;
      if (modified && !dirty) card.badges.append(h('span.a-badge', `${modified} lokalnie`));
      card.details.classList.toggle('has-dirty', dirty > 0);
    }
  }

  discard() {
    for (const row of this.rows.values()) {
      row.mode = 'edit';
      row.control.set(row.field.value);
    }
    this.serverErrors.clear();
    this.barError = null;
    this.refreshAll();
  }

  // ------------------------------------------------------------ saving

  async save() {
    const changes = this.changes();
    if (!changes.size || this.saving) return;
    const danger = [...changes.keys()].filter((p) => this.rows.get(p).field.danger);
    if (danger.length) {
      const list = h('ul.a-confirm-list', danger.map((p) => {
        const f = this.rows.get(p).field;
        return h('li', h('b', fieldLabel(p)), h('code.a-path', p), h('span', 'nowa wartość: ', h('em', changes.get(p) === null ? 'domyślna' : describeValue(f, changes.get(p)))));
      }));
      const ok = await confirm({
        title: 'Zmiana wrażliwych ustawień',
        body: h('div', h('p', 'Błędna wartość może odciąć dostęp do panelu lub uniemożliwić start adaptera. Na pewno zapisać?'), list),
        confirmText: 'Zapisz mimo to',
        danger: true,
      });
      if (!ok) return;
    }
    this.saving = true;
    this.barError = null;
    this.refreshBar();
    try {
      await this.api.put('/api/admin/config', { values: Object.fromEntries(changes) });
    } catch (err) {
      this.saving = false;
      if (err.status === 401) return;
      this.showSaveError(err, changes);
      return;
    }
    this.saving = false;
    toast(`Zapisano ${plural(changes.size, 'zmianę', 'zmiany', 'zmian')}.`);
    await this.load({ keepScroll: true });
  }

  showSaveError(err, changes) {
    if (err.status !== 400) {
      this.barError = { summary: `Nie zapisano: ${err.message}` };
      this.refreshBar();
      return;
    }
    const parsed = parseValidation(err.message);
    let matched = 0;
    for (const [path, message] of Object.entries(parsed)) {
      if (changes.has(path)) {
        this.serverErrors.set(path, message);
        matched += 1;
      }
    }
    const first = [...this.serverErrors.keys()][0];
    this.barError = matched
      ? { summary: `Nie zapisano. Serwer odrzucił ${plural(matched, 'pole', 'pola', 'pól')}; nic nie zostało zmienione.` }
      : { summary: 'Nie zapisano. Serwer odrzucił zmiany; nic nie zostało zmienione.', raw: err.message };
    this.refreshAll();
    if (first) this.revealRow(first);
  }

  revealRow(path) {
    const row = this.rows.get(path);
    const card = this.cards.get(row.field.section);
    if (card && !card.details.open) { card.details.open = true; openSections.add(row.field.section); }
    row.el.scrollIntoView({ block: 'center', behavior: 'smooth' });
  }

  // ------------------------------------------------------------ restart banner

  renderBanner() {
    clear(this.banner);
    if (!this.config.restart_pending) return;
    const pending = this.config.fields.filter((f) => f.restart_pending);
    const status = h('p.a-banner-status', { role: 'status', hidden: true });
    const btn = h('button.btn.primary', { type: 'button' }, icon('power'), 'Zrestartuj adapter');
    btn.addEventListener('click', async () => {
      btn.disabled = true;
      status.hidden = true;
      const result = await restartAdapter(this.api);
      btn.disabled = false;
      if (result.cancelled) return;
      if (result.ok) {
        toast('Adapter wrócił po restarcie.');
        await this.load({ keepScroll: true });
        return;
      }
      status.hidden = false;
      status.textContent = result.message;
    });
    this.banner.append(h('div.a-banner.warn',
      icon('warn'),
      h('div.a-banner-text',
        h('strong', 'Zmiany wymagają restartu'),
        h('p', `Zapisane, ale jeszcze nie używane przez adapter: ${pending.map((f) => fieldLabel(f.path)).join(', ')}.`),
        status),
      btn));
  }

  // ------------------------------------------------------------ search / expand

  applyFilter() {
    const q = normalize(this.search.value.trim());
    this.filtering = q.length > 0;
    let any = false;
    for (const [name, card] of this.cards) {
      let shown = 0;
      for (const f of card.fields) {
        const row = this.rows.get(f.path);
        const match = !q || q.split(/\s+/).every((word) => row.labelText.includes(word));
        row.el.hidden = !match;
        if (match) shown += 1;
      }
      card.details.hidden = shown === 0;
      if (q) card.details.open = shown > 0;
      else card.details.open = openSections.has(name);
      if (shown) any = true;
    }
    this.empty.hidden = any;
    this.syncExpandBtn();
  }

  syncExpandBtn() {
    this.expandBtn.hidden = this.filtering;
    const allOpen = [...this.cards.values()].every((c) => c.details.open);
    this.expandBtn.textContent = allOpen ? 'Zwiń wszystko' : 'Rozwiń wszystko';
  }

  toggleAll() {
    const open = ![...this.cards.values()].every((c) => c.details.open);
    for (const [name, card] of this.cards) {
      card.details.open = open;
      if (open) openSections.add(name); else openSections.delete(name);
    }
    this.syncExpandBtn();
  }
}

export function plural(n, one, few, many) {
  const word = n === 1 ? one : n % 10 >= 2 && n % 10 <= 4 && !(n % 100 >= 12 && n % 100 <= 14) ? few : many;
  return `${n} ${word}`;
}
