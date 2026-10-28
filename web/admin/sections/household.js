// Domownicy: people, their Hermes profiles, voices and bound phones/devices
// (edited as a whole document through GET/PUT /api/admin/household).

import { h, icon, clear, toast, confirm, pageHead, emptyState, errorState } from '../ui.js';
import { HOUSEHOLD_LABELS as L } from '../labels.js';

// Household person ids: ^[a-z0-9][a-z0-9_-]{0,63}$ (gaja/household.py ID_RE).
const ID_RE = /^[a-z0-9][a-z0-9_-]{0,63}$/;

function slugify(name) {
  return name.toLowerCase().replace(/ł/g, 'l').normalize('NFD').replace(/[̀-ͯ]/g, '')
    .replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 64);
}

export default {
  id: 'domownicy',
  title: L.title,
  icon: 'home',
  render(container, api) {
    const view = new HouseholdView(container, api);
    view.load();
    return () => view.dispose();
  },
};

class HouseholdView {
  constructor(container, api) {
    this.container = container;
    this.api = api;
    this.doc = null;      // household as the server last returned it
    this.draft = null;    // the edited copy
    this.voices = [];
    this.languages = [];
    this.registryDevices = [];
    this.idEdited = new WeakSet(); // people whose id was typed by hand (by object: the id itself changes)
    this.saving = false;
    this.saveError = null;
    this.disposed = false;
    this.beforeUnload = (e) => {
      if (this.dirty()) { e.preventDefault(); e.returnValue = ''; }
    };
    window.addEventListener('beforeunload', this.beforeUnload);
    api.setLeaveGuard(async () => {
      if (!this.dirty()) return true;
      return confirm({ title: L.leaveTitle, body: L.leaveBody, confirmText: L.leaveConfirm, danger: true });
    });
  }

  dispose() {
    this.disposed = true;
    window.removeEventListener('beforeunload', this.beforeUnload);
    this.api.setLeaveGuard(null);
  }

  async load() {
    const [household, voices, devices] = await Promise.allSettled([
      this.api.get('/api/admin/household'),
      this.api.get('/api/admin/tts/voices'),
      this.api.get('/api/devices'),
    ]);
    if (this.disposed) return;
    if (household.status === 'rejected') {
      const err = household.reason;
      if (err.status === 401) return;
      const unavailable = err.status === 503;
      clear(this.container).append(unavailable
        ? emptyState('home', L.unavailable)
        : errorState(`${L.loadError}: ${err.message}`, () => this.load()));
      return;
    }
    this.doc = household.value.household;
    this.draft = structuredClone(this.doc);
    this.profiles = this.doc.profiles;
    if (voices.status === 'fulfilled') {
      this.voices = voices.value.voices;
      this.languages = Object.keys(voices.value.engine_by_language || {}); // languages Gaja answers in
    }
    if (devices.status === 'fulfilled') this.registryDevices = devices.value.devices || [];
    this.saveError = null;
    this.build();
  }

  /** Reload from the server; unsaved edits are dropped only after the same confirm as leaving. */
  async refresh() {
    if (this.dirty()) {
      const ok = await confirm({ title: L.leaveTitle, body: L.refreshBody, confirmText: L.refreshConfirm, danger: true });
      if (!ok || this.disposed) return;
    }
    await this.load();
  }

  dirty() {
    return this.draft !== null && JSON.stringify(this.draft) !== JSON.stringify(this.doc);
  }

  // ------------------------------------------------------------ view

  build() {
    this.banner = h('div.a-banner-slot');
    this.peopleList = h('ul.a-people');
    this.generalBox = h('section.a-block');
    this.saveCount = h('strong.a-tabular');
    this.saveMsg = h('div.a-savebar-msg');
    this.discardBtn = h('button.btn', { type: 'button', onclick: () => this.discard() }, L.discard);
    this.saveBtn = h('button.btn.primary', { type: 'button', onclick: () => this.save() }, L.save);
    this.saveBar = h('div.a-savebar', { hidden: true, role: 'region', 'aria-label': 'Zapis zmian' },
      h('div.a-savebar-info', h('div.a-savebar-count', this.saveCount), this.saveMsg),
      h('div.a-savebar-actions', this.discardBtn, this.saveBtn));

    clear(this.container).append(
      pageHead(L.title, L.lede,
        h('button.btn.a-sm', { type: 'button', onclick: () => this.addPerson() }, icon('plus'), L.addPerson),
        h('button.btn.a-sm', { type: 'button', onclick: () => this.refresh() }, icon('refresh'), L.refresh)),
      this.banner,
      this.peopleList,
      this.generalBox,
      this.saveBar);
    this.draw();
  }

  draw() {
    this.drawGeneral();
    this.drawPeople();
    this.drawBanner();
    this.refreshBar();
  }

  field(labelText, control, help, ariaLabel) {
    const target = control.querySelector('select, input');
    if (target) target.setAttribute('aria-label', ariaLabel || labelText);
    return h('div.a-person-field', h('span', labelText), control, help ? h('p.a-muted', help) : null);
  }

  /**
   * `languages`: what the voice must speak (the person's language, or every language Gaja answers
   * in). A voice that covers only some of them is marked "tylko EN": other turns fall back to the
   * default voice. A voice the adapter has not built yet gets a hint below the select.
   */
  voiceSelect(value, onchange, languages) {
    const isPartial = (v) => v.languages.length > 0 && languages.some((lang) => !v.languages.includes(lang));
    const options = [
      h('option', { value: '' }, L.voiceDefault),
      ...this.voices.map((v) => {
        const partial = isPartial(v);
        const text = partial ? `${v.label} — ${L.voiceOnly(v.languages)}`
          : `${v.label}${v.languages.length ? ` (${v.languages.join(', ')})` : ''}`;
        const option = h('option', { value: v.id }, text);
        if (partial) option.setAttribute('data-partial', '');
        return option;
      }),
    ];
    if (value && !this.voices.some((v) => v.id === value)) {
      options.push(h('option', { value }, value)); // a voice the server no longer lists: keep it visible
    }
    const hint = h('p.a-muted.a-voice-hint');
    const showHint = (id) => {
      // The closed select truncates long labels, so the selected voice's limits are repeated here.
      const voice = this.voices.find((v) => v.id === id);
      hint.textContent = voice ? [
        isPartial(voice) ? L.voicePartial(voice.languages) : '',
        voice.loaded ? '' : L.voiceNotLoaded,
      ].filter(Boolean).join(' ') : '';
      hint.hidden = !hint.textContent;
    };
    const select = h('select.a-input.a-select', {
      onchange: (e) => { showHint(e.target.value); onchange(e.target.value || null); },
    }, options);
    select.value = value || '';
    showHint(select.value);
    return h('div.a-voice', h('div.a-select-wrap', select, icon('chevron', 'a-select-chev')), hint);
  }

  /** Languages a person's voice must speak: their own, else every language Gaja answers in. */
  personLanguages(person) {
    return person.language ? [person.language] : this.languages;
  }

  /** `safeOnly`: list only non-sensitive profiles (plus the current value, so the select never lies). */
  profileSelect(value, onchange, { safeOnly = false } = {}) {
    const listed = safeOnly ? this.profiles.filter((p) => !p.sensitive || p.id === value) : this.profiles;
    const options = listed.map((p) => {
      const option = h('option', { value: p.id }, `${p.label}${p.sensitive ? ` · ${L.sensitiveBadge}` : ''}`);
      if (p.sensitive) option.setAttribute('data-sensitive', '');
      return option;
    });
    const select = h('select.a-input.a-select', { onchange: (e) => onchange(e.target.value) }, options);
    select.value = value;
    return h('div.a-select-wrap', select, icon('chevron', 'a-select-chev'));
  }

  /** Options for a phone/device bind select: only people whose id is valid (a new person's id can be empty). */
  personOptions() {
    return this.draft.people.filter((p) => ID_RE.test(p.id)).map((p) => h('option', { value: p.id }, p.name || p.id));
  }

  drawGeneral() {
    const generalProfile = this.field(L.generalProfile,
      // Unknown voices use it, so a sensitive profile is never offered (the server rejects one too).
      this.profileSelect(this.draft.general_profile, (v) => { this.draft.general_profile = v; this.change(); }, { safeOnly: true }),
      L.generalHelp);
    const generalVoice = this.field(L.generalVoice,
      this.voiceSelect(this.draft.general_voice, (v) => { this.draft.general_voice = v; this.change(); }, this.languages));
    const knownDeviceIds = [...new Set([...Object.keys(this.draft.devices), ...this.registryDevices.map((d) => d.id)])];
    const unassignedPhones = Object.keys(this.draft.clients || {}).filter((cid) => !this.draft.clients[cid].person);

    const deviceRow = (deviceId) => {
      const registered = this.registryDevices.find((d) => d.id === deviceId);
      const binding = this.draft.devices[deviceId] || null;
      const select = h('select.a-input.a-select', { 'aria-label': `${L.devices}: ${deviceId}`, onchange: (e) => this.bindDevice(deviceId, e.target.value || null) },
        h('option', { value: '' }, L.unassignedDevice),
        ...this.personOptions());
      select.value = binding && binding.person ? binding.person : '';
      return h('li.a-row',
        h('div.a-row-main',
          h('p.a-row-title', registered && registered.name ? registered.name : deviceId),
          h('p.a-row-sub', h('code.a-path', deviceId), registered && registered.online ? ' · online' : '')),
        h('div.a-row-actions', h('div.a-select-wrap', select, icon('chevron', 'a-select-chev'))));
    };

    clear(this.generalBox).append(
      h('h2.a-h2', L.generalHead),
      h('div.a-form-grid', generalProfile, generalVoice),
      h('h2.a-h2', L.unassignedPhones, ' ', h('span.a-count', unassignedPhones.length ? String(unassignedPhones.length) : '')),
      this.phoneRows((cid) => !this.draft.clients[cid] || !this.draft.clients[cid].person),
      h('h2.a-h2', L.devices, ' ', h('span.a-count', knownDeviceIds.length ? String(knownDeviceIds.length) : '')),
      knownDeviceIds.length ? h('ul.a-rows', knownDeviceIds.map(deviceRow)) : emptyState('speaker', L.noDevices),
    );
  }

  /** Client rows for the filter; empty state when none match. */
  phoneRows(match) {
    const ids = Object.keys(this.draft.clients || {}).filter(match);
    if (!ids.length) return emptyState('note', L.noPhones);
    return h('ul.a-rows', ids.map((cid) => {
      const client = this.draft.clients[cid];
      const select = h('select.a-input.a-select', { 'aria-label': `${L.phones}: ${cid}`, onchange: (e) => this.bindClient(cid, e.target.value || null) },
        h('option', { value: '' }, L.unassignedDevice),
        ...this.personOptions());
      select.value = client.person || '';
      return h('li.a-row',
        h('div.a-row-main',
          h('p.a-row-title', client.label || cid.slice(0, 12)),
          h('p.a-row-sub', h('code.a-path', cid))),
        h('div.a-row-actions', h('div.a-select-wrap', select, icon('chevron', 'a-select-chev'))));
    }));
  }

  drawPeople() {
    clear(this.peopleList);
    if (!this.draft.people.length) {
      this.peopleList.append(h('li', emptyState('home', 'Nie ma jeszcze nikogo.', 'Dodaj domownika przyciskiem wyżej.')));
      return;
    }
    for (const person of this.draft.people) this.peopleList.append(this.personCard(person));
  }

  personCard(person) {
    const isNew = !this.doc.people.some((p) => p.id === person.id);
    const nameInput = h('input.a-input', {
      type: 'text', value: person.name, maxlength: 40, autocomplete: 'off',
      placeholder: L.newPersonName, 'aria-label': `${L.name}`,
      oninput: (e) => {
        person.name = e.target.value;
        if (isNew && !this.idEdited.has(person)) {
          this.renamePerson(person, slugify(e.target.value));
          idInput.value = person.id;
        }
        this.drawGeneral(); // bind selects list people by name
        this.change();
      },
    });
    const idInput = h('input.a-input', {
      type: 'text', value: person.id, maxlength: 64, autocomplete: 'off', pattern: '[a-z0-9][a-z0-9_-]*',
      'aria-label': L.personId, readonly: !isNew,
      oninput: (e) => {
        this.idEdited.add(person);
        this.renamePerson(person, e.target.value.trim());
        this.drawGeneral();
        this.change();
      },
    });
    const del = h('button.icon-btn.small.a-danger-ico', { type: 'button', 'aria-label': `${L.removePerson}: ${person.name}`, title: L.removePerson }, icon('trash'));
    del.addEventListener('click', () => this.removePerson(person));

    const profileChecks = h('div.a-profiles', this.profiles.map((profile) => {
      const checked = profile.id === person.default_profile || person.profiles.includes(profile.id);
      const input = h('input', { type: 'checkbox', checked, disabled: profile.id === person.default_profile });
      input.addEventListener('change', () => {
        if (input.checked) {
          if (!person.profiles.includes(profile.id)) person.profiles.push(profile.id);
        } else {
          person.profiles = person.profiles.filter((id) => id !== profile.id);
        }
        this.change();
      });
      const chip = h(`label.a-profile-check${profile.sensitive ? '.is-sensitive' : ''}`,
        { title: profile.sensitive ? L.sensitiveHint : profile.description || undefined },
        input,
        profile.sensitive ? icon('lock') : null,
        profile.label,
        profile.sensitive ? h('span.a-badge.warn', L.sensitiveBadge) : null,
        profile.id === person.default_profile ? h('span.a-badge', 'domyślny') : null);
      return chip;
    }));

    const phoneIds = Object.keys(this.draft.clients || {}).filter((cid) => this.draft.clients[cid].person === person.id);
    const deviceIds = Object.keys(this.draft.devices || {}).filter((did) => this.draft.devices[did].person === person.id);

    return h(`li.a-person${isNew ? '.is-new' : ''}`,
      h('div.a-person-head', nameInput, del),
      h('div.a-person-grid',
        this.field(L.personId, idInput, isNew ? L.idFromName : null, `${L.personId}: ${person.name || '?'}`),
        this.field(L.defaultProfile, this.profileSelect(person.default_profile, (v) => {
          // Person.allowed() already includes the default, so it is never pushed into `profiles`:
          // switching the default away must not quietly leave the old one in the extras.
          person.default_profile = v;
          this.redraw(); // the chips show the real checked/locked state right away
        }), null, `${L.defaultProfile}: ${person.name || '?'}`),
        this.field(L.voice, this.voiceSelect(person.tts, (v) => { person.tts = v; this.change(); }, this.personLanguages(person)), null, `${L.voice}: ${person.name || '?'}`)),
      h('div.a-person-field', h('span', L.extraProfiles), profileChecks),
      h('div.a-person-field', h('span', L.phones),
        phoneIds.length ? h('ul.a-rows', phoneIds.map((cid) => {
          const client = this.draft.clients[cid];
          const unbind = h('button.icon-btn.small', { type: 'button', 'aria-label': `${L.unbind}: ${cid}`, title: L.unbind }, icon('close'));
          unbind.addEventListener('click', () => this.bindClient(cid, null));
          return h('li.a-row',
            h('div.a-row-main', h('p.a-row-title', client.label || cid.slice(0, 12)), h('p.a-row-sub', h('code.a-path', cid))),
            h('div.a-row-actions', unbind));
        })) : h('p.a-muted', L.noPhones)),
      deviceIds.length ? h('div.a-person-field', h('span', L.devices),
        h('ul.a-chips', deviceIds.map((did) => {
          const registered = this.registryDevices.find((d) => d.id === did);
          const unbind = h('button.icon-btn.small', { type: 'button', 'aria-label': `${L.unbind}: ${did}`, title: L.unbind }, icon('close'));
          unbind.addEventListener('click', () => this.bindDevice(did, null));
          return h('li.a-chip', { title: did }, (registered && registered.name) || did, unbind);
        }))) : null);
  }

  // ------------------------------------------------------------ mutations

  addPerson() {
    // A new person starts on a non-sensitive profile: the general one, unless a household stored
    // before that rule still has a sensitive general profile.
    const general = this.profiles.find((p) => p.id === this.draft.general_profile);
    const start = general && !general.sensitive ? general : this.profiles.find((p) => !p.sensitive) || general;
    this.draft.people.push({ id: '', name: '', default_profile: start ? start.id : this.draft.general_profile, profiles: [], tts: null });
    this.redraw();
    const name = this.peopleList.querySelector('li:last-child .a-person-head input');
    if (name) name.focus();
  }

  /** Change a (new) person's id; phones and devices bound to them follow the new id. */
  renamePerson(person, id) {
    const old = person.id;
    person.id = id;
    if (old === id || this.draft.people.some((p) => p !== person && p.id === old)) return;
    for (const client of Object.values(this.draft.clients || {})) if (client.person === old) client.person = id;
    for (const room of Object.values(this.draft.devices || {})) if (room.person === old) room.person = id;
  }

  async removePerson(person) {
    const ok = await confirm({
      title: L.removeConfirmTitle(person.name || person.id),
      body: L.removeConfirmBody,
      confirmText: L.removeConfirm,
      danger: true,
    });
    if (!ok) return;
    this.draft.people = this.draft.people.filter((p) => p !== person);
    for (const client of Object.values(this.draft.clients || {})) if (client.person === person.id) client.person = null;
    for (const room of Object.values(this.draft.devices || {})) if (room.person === person.id) room.person = null;
    this.redraw();
  }

  bindClient(clientId, personId) {
    const client = this.draft.clients[clientId];
    if (!client) return;
    client.person = personId;
    this.redraw();
  }

  bindDevice(deviceId, personId) {
    if (personId === null) delete this.draft.devices[deviceId];
    else if (this.draft.devices[deviceId]) this.draft.devices[deviceId].person = personId;
    else this.draft.devices[deviceId] = { name: '', person: personId };
    this.redraw();
  }

  /** Client-side validation before the PUT; returns a Polish message or null. */
  validate() {
    const ids = new Set();
    for (const person of this.draft.people) {
      if (!person.name.trim()) return L.nameRequired;
      if (!ID_RE.test(person.id)) return L.idInvalid;
      if (ids.has(person.id)) return L.idTaken(person.id);
      ids.add(person.id);
    }
    return null;
  }

  change() {
    this.saveError = null;
    this.drawBanner();
    this.refreshBar();
  }

  /** A structural edit (add/remove/bind): redraw everything from the draft. */
  redraw() {
    this.saveError = null;
    this.draw();
  }

  discard() {
    this.draft = structuredClone(this.doc);
    this.saveError = null;
    this.draw();
  }

  refreshBar() {
    const dirty = this.dirty();
    this.saveBar.hidden = !dirty && !this.saveError;
    this.saveCount.textContent = dirty ? 'Niezapisane zmiany w domu' : 'Brak zmian do zapisu';
    clear(this.saveMsg);
    if (this.saveError) this.saveMsg.append(h('span.a-bar-error', this.saveError));
    this.saveBtn.disabled = this.saving || !dirty;
    this.discardBtn.disabled = this.saving || !dirty;
    this.saveBtn.textContent = this.saving ? L.saving : L.save;
  }

  drawBanner() {
    clear(this.banner);
    if (!this.saveError) return;
    this.banner.append(h('div.a-banner.warn', icon('warn'),
      h('div.a-banner-text', h('strong', L.saveRejected), h('p', this.saveError))));
  }

  async save() {
    if (this.saving || !this.dirty()) return;
    const invalid = this.validate();
    if (invalid) {
      this.saveError = invalid;
      this.drawBanner();
      this.refreshBar();
      return;
    }
    // The document goes out as a whole; fields the panel does not edit (the house note) pass
    // through unchanged, so saving here can never rewrite them.
    const payload = { ...this.draft, note: this.doc.note };
    this.saving = true;
    this.saveError = null;
    this.refreshBar();
    try {
      const res = await this.api.put('/api/admin/household', payload);
      this.doc = res.household;
      this.draft = structuredClone(this.doc);
    } catch (err) {
      this.saving = false;
      if (err.status === 401) return;
      this.saveError = err.message;
      this.drawBanner();
      this.refreshBar();
      return;
    }
    this.saving = false;
    toast(L.savedCount(this.doc.people.length));
    this.draw();
  }
}
