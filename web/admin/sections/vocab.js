// Słownik: the user's STT vocabulary and the suggestions collected from conversations (/api/vocab).

import { h, icon, clear, toast, confirm, pageHead, emptyState, errorState } from '../ui.js';

const REASONS = {
  'near-miss': 'podobne brzmienie do znanego słowa',
  'user-correction': 'poprawka wypowiedziana przez użytkownika',
  'unknown-token': 'nieznane słowo w wypowiedzi',
};
const LANGS = { '': 'dowolny', pl: 'polski', en: 'angielski' };

export default {
  id: 'slownik',
  title: 'Słownik',
  icon: 'book',
  render(container, api) {
    let data = { terms: [], suggestions: [] };
    let editing = null; // the user term being edited (by its current spelling)

    const suggestionsBox = h('section.a-block');
    const termsBox = h('section.a-block');
    const builtinBox = h('details.a-fold');

    // ---- add / edit form
    const term = h('input.a-input', { type: 'text', id: 'v-term', placeholder: 'Np. Tailscale', autocomplete: 'off', required: true, maxlength: 80 });
    const aliases = h('input.a-input', { type: 'text', id: 'v-aliases', placeholder: 'tejlskil, tailskejl', autocomplete: 'off' });
    const lang = h('select.a-input.a-select', { id: 'v-lang' }, Object.entries(LANGS).map(([v, t]) => h('option', { value: v }, t)));
    const submit = h('button.btn.primary', { type: 'submit' }, icon('plus'), 'Dodaj słowo');
    const cancelEdit = h('button.btn', { type: 'button', hidden: true, onclick: () => resetForm() }, 'Anuluj edycję');
    const form = h('form.a-vocab-form',
      h('div.a-form-grid',
        h('div.a-form-field', h('label', { for: 'v-term' }, 'Słowo'), term),
        h('div.a-form-field', h('label', { for: 'v-aliases' }, 'Jak bywa błędnie słyszane'), aliases),
        h('div.a-form-field.narrow', h('label', { for: 'v-lang' }, 'Język'), h('div.a-select-wrap', lang, icon('chevron', 'a-select-chev')))),
      h('p.a-muted', 'Warianty oddziel przecinkami. Wpisane słowo trafia do podpowiedzi dla rozpoznawania mowy.'),
      h('div.a-form-actions', submit, cancelEdit));

    function resetForm() {
      editing = null;
      form.reset();
      submit.replaceChildren(icon('plus'), 'Dodaj słowo');
      cancelEdit.hidden = true;
    }

    function startEdit(item) {
      editing = item.term;
      term.value = item.term;
      aliases.value = (item.aliases || []).join(', ');
      lang.value = item.lang || '';
      submit.replaceChildren(icon('check'), 'Zapisz zmiany');
      cancelEdit.hidden = false;
      form.scrollIntoView({ block: 'center', behavior: 'smooth' });
      term.focus();
    }

    const userTerms = () => data.terms.filter((t) => !t.builtin);
    const payload = (terms) => terms.map((t) => (t.lang ? { term: t.term, aliases: t.aliases || [], lang: t.lang } : { term: t.term, aliases: t.aliases || [] }));

    async function putTerms(terms, okMessage) {
      try {
        data = await api.put('/api/vocab', { terms: payload(terms) });
        toast(okMessage);
        draw();
        return true;
      } catch (err) {
        toast(`Nie udało się zapisać słownika: ${err.message}`, 'err');
        return false;
      }
    }

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const word = term.value.trim();
      if (!word) { term.focus(); return; }
      const entry = { term: word, aliases: aliases.value.split(',').map((s) => s.trim()).filter(Boolean), lang: lang.value || null };
      const others = userTerms().filter((t) => t.term !== editing);
      if (others.some((t) => t.term.toLowerCase() === word.toLowerCase()) || data.terms.some((t) => t.builtin && t.term.toLowerCase() === word.toLowerCase())) {
        toast(`„${word}” jest już w słowniku.`, 'err');
        return;
      }
      const wasEditing = editing !== null;
      const index = wasEditing ? userTerms().findIndex((t) => t.term === editing) : -1;
      const next = userTerms();
      if (index >= 0) next[index] = entry; else next.push(entry);
      if (await putTerms(next, wasEditing ? 'Zapisano zmiany.' : 'Dodano do słownika.')) resetForm();
    });

    // ---- drawing
    function draw() {
      drawSuggestions();
      drawTerms();
    }

    function drawSuggestions() {
      clear(suggestionsBox);
      const list = data.suggestions || [];
      if (!list.length) {
        suggestionsBox.append(h('h2.a-h2', 'Podpowiedzi'), h('p.a-muted', 'Brak podpowiedzi. Gaja zbiera je, gdy słyszy nieznane słowa lub jesteś poprawiana.'));
        return;
      }
      suggestionsBox.append(h('h2.a-h2', 'Podpowiedzi ', h('span.a-count', String(list.length))),
        h('ul.a-rows', list.map((s) => suggestionRow(s))));
    }

    function suggestionRow(s) {
      const act = async (verb) => {
        try {
          data = await api.post(`/api/vocab/suggestions/${encodeURIComponent(s.id)}/${verb}`);
          toast(verb === 'approve' ? `„${s.term}” dodane do słownika.` : 'Podpowiedź odrzucona.');
          draw();
        } catch (err) { toast(`Nie udało się: ${err.message}`, 'err'); }
      };
      return h('li.a-row',
        h('div.a-row-main',
          h('p.a-row-title', s.term),
          h('p.a-row-sub', `słyszane jako: ${(s.heard || []).join(', ') || '–'} · ${s.count}× · ${REASONS[s.reason] || s.reason}`)),
        h('div.a-row-actions',
          h('button.btn.a-sm.primary', { type: 'button', onclick: () => act('approve'), 'aria-label': `Zatwierdź: ${s.term}` }, icon('check'), 'Zatwierdź'),
          h('button.btn.a-sm', { type: 'button', onclick: () => act('reject'), 'aria-label': `Odrzuć: ${s.term}` }, 'Odrzuć')));
    }

    function drawTerms() {
      const mine = userTerms();
      clear(termsBox).append(h('h2.a-h2', 'Twoje słowa ', h('span.a-count', mine.length ? String(mine.length) : '')));
      if (!mine.length) termsBox.append(emptyState('book', 'Nie dodano jeszcze własnych słów.', 'Dodaj nazwy własne, imiona i terminy techniczne, które Gaja myli.'));
      else termsBox.append(h('ul.a-rows', mine.map((t) => termRow(t))));

      const builtin = data.terms.filter((t) => t.builtin);
      clear(builtinBox).append(
        h('summary', `Wbudowane słowa (${builtin.length})`),
        h('ul.a-chips', builtin.map((t) => h('li.a-chip', { title: (t.aliases || []).join(', ') || undefined }, t.term))));
    }

    function termRow(t) {
      const del = h('button.icon-btn.small.a-danger-ico', { type: 'button', 'aria-label': `Usuń: ${t.term}`, title: 'Usuń słowo' }, icon('trash'));
      del.addEventListener('click', async () => {
        const ok = await confirm({ title: `Usunąć „${t.term}”?`, body: 'Gaja przestanie podpowiadać to słowo rozpoznawaniu mowy.', confirmText: 'Usuń', danger: true });
        if (ok) await putTerms(userTerms().filter((x) => x.term !== t.term), 'Usunięto ze słownika.');
      });
      const edit = h('button.btn.a-sm', { type: 'button', 'aria-label': `Edytuj: ${t.term}`, onclick: () => startEdit(t) }, 'Edytuj');
      const meta = [(t.aliases || []).length ? `warianty: ${t.aliases.join(', ')}` : 'bez wariantów', t.lang ? LANGS[t.lang] || t.lang : null].filter(Boolean).join(' · ');
      return h('li.a-row', h('div.a-row-main', h('p.a-row-title', t.term), h('p.a-row-sub', meta)), h('div.a-row-actions', edit, del));
    }

    async function load() {
      try {
        data = await api.get('/api/vocab');
      } catch (err) {
        const unavailable = err.status === 503;
        clear(body).append(unavailable
          ? emptyState('book', 'Słownik jest wyłączony.', 'Włącz go w Ustawieniach (vocab.enabled) i zrestartuj adapter.')
          : errorState(`Nie udało się wczytać słownika: ${err.message}`, load));
        return;
      }
      clear(body).append(suggestionsBox, h('h2.a-h2', 'Dodaj lub zmień słowo'), form, termsBox, builtinBox);
      draw();
    }

    const body = h('div');
    container.append(pageHead('Słownik', 'Słowa, które pomagają rozpoznawaniu mowy: nazwy własne, terminy, imiona.'), body);
    load();
  },
};
