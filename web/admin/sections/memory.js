// Pamięć: facts Gaja remembers about the household (/api/memory).

import { h, icon, clear, toast, confirm, pageHead, emptyState, errorState } from '../ui.js';

export default {
  id: 'pamiec',
  title: 'Pamięć',
  icon: 'note',
  render(container, api) {
    const list = h('ul.a-rows');
    const input = h('input.a-input', { type: 'text', placeholder: 'Np. Ulubiona kawa Tymka to flat white.', 'aria-label': 'Nowy fakt', maxlength: 500, autocomplete: 'off' });
    const addBtn = h('button.btn.primary', { type: 'submit' }, icon('plus'), 'Dodaj');
    const form = h('form.a-inline-form', input, addBtn);
    const count = h('span.a-count');

    async function load() {
      try {
        const { facts } = await api.get('/api/memory');
        count.textContent = facts.length ? String(facts.length) : '';
        clear(list);
        if (!facts.length) {
          list.append(h('li', emptyState('note', 'Gaja jeszcze niczego nie zapamiętała.', 'Dodaj fakt powyżej albo powiedz „zapamiętaj, że…”.')));
          return;
        }
        for (const fact of facts) list.append(row(fact));
      } catch (err) {
        clear(list).append(h('li', errorState(`Nie udało się wczytać pamięci: ${err.message}`, load)));
      }
    }

    function row(fact) {
      const created = fact.created ? new Date(fact.created).toLocaleDateString('pl-PL', { day: 'numeric', month: 'short', year: 'numeric' }) : '';
      const del = h('button.icon-btn.small.a-danger-ico', { type: 'button', 'aria-label': `Usuń: ${fact.text}`, title: 'Usuń fakt' }, icon('trash'));
      del.addEventListener('click', async () => {
        const ok = await confirm({ title: 'Usunąć ten fakt?', body: h('p.a-quote', fact.text), confirmText: 'Usuń', danger: true });
        if (!ok) return;
        try {
          await api.del(`/api/memory/${encodeURIComponent(fact.id)}`);
          toast('Fakt usunięty.');
          await load();
        } catch (err) { toast(`Nie udało się usunąć: ${err.message}`, 'err'); }
      });
      return h('li.a-row', h('div.a-row-main', h('p.a-row-title', fact.text), created ? h('p.a-row-sub', `dodano ${created}`) : null), del);
    }

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const text = input.value.trim();
      if (!text) { input.focus(); return; }
      addBtn.disabled = true;
      try {
        await api.post('/api/memory', { text });
        input.value = '';
        toast('Zapamiętane.');
        await load();
      } catch (err) { toast(`Nie udało się dodać: ${err.message}`, 'err'); }
      addBtn.disabled = false;
    });

    container.append(
      pageHead('Pamięć', 'Fakty, które Gaja dołącza do rozmów. Usunięcie faktu działa od razu.'),
      form,
      h('h2.a-h2', 'Zapamiętane fakty ', count),
      list,
    );
    load();
  },
};
