// Minutniki: active timers and reminders (/api/timers).

import { h, icon, clear, toast, pageHead, emptyState, errorState, relative } from '../ui.js';

const KINDS = { timer: 'minutnik', reminder: 'przypomnienie' };

export default {
  id: 'minutniki',
  title: 'Minutniki',
  icon: 'timer',
  render(container, api) {
    const list = h('ul.a-rows');
    const refresh = h('button.btn.a-sm', { type: 'button', onclick: () => load() }, icon('refresh'), 'Odśwież');

    async function load() {
      try {
        const { timers } = await api.get('/api/timers');
        clear(list);
        if (!timers.length) {
          list.append(h('li', emptyState('timer', 'Brak aktywnych minutników.', 'Powiedz „ustaw minutnik na 10 minut”.')));
          return;
        }
        timers.sort((a, b) => new Date(a.due) - new Date(b.due));
        for (const timer of timers) list.append(row(timer));
      } catch (err) {
        clear(list).append(h('li', errorState(`Nie udało się wczytać minutników: ${err.message}`, load)));
      }
    }

    function row(timer) {
      const due = new Date(timer.due);
      const clock = due.toLocaleString('pl-PL', { weekday: 'short', hour: '2-digit', minute: '2-digit' });
      const cancel = h('button.btn.a-sm', { type: 'button', 'aria-label': `Anuluj: ${timer.label}` }, icon('close'), 'Anuluj');
      cancel.addEventListener('click', async () => {
        cancel.disabled = true;
        try {
          await api.del(`/api/timers/${encodeURIComponent(timer.id)}`);
          toast('Minutnik anulowany.');
          await load();
        } catch (err) {
          cancel.disabled = false;
          toast(`Nie udało się anulować: ${err.message}`, 'err');
        }
      });
      const sub = [KINDS[timer.kind] || timer.kind, `o ${clock}`, timer.device_id ? `urządzenie: ${timer.device_id}` : null].filter(Boolean).join(' · ');
      return h('li.a-row',
        h('div.a-row-main', h('p.a-row-title', timer.label || 'Minutnik'), h('p.a-row-sub', sub)),
        h('span.a-badge.accent.a-tabular', relative(due.getTime())),
        cancel);
    }

    container.append(pageHead('Minutniki', 'Aktywne minutniki i przypomnienia, także te ustawione głosem.', refresh), list);
    load();
    const timer = setInterval(load, 15000);
    return () => clearInterval(timer);
  },
};
