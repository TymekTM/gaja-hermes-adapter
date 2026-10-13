// Urządzenia: voice devices (/api/devices).

import { h, icon, clear, pageHead, emptyState, errorState, relative } from '../ui.js';

export default {
  id: 'urzadzenia',
  title: 'Urządzenia',
  icon: 'speaker',
  render(container, api) {
    const list = h('ul.a-cards-grid');
    const refresh = h('button.btn.a-sm', { type: 'button', onclick: () => load() }, icon('refresh'), 'Odśwież');

    async function load() {
      try {
        const { devices } = await api.get('/api/devices');
        clear(list);
        if (!devices.length) {
          list.append(h('li.a-span', emptyState('speaker', 'Brak skonfigurowanych urządzeń.', 'Dodaj urządzenie poleceniem scripts/device_token.py na serwerze.')));
          return;
        }
        for (const dev of devices) list.append(card(dev));
      } catch (err) {
        clear(list).append(h('li.a-span', errorState(`Nie udało się wczytać urządzeń: ${err.message}`, load)));
      }
    }

    function when(ts) {
      return ts ? relative(ts * 1000) : 'nigdy';
    }

    function card(dev) {
      return h(`li.a-device${dev.online ? '.is-online' : ''}`,
        h('div.a-device-head',
          h('span.a-dot', { 'aria-hidden': 'true' }),
          h('strong', dev.name || dev.id),
          h(`span.a-badge${dev.online ? '.ok' : ''}`, dev.online ? 'online' : 'offline')),
        h('code.a-path', dev.id),
        h('dl.a-dl',
          h('div', h('dt', 'Ostatnio widziane'), h('dd', when(dev.last_seen))),
          h('div', h('dt', 'Ostatnie wybudzenie'), h('dd', when(dev.last_wake)))));
    }

    container.append(pageHead('Urządzenia', 'Głośniki z mikrofonem połączone z Gają.', refresh), list);
    load();
    const timer = setInterval(load, 10000);
    return () => clearInterval(timer);
  },
};
