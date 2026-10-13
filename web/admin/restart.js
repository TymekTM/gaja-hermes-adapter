// "Zrestartuj adapter": confirm, POST /api/admin/restart, then wait until /api/admin/status reports a new start time.

import { h, confirm } from './ui.js';

const POLL_MS = 1500;
const GIVE_UP_S = 150;

function explain(err) {
  const text = err.message || '';
  if (err.status === 409 && /benchmark/i.test(text)) {
    return 'Trwa benchmark, który blokuje restart. Spróbuj ponownie, gdy się skończy.';
  }
  if (err.status === 409) {
    return 'Adapter nie działa jako usługa systemd, więc panel nie może go zrestartować. Zrestartuj go ręcznie na serwerze (systemctl --user restart gaja-adapter).';
  }
  return text || 'Nie udało się zlecić restartu.';
}

/**
 * Resolves { ok: true } once the adapter is back, { ok: false, message } when it could not be restarted
 * (409 and others) or { cancelled: true } when the user backed out.
 */
export async function restartAdapter(api) {
  const yes = await confirm({
    title: 'Zrestartować adapter?',
    body: h('div', h('p', 'Adapter będzie niedostępny przez 20-40 sekund. W tym czasie Gaja nie odpowie głosem, a otwarte rozmowy zostaną przerwane.'), h('p.a-muted', 'Zapisane ustawienia zaczną działać po restarcie.')),
    confirmText: 'Zrestartuj',
    danger: true,
  });
  if (!yes) return { cancelled: true };

  let before = null;
  try {
    before = (await api.get('/api/admin/status')).started_at;
  } catch (err) {
    return { ok: false, message: explain(err) };
  }
  try {
    await api.post('/api/admin/restart');
  } catch (err) {
    return { ok: false, message: explain(err), status: err.status };
  }
  return waitForRestart(api, before);
}

function waitForRestart(api, before) {
  return new Promise((resolve) => {
    const started = Date.now();
    const elapsed = h('span.a-tabular', '0 s');
    const title = h('h2', 'Restartuję adapter…');
    const note = h('p.a-muted', 'Strona odświeży się sama, gdy adapter wróci.');
    const spinner = h('div.a-spinner', { 'aria-hidden': 'true' });
    const actions = h('div.a-dialog-actions', { hidden: true });
    const box = h('div.a-overlay-box', { role: 'alertdialog', 'aria-labelledby': 'a-restart-title', 'aria-describedby': 'a-restart-note' }, spinner, title, note, h('p.a-muted', 'Czas: ', elapsed), actions);
    title.id = 'a-restart-title';
    note.id = 'a-restart-note';
    const overlay = h('div.a-overlay', box);
    document.body.append(overlay);

    let timer = null;
    const finish = (result) => {
      clearTimeout(timer);
      overlay.remove();
      resolve(result);
    };

    const tick = async () => {
      const secs = Math.round((Date.now() - started) / 1000);
      elapsed.textContent = `${secs} s`;
      try {
        const status = await api.get('/api/admin/status');
        if (status.started_at !== before) return finish({ ok: true });
      } catch (_) {
        // Down or not yet up: exactly what we expect while restarting.
      }
      if (secs > GIVE_UP_S) {
        spinner.hidden = true;
        title.textContent = 'Adapter nie wrócił w oczekiwanym czasie';
        note.textContent = 'Sprawdź usługę na serwerze (journalctl --user -u gaja-adapter). Panel nadal będzie próbował, gdy odświeżysz stronę.';
        const close = h('button.btn.primary', { type: 'button', onclick: () => finish({ ok: false, message: 'Adapter nie odpowiada po restarcie.' }) }, 'Zamknij');
        actions.hidden = false;
        actions.replaceChildren(close);
        close.focus();
        return;
      }
      timer = setTimeout(tick, POLL_MS);
    };
    timer = setTimeout(tick, 2500);
  });
}

