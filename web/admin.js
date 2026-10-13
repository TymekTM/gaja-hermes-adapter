// Gaja admin panel: login screen, shell (navigation) and section routing. Sections live in web/admin/sections/.

import { createApi } from './admin/api.js';
import { SECTIONS } from './admin/sections/index.js';
import { h, icon, clear, toast } from './admin/ui.js';

const root = document.getElementById('admin-root');
let shell = null; // { view, setActive, dispose } while logged in
let leaveGuard = null;

const api = {
  ...createApi({ onUnauthorized: () => showLogin({ notice: 'Sesja wygasła. Zaloguj się ponownie.' }) }),
  setLeaveGuard(fn) { leaveGuard = fn; },
};

// ------------------------------------------------------------------ theme (shares the voice client's setting)

const THEMES = ['system', 'light', 'dark'];
const THEME_LABELS = { system: 'systemowy', light: 'jasny', dark: 'ciemny' };

function currentTheme() {
  try { return JSON.parse(localStorage.getItem('gaja.settings') || '{}').theme || 'system'; } catch (_) { return 'system'; }
}

function applyTheme(theme) {
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem('gaja.settings') || '{}'); } catch (_) { /* start fresh */ }
  if (theme === 'system') delete saved.theme; else saved.theme = theme;
  try { localStorage.setItem('gaja.settings', JSON.stringify(saved)); } catch (_) { /* private mode */ }
  if (theme === 'system') delete document.documentElement.dataset.theme; else document.documentElement.dataset.theme = theme;
}

// ------------------------------------------------------------------ login

function loginMessage(err) {
  switch (err.status) {
    case 401: return { text: 'Nieprawidłowe hasło.' };
    case 429: return { text: 'Zbyt wiele nieudanych prób logowania. Odczekaj kilka minut i spróbuj ponownie.' };
    case 503: return { text: 'Na serwerze nie ustawiono jeszcze hasła do panelu. Ustaw je poleceniem:', code: 'uv run python scripts/admin_password.py' };
    case 403: return { text: 'Przeglądarka łączy się z adresu, który nie jest dozwolony (server.allowed_origins).' };
    case 0: return { text: 'Brak połączenia z adapterem.' };
    default: return { text: `Logowanie nie powiodło się: ${err.message}` };
  }
}

async function showLogin({ notice = null } = {}) {
  if (shell) { shell.dispose(); shell = null; }
  leaveGuard = null;
  document.title = 'Logowanie · Gaja';
  root.dataset.view = 'login';

  let configured = true;
  try { configured = (await api.get('/api/admin/session')).configured; } catch (_) { /* the login attempt reports it */ }

  const password = h('input.a-input', { type: 'password', id: 'a-password', name: 'password', autocomplete: 'current-password', required: true, 'aria-describedby': 'a-login-msg' });
  const reveal = h('button.a-reveal', { type: 'button', 'aria-label': 'Pokaż hasło', 'aria-pressed': 'false' }, 'Pokaż');
  reveal.addEventListener('click', () => {
    const show = password.type === 'password';
    password.type = show ? 'text' : 'password';
    reveal.textContent = show ? 'Ukryj' : 'Pokaż';
    reveal.setAttribute('aria-pressed', String(show));
    reveal.setAttribute('aria-label', show ? 'Ukryj hasło' : 'Pokaż hasło');
  });
  const msg = h('div#a-login-msg.a-login-msg', { role: 'alert', 'aria-live': 'assertive' });
  const submit = h('button.btn.primary.a-login-btn', { type: 'submit' }, 'Zaloguj');
  const form = h('form.a-login-form', { novalidate: true },
    h('label', { for: 'a-password' }, 'Hasło administratora'),
    h('div.a-pass', password, reveal),
    msg, submit);

  const showMessage = (m) => {
    clear(msg);
    if (!m) return;
    msg.append(icon('warn'), h('div', h('p', m.text), m.code ? h('code.a-code', m.code) : null));
    msg.classList.add('on');
  };
  if (notice) showMessage({ text: notice });
  else if (!configured) showMessage(loginMessage({ status: 503 }));

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!password.value) { password.focus(); return; }
    submit.disabled = true;
    submit.textContent = 'Loguję…';
    msg.classList.remove('on');
    try {
      await api.post('/api/admin/login', { password: password.value });
    } catch (err) {
      showMessage(loginMessage(err));
      submit.disabled = false;
      submit.textContent = 'Zaloguj';
      password.select();
      return;
    }
    showShell();
  });

  clear(root).append(
    h('main.a-login',
      h('div.a-login-card',
        h('div.a-login-brand', h('span.brand-dot', { 'aria-hidden': 'true' }), h('span', 'Gaja')),
        h('h1', 'Panel administracyjny'),
        h('p.a-lede', 'Ustawienia, pamięć i diagnostyka asystenta. Zaloguj się hasłem administratora.'),
        form,
        h('a.a-login-back', { href: '/' }, 'Wróć do Gai'))));
  password.focus();
}

// ------------------------------------------------------------------ shell

function sectionFromHash() {
  const id = decodeURIComponent((location.hash || '').replace(/^#\/?/, ''));
  return SECTIONS.find((s) => s.id === id) || SECTIONS[0];
}

function showShell() {
  if (shell) shell.dispose();
  root.dataset.view = 'shell';
  root.dataset.menu = 'closed';

  const main = h('main.a-main', { id: 'a-main', tabindex: '-1' });
  const titleEl = h('span.a-topbar-title');
  const menuBtn = h('button.icon-btn.a-menu-btn', { type: 'button', 'aria-label': 'Menu', 'aria-expanded': 'false', 'aria-controls': 'a-nav' }, icon('menu'));
  const links = new Map();
  const nav = h('nav#a-nav.a-nav', { 'aria-label': 'Sekcje panelu' },
    h('ul.a-nav-list', SECTIONS.map((s) => {
      const link = h('a.a-nav-link', { href: `#/${s.id}`, dataset: { id: s.id } }, icon(s.icon || 'sliders'), h('span', s.title));
      links.set(s.id, link);
      return h('li', link);
    })));

  const themeBtn = h('button.a-nav-link', { type: 'button' });
  const paintTheme = () => {
    const t = currentTheme();
    clear(themeBtn).append(icon(t === 'dark' ? 'moon' : 'sun'), h('span', `Motyw: ${THEME_LABELS[t]}`));
  };
  themeBtn.addEventListener('click', () => {
    applyTheme(THEMES[(THEMES.indexOf(currentTheme()) + 1) % THEMES.length]);
    paintTheme();
  });
  paintTheme();
  const logoutBtn = h('button.a-nav-link', { type: 'button' }, icon('logout'), h('span', 'Wyloguj'));
  logoutBtn.addEventListener('click', async () => {
    if (leaveGuard && !(await leaveGuard())) return;
    try { await api.post('/api/admin/logout'); } catch (_) { /* the cookie dies with the session anyway */ }
    showLogin();
  });
  nav.append(h('div.a-nav-foot',
    h('a.a-nav-link', { href: '/' }, icon('home'), h('span', 'Wróć do Gai')),
    themeBtn, logoutBtn));

  const setMenu = (open) => {
    root.dataset.menu = open ? 'open' : 'closed';
    menuBtn.setAttribute('aria-expanded', String(open));
  };
  menuBtn.addEventListener('click', () => setMenu(root.dataset.menu !== 'open'));
  const scrim = h('div.a-scrim', { onclick: () => setMenu(false) });
  const onKey = (e) => { if (e.key === 'Escape' && root.dataset.menu === 'open') { setMenu(false); menuBtn.focus(); } };
  document.addEventListener('keydown', onKey);

  let current = null;
  let disposeView = null;
  let navigating = false;

  async function show(section, { push = true } = {}) {
    if (navigating) return;
    if (current && current !== section && leaveGuard) {
      navigating = true;
      const ok = await leaveGuard();
      navigating = false;
      if (!ok) {
        history.replaceState(null, '', `#/${current.id}`);
        return;
      }
    }
    if (disposeView) { disposeView(); disposeView = null; }
    leaveGuard = null;
    current = section;
    if (push && location.hash !== `#/${section.id}`) history.pushState(null, '', `#/${section.id}`);
    for (const [id, link] of links) {
      if (id === section.id) link.setAttribute('aria-current', 'page'); else link.removeAttribute('aria-current');
    }
    titleEl.textContent = section.title;
    document.title = `${section.title} · Gaja`;
    setMenu(false);
    const view = h('div.a-view');
    clear(main).append(view);
    main.scrollTop = 0;
    try {
      const result = section.render(view, api);
      if (typeof result === 'function') disposeView = result;
    } catch (err) {
      view.append(h('div.a-empty.is-error', icon('warn'), h('p', `Sekcja nie wczytała się: ${err.message}`)));
    }
  }

  nav.addEventListener('click', (e) => {
    const link = e.target.closest('a[data-id]');
    if (!link) return;
    e.preventDefault();
    show(SECTIONS.find((s) => s.id === link.dataset.id));
  });
  const onPop = () => show(sectionFromHash(), { push: false });
  window.addEventListener('popstate', onPop);

  clear(root).append(
    h('header.a-topbar', menuBtn, h('span.a-topbar-brand', h('span.brand-dot', { 'aria-hidden': 'true' }), 'Gaja'), titleEl),
    h('aside.a-side',
      h('div.a-side-brand', h('span.brand-dot', { 'aria-hidden': 'true' }), h('span', 'Gaja'), h('small', 'panel')),
      nav),
    scrim,
    main);

  shell = {
    dispose() {
      if (disposeView) disposeView();
      document.removeEventListener('keydown', onKey);
      window.removeEventListener('popstate', onPop);
    },
  };
  show(sectionFromHash(), { push: false });
}

// ------------------------------------------------------------------ boot

async function boot() {
  try {
    const session = await api.get('/api/admin/session');
    if (session.logged_in) showShell(); else showLogin();
  } catch (err) {
    clear(root).append(h('main.a-login', h('div.a-login-card', h('h1', 'Panel niedostępny'), h('p.a-lede', 'Nie udało się połączyć z adapterem. Sprawdź, czy działa, i odśwież stronę.'), h('button.btn.primary', { type: 'button', onclick: () => location.reload() }, 'Odśwież'))));
    toast(err.message, 'err');
  }
}

boot();
