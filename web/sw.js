// Service Worker for Gaja Voice Assistant PWA
// Caches the app shell for offline presentation. NEVER caches /ws or /api/*.

const CACHE_NAME = 'gaja-shell-v1';
const SHELL_ASSETS = [
  './',
  'index.html',
  'style.css',
  'app.js',
  'audio.js',
  'capture-processor.js',
  'playback-processor.js',
  'chat.js',
  'orb.js',
  'settings.js',
  'history.js',
  'i18n.js',
  'ws.js',
  'tasks.js',
  'manifest.webmanifest',
  'icon.svg',
  'icon-maskable.svg',
  'icons.svg',
  'icons/apple-touch-icon.png',
  'icons/icon-192.png',
  'icons/icon-maskable-192.png',
  'icons/icon-512.png',
  'icons/icon-maskable-512.png',
  'fonts/bricolage-grotesque-latin.woff2',
  'fonts/bricolage-grotesque-latin-ext.woff2',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(SHELL_ASSETS)).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;

  const url = new URL(req.url);

  // CRITICAL: Never cache /ws or /api/* endpoints
  if (url.pathname.startsWith('/api/') || url.pathname === '/ws' || url.protocol === 'ws:' || url.protocol === 'wss:') {
    return;
  }

  // App shell caching: cache-first or stale-while-revalidate
  event.respondWith(
    caches.match(req).then((cached) => {
      const networkFetch = fetch(req).then((response) => {
        if (response && response.status === 200 && response.type === 'basic') {
          const clone = response.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(req, clone));
        }
        return response;
      }).catch(() => cached);

      return cached || networkFetch;
    })
  );
});
