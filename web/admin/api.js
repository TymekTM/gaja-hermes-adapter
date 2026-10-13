// Fetch wrapper for the admin panel. A 401 from /api/admin/* returns the user to the login screen.

export class ApiError extends Error {
  constructor(status, message, body) {
    super(message);
    this.status = status;
    this.body = body;
  }
}

export function createApi({ onUnauthorized }) {
  async function request(method, url, body) {
    const init = { method, credentials: 'same-origin', headers: {} };
    if (body !== undefined) {
      init.headers['Content-Type'] = 'application/json';
      init.body = JSON.stringify(body);
    }
    let res;
    try {
      res = await fetch(url, init);
    } catch (_) {
      throw new ApiError(0, 'Brak połączenia z adapterem.', null);
    }
    let data = null;
    const text = await res.text();
    if (text) {
      try { data = JSON.parse(text); } catch (_) { data = text; }
    }
    if (!res.ok) {
      const detail = data && typeof data === 'object' && 'detail' in data ? data.detail : data;
      const message = typeof detail === 'string' ? detail : detail ? JSON.stringify(detail) : `HTTP ${res.status}`;
      if (res.status === 401 && url.startsWith('/api/admin/') && !url.endsWith('/login')) onUnauthorized();
      throw new ApiError(res.status, message, data);
    }
    return data;
  }

  return {
    get: (url) => request('GET', url),
    post: (url, body) => request('POST', url, body ?? {}),
    put: (url, body) => request('PUT', url, body),
    del: (url) => request('DELETE', url),
  };
}
