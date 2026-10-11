// WebSocket client: hello handshake, binary + JSON frames, reconnect with exponential backoff.
//
// onStatus({ status, attempt, retryInMs }) with status in
//   'connecting' | 'open' | 'closed' | 'retrying'

export class WSClient {
  constructor({ url, onMessage, onStatus } = {}) {
    this.url = url || WSClient.defaultUrl();
    this.onMessage = onMessage || (() => {});
    this.onStatus = onStatus || (() => {});

    this.ws = null;
    this.attempt = 0;
    this.retryTimer = null;
    this.pingTimer = null;
    this.wantOpen = false;
    this.hello = { lang: 'pl', tts: null, client: 'web' };
  }

  static defaultUrl() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    // keep ?demo=… so the mock server can run a scripted scenario
    return `${proto}//${location.host}/ws${location.search}`;
  }

  get isOpen() {
    return !!this.ws && this.ws.readyState === WebSocket.OPEN;
  }

  setHello(patch) {
    this.hello = { ...this.hello, ...patch, client: 'web' };
  }

  connect() {
    this.wantOpen = true;
    this.clearTimers();
    if (this.ws) {
      this.ws.onclose = null;
      try { this.ws.close(); } catch (_) { /* closing anyway */ }
    }
    this.onStatus({ status: 'connecting', attempt: this.attempt });

    let ws;
    try {
      ws = new WebSocket(this.url);
    } catch (err) {
      console.warn('WebSocket construction failed', err);
      this.scheduleRetry();
      return;
    }
    ws.binaryType = 'arraybuffer';
    this.ws = ws;

    ws.onopen = () => {
      this.attempt = 0;
      this.sendJson({ type: 'hello', ...this.hello });
      this.pingTimer = setInterval(() => this.sendJson({ type: 'ping' }), 20000);
      this.onStatus({ status: 'open', attempt: 0 });
    };

    ws.onmessage = (event) => {
      if (typeof event.data === 'string') {
        try {
          this.onMessage({ binary: false, data: JSON.parse(event.data) });
        } catch (err) {
          console.warn('invalid JSON from server', err);
        }
      } else {
        this.onMessage({ binary: true, data: event.data });
      }
    };

    ws.onerror = () => { /* the close handler reports it */ };

    ws.onclose = () => {
      clearInterval(this.pingTimer);
      this.pingTimer = null;
      this.onStatus({ status: 'closed', attempt: this.attempt });
      if (this.wantOpen) this.scheduleRetry();
    };
  }

  /** Re-handshake with new hello settings (language / voice). */
  reconnect() {
    this.attempt = 0;
    this.connect();
  }

  retryNow() {
    this.attempt = 0;
    this.connect();
  }

  scheduleRetry() {
    if (!this.wantOpen) return;
    this.attempt += 1;
    const base = Math.min(500 * 1.7 ** (this.attempt - 1), 10000);
    const delay = Math.round(base * (0.85 + Math.random() * 0.3));
    this.onStatus({ status: 'retrying', attempt: this.attempt, retryInMs: delay });
    this.retryTimer = setTimeout(() => this.connect(), delay);
  }

  clearTimers() {
    clearTimeout(this.retryTimer);
    clearInterval(this.pingTimer);
    this.retryTimer = this.pingTimer = null;
  }

  disconnect() {
    this.wantOpen = false;
    this.clearTimers();
    if (this.ws) {
      this.ws.onclose = null;
      try { this.ws.close(); } catch (_) { /* closing anyway */ }
      this.ws = null;
    }
    this.onStatus({ status: 'closed', attempt: 0 });
  }

  sendJson(obj) {
    if (this.isOpen) this.ws.send(JSON.stringify(obj));
  }

  sendBinary(buf) {
    if (this.isOpen) this.ws.send(buf);
  }

  sendText(text) { this.sendJson({ type: 'text', text }); }
  sendInterrupt() { this.sendJson({ type: 'interrupt' }); }
  /** cancel one Hermes task; without an id the server cancels all of them */
  sendCancelTask(id) { this.sendJson(id ? { type: 'cancel_task', id } : { type: 'cancel_task' }); }
  sendEndOfInput() { this.sendJson({ type: 'end_of_input' }); }
}
