// Settings sheet: a bottom sheet on phones (drag the grip down to close), a floating panel on wide
// screens. The rest of the page is made inert while it is open, so focus cannot wander behind it.

const WIDE = window.matchMedia('(min-width: 900px) and (min-height: 480px)');

export class SettingsSheet {
  constructor({ sheet, backdrop, app, openBtn, closeBtn, grip, onOpen, onClose }) {
    this.sheet = sheet;
    this.backdrop = backdrop;
    this.app = app;
    this.openBtn = openBtn;
    this.closeBtn = closeBtn;
    this.onOpen = onOpen || (() => {});
    this.onClose = onClose || (() => {});
    this.isOpen = false;
    this.lastFocus = null;

    openBtn.addEventListener('click', () => (this.isOpen ? this.close() : this.open()));
    closeBtn.addEventListener('click', () => this.close());
    backdrop.addEventListener('click', () => this.close());
    this.attachDrag(grip);
  }

  open() {
    if (this.isOpen) return;
    this.isOpen = true;
    this.lastFocus = document.activeElement;
    this.onOpen();
    this.sheet.inert = false;
    this.app.inert = true;
    this.backdrop.hidden = false;
    this.openBtn.setAttribute('aria-expanded', 'true');
    // next frame, so the slide-in transition runs from the closed position
    requestAnimationFrame(() => {
      this.sheet.classList.add('is-open');
      this.closeBtn.focus({ preventScroll: true });
    });
  }

  close() {
    if (!this.isOpen) return false;
    this.isOpen = false;
    this.sheet.classList.remove('is-open');
    this.sheet.style.removeProperty('--drag');
    this.sheet.inert = true;
    this.app.inert = false;
    this.backdrop.hidden = true;
    this.openBtn.setAttribute('aria-expanded', 'false');
    const back = this.lastFocus && document.contains(this.lastFocus) ? this.lastFocus : this.openBtn;
    back.focus({ preventScroll: true });
    this.onClose();
    return true;
  }

  attachDrag(grip) {
    const sheet = this.sheet;
    let startY = 0;
    let dy = 0;
    let startT = 0;
    let dragging = false;
    let moved = false;

    const onMove = (e) => {
      if (!dragging) return;
      dy = Math.max(0, e.clientY - startY);
      if (dy > 4) moved = true;
      sheet.style.setProperty('--drag', `${dy}px`);
    };
    const onUp = () => {
      if (!dragging) return;
      dragging = false;
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
      window.removeEventListener('pointercancel', onUp);
      sheet.classList.remove('dragging');
      const velocity = dy / Math.max(1, performance.now() - startT); // px/ms
      sheet.style.removeProperty('--drag');
      if (moved && (dy > 110 || velocity > 0.6)) this.close();
    };

    const head = sheet.querySelector('.sheet-head');
    [grip, head].forEach((h) => {
      if (!h) return;
      h.addEventListener('pointerdown', (e) => {
        if (WIDE.matches || e.button > 0 || e.target.closest('button')) return;
        dragging = true;
        moved = false;
        startY = e.clientY;
        startT = performance.now();
        dy = 0;
        sheet.classList.add('dragging');
        window.addEventListener('pointermove', onMove);
        window.addEventListener('pointerup', onUp);
        window.addEventListener('pointercancel', onUp);
      });
    });
  }
}
