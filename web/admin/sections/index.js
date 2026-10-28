// The one place where admin sections are registered. To add a section (people, Hermes profiles,
// household, speaker log, ...) create a module that default-exports
//
//   { id: 'ludzie', title: 'Ludzie', icon: 'sliders', render(container, api) { ...; return dispose? } }
//
// (`icon` is a symbol name from web/admin-icons.svg) and add it to the list below. The shell builds the
// navigation, the #/id routes and the unsaved-changes guard from this list.
//
// api: { get, post, put, del } (JSON, rejects with ApiError{status, message}; a 401 from /api/admin/*
// sends the user back to the login screen) and setLeaveGuard(async () => boolean) to veto navigation.

import settings from './settings.js';
import memory from './memory.js';
import timers from './timers.js';
import vocab from './vocab.js';
import devices from './devices.js';
import household from './household.js';
import diagnostics from './diagnostics.js';

export const SECTIONS = [settings, household, memory, timers, vocab, devices, diagnostics];
