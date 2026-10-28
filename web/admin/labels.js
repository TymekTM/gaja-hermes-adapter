// Polish labels for the settings form. The dotted paths match /api/admin/config (config.toml).
// A field or section missing here still renders (with its raw path as label), so new config keys never break the panel.

export const SECTION_TITLES = {
  server: 'Serwer',
  llm: 'Model językowy (LLM)',
  vad: 'Wykrywanie mowy (VAD)',
  stt: 'Rozpoznawanie mowy (STT)',
  'stt.faster_whisper': 'STT: faster-whisper',
  tts: 'Synteza mowy (TTS)',
  'tts.piper': 'TTS: Piper (Gosia)',
  'tts.piper_darkman': 'TTS: Piper (Darkman)',
  'tts.paradee': 'TTS: Paradee',
  'tts.supertonic': 'TTS: Supertonic',
  chunker: 'Dzielenie odpowiedzi na zdania',
  router: 'Router zapytań',
  style: 'Styl i emocje głosu',
  tools: 'Narzędzia',
  text: 'Przetwarzanie tekstu',
  hermes: 'Hermes (agent)',
  metrics: 'Metryki',
  wakeword: 'Słowo wybudzające',
  devices: 'Urządzenia głosowe',
  vocab: 'Słownik STT',
  admin: 'Panel administracyjny',
};

// Display order of the sections; anything not listed follows alphabetically.
export const SECTION_ORDER = Object.keys(SECTION_TITLES);

export const FIELD_LABELS = {
  'server.host': 'Adres nasłuchu',
  'server.port': 'Port',
  'server.web_dir': 'Katalog klienta WWW',
  'server.allowed_origins': 'Dozwolone originy przeglądarki',

  'llm.base_url': 'Adres serwera LLM',
  'llm.model': 'Nazwa modelu',
  'llm.temperature': 'Temperatura',
  'llm.max_tokens': 'Maks. długość odpowiedzi (tokeny)',
  'llm.timeout_s': 'Limit czasu (s)',
  'llm.tool_strategy': 'Strategia wywołań narzędzi',
  'llm.system_prompt_path': 'Plik promptu systemowego',
  'llm.history_max_turns': 'Maks. liczba tur historii',
  'llm.history_max_chars': 'Maks. długość historii (znaki)',
  'llm.router_slot': 'Slot routera',
  'llm.chat_slot': 'Slot rozmowy',
  'llm.prime_after_turn': 'Rozgrzewaj cache po turze',

  'vad.threshold': 'Próg wykrycia mowy',
  'vad.min_silence_ms': 'Cisza kończąca wypowiedź (ms)',
  'vad.spec_silence_ms': 'Cisza dla startu spekulatywnego (ms)',
  'vad.pre_roll_ms': 'Bufor przed mową (ms)',
  'vad.min_speech_ms': 'Minimalna długość mowy (ms)',
  'vad.max_utterance_s': 'Maks. długość wypowiedzi (s)',
  'vad.barge_in_ms': 'Czas mowy do przerwania (ms)',
  'vad.barge_in_strict_ms': 'Czas mowy do przerwania, gdy Gaja mówi (ms)',
  'vad.strict_threshold': 'Próg wykrycia, gdy Gaja mówi',
  'vad.smart_turn': 'Smart Turn (semantyczny koniec wypowiedzi)',
  'vad.smart_turn_model_path': 'Plik modelu Smart Turn',
  'vad.smart_turn_check_ms': 'Sprawdzenie Smart Turn po ciszy (ms)',
  'vad.smart_turn_threshold': 'Próg Smart Turn',
  'vad.max_incomplete_silence_ms': 'Maks. cisza przy niedokończonej myśli (ms)',

  'stt.default_language': 'Domyślny język',
  'stt.engine': 'Silnik STT',
  'stt.faster_whisper.model': 'Model Whisper',
  'stt.faster_whisper.device': 'Urządzenie obliczeń',
  'stt.faster_whisper.compute_type': 'Typ obliczeń (kwantyzacja)',
  'stt.faster_whisper.beam_size': 'Szerokość wiązki (beam)',

  'tts.engine_by_language': 'Silnik głosu według języka',
  'tts.piper.model_path': 'Plik modelu',
  'tts.piper.threads': 'Liczba wątków',
  'tts.piper_darkman.model_path': 'Plik modelu',
  'tts.piper_darkman.threads': 'Liczba wątków',
  'tts.paradee.threads': 'Liczba wątków',
  'tts.supertonic.steps': 'Liczba kroków syntezy',
  'tts.supertonic.voice': 'Głos',
  'tts.supertonic.speed': 'Tempo mowy',
  'tts.supertonic.threads': 'Liczba wątków',
  'tts.supertonic.output_sample_rate': 'Częstotliwość wyjściowa (Hz)',

  'chunker.first_chunk_min_words': 'Pierwsze zdanie: min. słów',
  'chunker.first_chunk_max_words': 'Pierwsze zdanie: maks. słów',
  'chunker.min_words': 'Kolejne zdania: min. słów',
  'chunker.max_chars': 'Maks. długość fragmentu (znaki)',

  'router.enabled': 'Router włączony',
  'router.threshold': 'Próg decyzji',
  'router.tool_threshold': 'Próg wyboru narzędzia',
  'router.history_turns': 'Tury historii dla routera',
  'router.timeout_s': 'Limit czasu (s)',
  'router.prompt_path': 'Plik promptu routera',

  'style.auto_emotion': 'Automatyczna emocja głosu',
  'style.pitch_shift': 'Zmiana wysokości głosu',

  'tools.enabled': 'Narzędzia włączone',
  'tools.default_location': 'Domyślna lokalizacja',
  'tools.timeout_s': 'Limit czasu narzędzia (s)',

  'text.normalize_tts': 'Normalizuj tekst przed syntezą',

  'hermes.backend': 'Backend Hermesa',
  'hermes.command': 'Polecenie uruchamiające',
  'hermes.cwd': 'Katalog roboczy',
  'hermes.timeout_s': 'Limit czasu zadania (s)',
  'hermes.stub_delay_s': 'Opóźnienie atrapy (s)',
  'hermes.direct_max_chars': 'Maks. długość odpowiedzi bezpośredniej (znaki)',
  'hermes.progress_interval_s': 'Odstęp raportów postępu (s)',

  'metrics.turns_log': 'Plik dziennika tur',
  'metrics.window': 'Okno statystyk (liczba tur)',

  'wakeword.recordings_dir': 'Katalog nagrań próbek',
  'wakeword.max_total_mb': 'Limit rozmiaru nagrań (MB)',

  'devices.enabled': 'Nasłuch urządzeń włączony',
  'devices.listen_host': 'Adres nasłuchu urządzeń',
  'devices.port': 'Port urządzeń',
  'devices.tokens_file': 'Plik tokenów urządzeń',
  'devices.allowed_networks': 'Dozwolone sieci',
  'devices.max_connections': 'Maks. liczba połączeń',
  'devices.hello_timeout_s': 'Limit czasu powitania (s)',
  'devices.tls_certfile': 'Certyfikat TLS',
  'devices.tls_keyfile': 'Klucz TLS',
  'devices.verify_wake': 'Weryfikuj wybudzenie w STT',
  'devices.verify_ms': 'Czas weryfikacji (ms)',
  'devices.verify_hotwords': 'Słowa kluczowe weryfikacji',
  'devices.accept_bare_name': 'Akceptuj samo imię „Gaja"',
  'devices.extra_wake_names': 'Dodatkowe imiona wybudzające',
  'devices.extra_wake_greetings': 'Dodatkowe powitania wybudzające',
  'devices.wake_timeout_s': 'Limit czasu po wybudzeniu (s)',
  'devices.listen_timeout_s': 'Limit czasu nasłuchu (s)',
  'devices.barge_in_min_score': 'Minimalny wynik przerwania',
  'devices.ack': 'Potwierdzenie wybudzenia',
  'devices.ack_text': 'Tekst potwierdzenia (według języka)',
  'devices.follow_up_ms': 'Okno dopowiedzenia (ms)',
  'devices.listen_grace_s': 'Zwłoka nasłuchu (s)',
  'devices.preferred_out_rate': 'Częstotliwość wyjścia audio (Hz)',
  'devices.out_chunk_ms': 'Rozmiar porcji audio (ms)',

  'vocab.enabled': 'Słownik włączony',
  'vocab.default_path': 'Plik słownika domyślnego',
  'vocab.user_path': 'Plik słownika użytkownika',
  'vocab.suggestions_path': 'Plik podpowiedzi',
  'vocab.max_hotwords': 'Maks. liczba słów kluczowych',
  'vocab.fuzzy': 'Dopasowanie przybliżone',

  'admin.enabled': 'Panel włączony',
  'admin.credentials_file': 'Plik z hasłem panelu',
  'admin.session_hours': 'Czas sesji (godziny)',
};

// Labels of the values of choice fields; unknown values are shown as they are.
export const CHOICE_LABELS = {
  'llm.tool_strategy': { native: 'native (serwer parsuje wywołania)', pythonic: 'pythonic (parsowanie z treści)' },
  'stt.default_language': { pl: 'polski', en: 'angielski' },
  'stt.engine': { whistle: 'whistle', 'faster-whisper': 'faster-whisper' },
  'hermes.backend': { acp: 'acp (prawdziwy Hermes)', stub: 'stub (atrapa)' },
  'devices.ack': { spoken: 'głosowe („Słucham?")', earcon: 'dźwięk', none: 'brak' },
};

export function sectionTitle(section) {
  return SECTION_TITLES[section] || section;
}

export function fieldLabel(path) {
  return FIELD_LABELS[path] || path.split('.').pop().replace(/_/g, ' ');
}

export function choiceLabel(path, value) {
  return (CHOICE_LABELS[path] && CHOICE_LABELS[path][value]) || String(value);
}

// Polish labels of the household section (web/admin/sections/household.js).
export const HOUSEHOLD_LABELS = {
  title: 'Domownicy',
  lede: 'Kto mieszka w domu: jakie profile Hermesa może używać każda osoba i jakim głosem Gaja odpowiada.',
  refresh: 'Odśwież',
  addPerson: 'Dodaj osobę',
  save: 'Zapisz zmiany',
  discard: 'Odrzuć',
  saving: 'Zapisuję…',
  savedCount: (n) => `Zapisano zmiany dla domu (${n}).`,
  loadError: 'Nie udało się wczytać domowników',
  unavailable: 'Dom nie jest wczytany (brak pliku household).',
  // unsaved-changes guard
  leaveTitle: 'Odrzucić niezapisane zmiany?',
  leaveBody: 'Masz niezapisane zmiany w domownikach. Wyjście z tej sekcji je odrzuci.',
  leaveConfirm: 'Odrzuć i wyjdź',
  refreshBody: 'Masz niezapisane zmiany w domownikach. Odświeżenie wczyta zapisany stan i je odrzuci.',
  refreshConfirm: 'Odrzuć i odśwież',
  // general block
  generalHead: 'Gdy Gaja nie wie, kto mówi',
  generalHelp: 'Profil i głos, których Gaja używa, gdy nie rozpoznaje, kto mówi.',
  generalProfile: 'Profil ogólny',
  generalVoice: 'Głos odpowiedzi',
  voiceDefault: 'domyślny (według języka)',
  voicePerPerson: 'głos ogólny',
  voiceOnly: (langs) => `tylko ${langs.map((l) => l.toUpperCase()).join(', ')}`,
  voicePartial: (langs) => `Tylko ${langs.map((l) => l.toUpperCase()).join(', ')}: w innym języku odpowie głos domyślny.`,
  voiceNotLoaded: 'Niewczytany: pierwsza odpowiedź tym głosem potrwa dłużej.',
  // person card
  name: 'Imię',
  personId: 'Identyfikator',
  defaultProfile: 'Profil domyślny',
  extraProfiles: 'Dostępne profile',
  voice: 'Głos',
  sensitiveHint: 'Czuły: tylko wpisywanie z telefonu tej osoby, nigdy przez głos.',
  sensitiveBadge: 'czuły',
  phones: 'Sparowane telefony',
  unassignedPhones: 'Telefony bez osoby',
  noPhones: 'Żaden telefon nie jest jeszcze przypisany.',
  unbind: 'Rozłącz',
  devices: 'Urządzenia w domu',
  noDevices: 'Brak zarejestrowanych urządzeń głosowych.',
  unassignedDevice: 'nieprzypisane',
  removePerson: 'Usuń osobę',
  removeConfirmTitle: (name) => `Usunąć ${name}?`,
  removeConfirmBody: 'Gaja przestanie rozpoznawać tę osobę; historia i pamięć pozostaną na serwerze.',
  removeConfirm: 'Usuń',
  newPersonName: 'Np. Ala',
  idFromName: 'Identyfikator tworzy się z imienia; może zawierać małe litery, cyfry, myślniki.',
  // validation
  nameRequired: 'Podaj imię każdego domownika.',
  idInvalid: 'Identyfikator może zawierać tylko małe litery bez polskich znaków, cyfry, „-” i „_”.',
  idTaken: (id) => `Identyfikator „${id}” już jest używany.`,
  saveRejected: 'Serwer odrzucił zmiany; nic nie zostało zapisane.',
};
