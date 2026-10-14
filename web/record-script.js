// The recording script for /record.html. Edit this file to change what is asked of the speaker.
//
// Section  { id, label, title, intro, ambient? , prompts }
//   label   what the server stores it under: 'wake' | 'hard_negative' | 'ambient' (folder name, see gaja/wakeword_samples.py)
// Prompt   { id, say, how, target }
//   id      [a-z0-9_-], stable: it is the folder name and the key progress is counted by - do not rename after recording
//   say     the words to say (shown big); for ambient prompts a short title of the scene
//   how     how to say it (one sentence)
//   target  how many takes; for ambient prompts the target length in seconds (one take = up to MAX_AMBIENT_S)
//   continuous  (ambient only) no MAX_AMBIENT_S cap: records until Stop, every AMBIENT_CHUNK_S chunk is its own take

export const MAX_TAKE_S = 12; // a wake/hard-negative take stops by itself after this long
export const MAX_AMBIENT_S = 600; // an ambient take stops by itself after this long
export const AMBIENT_CHUNK_S = 60; // ambient audio is uploaded in chunks of this length

const w = (id, say, how, target, extra) => ({ id, say, how, target, ...extra });

export const SECTIONS = [
  {
    id: 'wake',
    label: 'wake',
    title: 'Pozytywy',
    intro: 'Słowo kluczowe, tak jak naprawdę będziesz je mówić.',
    prompts: [
      w('wake_normal', 'Hej Gaja', 'Normalnym głosem, jak do domownika.', 15),
      w('wake_vocative', 'Hej Gajo', 'Normalnym głosem, wołacz.', 10),
      w('wake_fast', 'Hej Gaja', 'Szybko, jednym tchem.', 8),
      w('wake_slow', 'Hej Gaja', 'Wolno, wyraźnie, z przerwą między słowami.', 8),
      w('wake_long_hej', 'Heeej Gaja', 'Przeciągnij „hej”.', 6),
      w('wake_long_gaja', 'Hej Gaaaja', 'Przeciągnij „a” w środku imienia.', 6),
      w('wake_long_end', 'Hej Gajaaa', 'Przeciągnij końcówkę.', 4),
      w('wake_long_voc', 'Hej Gajooo', 'Przeciągnij końcówkę wołacza.', 4),
      w('wake_whisper', 'Hej Gaja', 'Cicho, półszeptem.', 6),
      w('wake_far', 'Hej Gaja', 'Głośno, z drugiego końca pokoju (2–4 m od telefonu/mikrofonu).', 10),
      w('wake_media', 'Hej Gaja', 'Z włączonym telewizorem lub muzyką w tle.', 8),
      w('wake_s1', 'Hej Gaja, jaka jest pogoda?', 'Jak zwykłe pytanie, jednym ciągiem.', 1),
      w('wake_s2', 'Hej Gaja, która jest godzina?', 'Jak zwykłe pytanie, jednym ciągiem.', 1),
      w('wake_s3', 'Hej Gaja, włącz muzykę.', 'Jak polecenie.', 1),
      w('wake_s4', 'Hej Gaja, ustaw minutnik na dziesięć minut.', 'Jak polecenie.', 1),
      w('wake_s5', 'Hej Gajo, powiedz mi coś ciekawego.', 'Wołacz, swobodnie.', 1),
      w('wake_s6', 'Hej Gaja, zapytaj Hermesa o mój kalendarz.', 'Jak polecenie.', 1),
      w('wake_s7', 'Hej Gaja, zgaś światło w kuchni.', 'Jak polecenie.', 1),
      w('wake_s8', 'Hej Gaja, co mamy dzisiaj w planie?', 'Jak zwykłe pytanie.', 1),
      w('wake_sleepy', 'Hej Gaja', 'Zaspanym, zmęczonym głosem.', 4),
      w('wake_english', 'Hey Gaja', 'Po angielsku („hej” jak „hey”).', 6),
      w('wake_busy', 'Hej Gaja', 'Jedząc / z pełnymi ustami albo w ruchu (chodząc po pokoju).', 4),
    ],
  },
  {
    id: 'hard_negative',
    label: 'hard_negative',
    title: 'Trudne negatywy',
    intro: 'Podobne frazy, które NIE mogą obudzić Gai. Mów je naturalnie, tak jak brzmiałyby w rozmowie.',
    prompts: [
      w('hn_hej_maja', 'Hej Maja', 'Normalnie.', 3),
      w('hn_hej_kaja', 'Hej Kaja', 'Normalnie.', 3),
      w('hn_hej_daj', 'Hej, daj', 'Jak „hej, daj mi to”.', 3),
      w('hn_hej_gadaj', 'Hej, gadaj', 'Normalnie.', 3),
      w('hn_hej_gosia', 'Hej Gosia', 'Normalnie.', 3),
      w('hn_hej_ja', 'Hej ja', 'Normalnie.', 3),
      w('hn_okej', 'Okej', 'Normalnie.', 3),
      w('hn_gaja_imie', 'Gaja to ładne imię', 'Samo imię w zdaniu, bez „hej”. Nie może obudzić.', 3),
      w('hn_gaj', 'Gaj za domem', 'Normalnie.', 3),
      w('hn_hejka', 'Hejka', 'Normalnie.', 3),
      w('hn_daj_ja', 'Daj ja', 'Normalnie.', 3),
      w('hn_hey_guys', 'Hey guys', 'Po angielsku.', 3),
      w('hn_hey_siri', 'Hey Siri', 'Po angielsku.', 3),
      w('hn_hej_gala', 'Hej Gala', 'Normalnie.', 3),
      w('hn_hej_saja', 'Hej Saja', 'Normalnie.', 3),
      w('hn_hej_haja', 'Hej Haja', 'Normalnie.', 3),
      w('hn_ej_maja', 'Ej, Maja', 'Normalnie.', 3),
      w('hn_hej_gej', 'Hej gej', 'Jak cytat z telewizji, normalnie.', 3),
      w('hn_ale_gaja', 'Ale gaja', 'Normalnie.', 3),
      w('hn_freeform', 'Powiedz coś podobnego, co mogłoby się pomylić', 'Dowolna fraza brzmiąca jak „Hej Gaja”, ale nią nie będąca. Za każdym razem inna.', 5),
    ],
  },
  {
    id: 'ambient',
    label: 'ambient',
    ambient: true,
    title: 'Tło',
    intro: 'Zwykły pokój bez słowa kluczowego. Włącz nagranie i rób swoje; do 10 minut (bez limitu w „Nagrywaniu ciągłym”), wysyłane co 60 s.',
    prompts: [
      w('amb_tv', 'Telewizor', 'Włączony film lub serial, normalna głośność.', MAX_AMBIENT_S),
      w('amb_music', 'Muzyka', 'Muzyka w pokoju, najlepiej z wokalem.', MAX_AMBIENT_S),
      w('amb_talk', 'Rozmowa', 'Rozmawiaj normalnie (lub z kimś), bez „Hej Gaja”.', MAX_AMBIENT_S),
      w('amb_kitchen', 'Kuchnia', 'Szum wody, naczynia, wyciąg, krzątanie się.', MAX_AMBIENT_S),
      w('amb_quiet', 'Cichy pokój', 'Cisza domu: kroki, klawiatura, szelesty.', MAX_AMBIENT_S),
      w('amb_mix', 'Zwykły dzień', 'Mieszanka: telewizor, rozmowa, kuchnia.', MAX_AMBIENT_S),
      w('amb_continuous', 'Nagrywanie ciągłe', 'Zostaw włączone na godziny lub dni: telewizor, rozmowy, zwykłe życie. Uprzedź domowników, że mikrofon nagrywa.', 3600, { continuous: true }),
    ],
  },
];
