from __future__ import annotations

import pytest
from gaja.text.normalize import normalize_for_tts


class TestNormalizePolish:
    """Test Polish text normalization for speech synthesis."""

    def test_temperature_celsius(self):
        assert "dwanaście stopni Celsjusza" in normalize_for_tts("12°C", "pl")
        assert "minus pięć stopni Celsjusza" in normalize_for_tts("-5°C", "pl")
        assert "jeden stopień Celsjusza" in normalize_for_tts("1°C", "pl")
        assert "dwa stopnie Celsjusza" in normalize_for_tts("2°C", "pl")

    def test_temperature_decimals(self):
        res = normalize_for_tts("10.8°C", "pl")
        assert "dziesięć przecinek osiem" in res
        assert "Celsjusza" in res

    def test_standalone_degrees(self):
        assert "dwanaście stopni" in normalize_for_tts("12°", "pl")

    def test_time_nominative(self):
        res = normalize_for_tts("06:54", "pl")
        assert "szósta pięćdziesiąt cztery" in res

    def test_time_with_godzina(self):
        res = normalize_for_tts("godzina 09:46", "pl")
        assert "dziewiąta czterdzieści sześć" in res

    def test_time_locative_after_o(self):
        res = normalize_for_tts("Spotkanie o 17:59 w biurze", "pl")
        assert "o siedemnastej pięćdziesiąt dziewięć" in res

    def test_time_locative_o_morning(self):
        res = normalize_for_tts("o 06:54 wschód słońca", "pl")
        assert "o szóstej pięćdziesiąt cztery" in res

    def test_date_ordinal_genitive(self):
        res = normalize_for_tts("9 października", "pl")
        assert "dziewiątego października" in res

    def test_date_full_with_year(self):
        res = normalize_for_tts("9 października 2026 r.", "pl")
        assert "dziewiątego października dwa tysiące dwudziestego szóstego roku" in res

    def test_date_first_of_month(self):
        res = normalize_for_tts("1 listopada", "pl")
        assert "pierwszego listopada" in res

    def test_year_standalone(self):
        res = normalize_for_tts("w 1990 roku", "pl")
        assert "tysiąc dziewięćset dziewięćdziesiątym roku" in res

    def test_currency_pln(self):
        res = normalize_for_tts("4,38 zł", "pl")
        assert "cztery złote i trzydzieści osiem groszy" in res

    def test_currency_pln_whole(self):
        assert "dziesięć złotych" in normalize_for_tts("10 zł", "pl")
        assert "jeden złoty" in normalize_for_tts("1 zł", "pl")
        assert "dwa złote" in normalize_for_tts("2 zł", "pl")

    def test_currency_eur(self):
        res = normalize_for_tts("100 EUR", "pl")
        assert "sto euro" in res

    def test_currency_usd(self):
        res = normalize_for_tts("50 USD", "pl")
        assert "pięćdziesiąt dolarów" in res
        assert "jeden dolar" in normalize_for_tts("1 USD", "pl")

    def test_currency_gbp(self):
        res = normalize_for_tts("20 GBP", "pl")
        assert "dwadzieścia funtów" in res

    def test_units_speed(self):
        res = normalize_for_tts("100 km/h", "pl")
        assert "sto kilometrów na godzinę" in res
        res2 = normalize_for_tts("25 m/s", "pl")
        assert "dwadzieścia pięć metrów na sekundę" in res2

    def test_units_mass_distance(self):
        assert "pięćdziesiąt kilometrów" in normalize_for_tts("50 km", "pl")
        assert "dwa kilogramy" in normalize_for_tts("2 kg", "pl")
        assert "pięć kilogramów" in normalize_for_tts("5 kg", "pl")

    def test_units_memory(self):
        res = normalize_for_tts("16 GB RAM", "pl")
        assert "szesnaście gigabajtów" in res

    def test_percentages(self):
        res = normalize_for_tts("15%", "pl")
        assert "piętnaście procent" in res

    def test_percentages_decimal(self):
        res = normalize_for_tts("12.5%", "pl")
        assert "dwanaście przecinek pięć procent" in res

    def test_abbreviations(self):
        assert "na przykład" in normalize_for_tts("np. jabłka", "pl")
        assert "to znaczy" in normalize_for_tts("tzn. tak", "pl")
        assert "między innymi" in normalize_for_tts("m.in. autor", "pl")
        assert "około" in normalize_for_tts("ok. dziesięciu", "pl")
        assert "świętego" in normalize_for_tts("św. Jan", "pl")
        assert "doktor" in normalize_for_tts("dr Kowalski", "pl")
        assert "ulica" in normalize_for_tts("ul. Polna", "pl")

    def test_century_roman_locative(self):
        res = normalize_for_tts("w XVI wieku", "pl")
        assert "w szesnastym wieku" in res

    def test_century_roman_nominative(self):
        res = normalize_for_tts("XXI wiek", "pl")
        assert "dwudziesty pierwszy wiek" in res

    def test_decimals(self):
        assert "dziesięć przecinek osiem" in normalize_for_tts("10.8", "pl")
        assert "dwa przecinek pięć" in normalize_for_tts("2,5", "pl")

    def test_negative_numbers(self):
        res = normalize_for_tts("W nocy -15 stopni", "pl")
        assert "minus piętnaście" in res

    def test_large_numbers_with_spaces(self):
        res = normalize_for_tts("1 000 000 osób", "pl")
        assert "milion" in res

    def test_cardinal_integers(self):
        assert "dwieście czterdzieści" in normalize_for_tts("240", "pl")


class TestNormalizeEnglish:
    """Test English text normalization for speech synthesis."""

    def test_en_temperature(self):
        res = normalize_for_tts("The temp is 12°C and 72°F.", "en")
        assert "twelve degrees Celsius" in res
        assert "seventy-two degrees Fahrenheit" in res

    def test_en_time(self):
        res = normalize_for_tts("The flight is at 06:54 and 17:59.", "en")
        assert "six fifty-four" in res
        assert "seventeen fifty-nine" in res

    def test_en_currency(self):
        res = normalize_for_tts("The cost is $50 or €100.", "en")
        assert "fifty dollars" in res
        assert "one hundred euros" in res

    def test_en_currency_code(self):
        res = normalize_for_tts("Send 20 GBP.", "en")
        assert "twenty pounds" in res

    def test_en_units(self):
        res = normalize_for_tts("Speed limit 65 mph or 100 km/h.", "en")
        assert "sixty-five miles per hour" in res
        assert "one hundred kilometers per hour" in res

    def test_en_mass(self):
        res = normalize_for_tts("Weight 10 kg.", "en")
        assert "ten kilograms" in res

    def test_en_percentage(self):
        res = normalize_for_tts("Success rate is 99%.", "en")
        assert "ninety-nine percent" in res

    def test_en_decimals(self):
        res = normalize_for_tts("Value is 10.8.", "en")
        assert "ten point eight" in res

    def test_en_abbreviations(self):
        res = normalize_for_tts("We offer e.g. fruits, i.e. fresh ones, etc.", "en")
        assert "for example" in res
        assert "that is" in res
        assert "et cetera" in res

    def test_en_negative_numbers(self):
        res = normalize_for_tts("It is -5 outside.", "en")
        assert "minus five" in res


class TestNormalizeRobustness:
    """Test safety and performance under edge conditions."""

    def test_empty_string(self):
        assert normalize_for_tts("") == ""
        assert normalize_for_tts("   ") == ""

    def test_plain_text_untouched(self):
        assert normalize_for_tts("Cześć Gaja!", "pl") == "Cześć Gaja!"
        assert normalize_for_tts("Hello there!", "en") == "Hello there!"

    def test_never_crashes_on_garbage(self):
        garbage = "!@#$%^&*()_+=-~`{}[]|:;'<>,.?/"
        res = normalize_for_tts(garbage, "pl")
        assert isinstance(res, str)

    def test_mixed_sentence_pl(self):
        text = "Jutro w Krakowie od 10 do 16 stopni, o 06:54 wschód słońca, wiatr 14 km/h."
        res = normalize_for_tts(text, "pl")
        assert "od dziesięciu do szesnastu stopni" in res
        assert "o szóstej pięćdziesiąt cztery" in res
        assert "czternaście kilometrów na godzinę" in res


class TestPolishInflectionCases:
    """Exact-string tests for Polish grammatical case inflection, noun agreement, and capitalization."""

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("Od 11 do 17 stopni.", "Od jedenastu do siedemnastu stopni."),
            ("od 11 do 17 stopni", "od jedenastu do siedemnastu stopni"),
            ("od 7 dni", "od siedmiu dni"),
            ("do 5 lat", "do pięciu lat"),
            ("z 3 metrów", "z trzech metrów"),
            ("około 10 kilometrów", "około dziesięciu kilometrów"),
            ("ok. 50 osób", "około pięćdziesięciu osób"),
            ("bez 2 sekund", "bez dwóch sekund"),
            ("dla 1 osoby", "dla jednej osoby"),
            ("o 17:59", "o siedemnastej pięćdziesiąt dziewięć"),
            ("O 08:30", "O ósmej trzydzieści"),
            ("2 godziny i 21 minut", "dwie godziny i dwadzieścia jeden minut"),
            ("1 godzina", "jedna godzina"),
            ("21 godzin", "dwadzieścia jeden godzin"),
            ("31 minut", "trzydzieści jeden minut"),
            ("101 minut", "sto jeden minut"),
            ("1 sekunda", "jedna sekunda"),
            ("21 sekund", "dwadzieścia jeden sekund"),
            ("1 euro", "jedno euro"),
            ("2 euro", "dwa euro"),
            ("21 euro", "dwadzieścia jeden euro"),
            ("1 złoty", "jeden złoty"),
            ("2 złote", "dwa złote"),
            ("5 złotych", "pięć złotych"),
            ("21 złotych", "dwadzieścia jeden złotych"),
            ("1 dolar", "jeden dolar"),
            ("2 dolary", "dwa dolary"),
            ("5 dolarów", "pięć dolarów"),
            ("21 dolarów", "dwadzieścia jeden dolarów"),
            ("2 godziny", "dwie godziny"),
            ("22 godziny", "dwadzieścia dwie godziny"),
            ("1 kilometr", "jeden kilometr"),
            ("2 kilometry", "dwa kilometry"),
            ("5 kilometrów", "pięć kilometrów"),
            ("1 litr", "jeden litr"),
            ("2 litry", "dwa litry"),
            ("5 litrów", "pięć litrów"),
            ("1 mila", "jedna mila"),
            ("2 mile", "dwie mile"),
            ("5 mil", "pięć mil"),
            ("1 kilogram", "jeden kilogram"),
            ("2 kilogramy", "dwa kilogramy"),
            ("5 kilogramów", "pięć kilogramów"),
            ("1 metr", "jeden metr"),
            ("2 metry", "dwa metry"),
            ("5 metrów", "pięć metrów"),
            ("1 dzień", "jeden dzień"),
            ("3 dni", "trzy dni"),
            ("5 dni", "pięć dni"),
            ("1 stopień", "jeden stopień"),
            ("3 stopnie", "trzy stopnie"),
            ("5 stopni", "pięć stopni"),
            ("1 procent", "jeden procent"),
            ("3 procent", "trzy procent"),
            ("5 procent", "pięć procent"),
            ("1 rok", "jeden rok"),
            ("3 lata", "trzy lata"),
            ("5 lat", "pięć lat"),
            ("1 gigabajt", "jeden gigabajt"),
            ("3 gigabajty", "trzy gigabajty"),
            ("5 gigabajtów", "pięć gigabajtów"),
            ("1 megabajt", "jeden megabajt"),
            ("3 megabajty", "trzy megabajty"),
            ("5 megabajtów", "pięć megabajtów"),
            ("12,6 z 15,5 GB", "dwanaście przecinek sześć z piętnastu przecinek pięciu gigabajtów"),
        ],
    )
    def test_exact_inflection_cases(self, raw: str, expected: str):
        assert normalize_for_tts(raw, "pl") == expected


class TestNegativeDecimalsAndPercent:
    """F2: dziesiętne były zamieniane na słowa przed odczytaniem znaku, więc "-10,8 stopnia"
    mówiło "-dziesięć przecinek osiem stopnia", a "-2,5%" gubiło minus."""

    def test_negative_decimal_temperature_comma(self):
        assert normalize_for_tts("Temperatura -10,8 stopnia.", "pl") == "Temperatura minus dziesięć przecinek osiem stopnia."

    def test_negative_decimal_temperature_dot(self):
        # the exact string home_tool._format_temp_pl produces for -2.5 °C
        assert normalize_for_tts("Jest -2.5 stopnia.", "pl") == "Jest minus dwa przecinek pięć stopnia."

    def test_negative_percent_decimal(self):
        assert normalize_for_tts("Spadło o -2,5%", "pl") == "Spadło o minus dwa przecinek pięć procent"

    def test_negative_percent_integer(self):
        assert normalize_for_tts("Spadło o -5%", "pl") == "Spadło o minus pięć procent"

    def test_english_negative_decimal(self):
        assert normalize_for_tts("Value is -2.5.", "en") == "Value is minus two point five."


class TestTimePrepositions:
    """F3: tylko o/od/do odmieniały godzinę; "przed 18:30" i "między 17:00 a 18:00" zostawały
    w mianowniku, a "po"/"około" działały tylko przez przypadkowe dopasowanie niezakotwiczonego
    'o\\s+' wewnątrz wyrazu."""

    def test_przed_takes_instrumental(self):
        assert normalize_for_tts("Wracam przed 18:30.", "pl") == "Wracam przed osiemnastą trzydzieści."

    def test_miedzy_a_takes_instrumental_on_both_sides(self):
        assert normalize_for_tts("między 17:00 a 18:00", "pl") == "między siedemnastą a osiemnastą"

    def test_miedzy_with_minutes(self):
        assert (
            normalize_for_tts("między 17:30 a 18:45", "pl")
            == "między siedemnastą trzydzieści a osiemnastą czterdzieści pięć"
        )

    def test_po_stays_locative_without_the_o_trick(self):
        assert normalize_for_tts("Będę w domu po 17:00.", "pl") == "Będę w domu po siedemnastej."

    def test_okolo_takes_genitive(self):
        assert normalize_for_tts("Przyjdę około 17:15.", "pl") == "Przyjdę około siedemnastej piętnaście."

    def test_przed_po_polnocy(self):
        assert normalize_for_tts("Wracam przed 00:30.", "pl") == "Wracam przed północą trzydzieści."

    def test_spelled_out_godzinie_is_kept(self):
        # the sun/timer tools emit "o godzinie HH:MM"; F3 must not swallow "godzinie"
        assert normalize_for_tts("Wschód o godzinie 06:25.", "pl") == (
            "Wschód o godzinie szóstej dwadzieścia pięć."
        )


class TestNormalizePhoneNumbers:
    """Test phone number recognition and formatting in Polish."""

    def test_phone_333_spaces(self):
        res = normalize_for_tts("Telefon 600 700 800.", "pl")
        assert res == "Telefon sześćset siedemset osiemset."

    def test_phone_333_prefix_plus48(self):
        res = normalize_for_tts("Infolinia pod numerem +48 600 700 800 działa całą dobę.", "pl")
        assert res == "Infolinia pod numerem plus czterdzieści osiem sześćset siedemset osiemset działa całą dobę."

    def test_phone_333_dashes(self):
        res = normalize_for_tts("Numer: 600-700-800", "pl")
        assert res == "Numer: sześćset siedemset osiemset"

    def test_phone_322_dashes(self):
        res = normalize_for_tts("Zadzwoń pod 123-45-67 do biura.", "pl")
        assert res == "Zadzwoń pod sto dwadzieścia trzy czterdzieści pięć sześćdziesiąt siedem do biura."

    def test_phone_322_with_area_code_and_plus48(self):
        res = normalize_for_tts("Kontakt: +48 22 123-45-67.", "pl")
        assert res == "Kontakt: plus czterdzieści osiem dwadzieścia dwa sto dwadzieścia trzy czterdzieści pięć sześćdziesiąt siedem."

    def test_thousands_formatting_preserved_with_currency_and_units(self):
        # A single space-separated thousands group followed by a unit or currency is still a number
        assert normalize_for_tts("12 500 zł", "pl") == "dwanaście tysięcy pięćset złotych"
        assert normalize_for_tts("1 299,99 zł", "pl") == "tysiąc dwieście dziewięćdziesiąt dziewięć złotych i dziewięćdziesiąt dziewięć groszy"
        assert normalize_for_tts("1 000 000 osób", "pl") == "milion osób"
        assert normalize_for_tts("100 000 000 zł", "pl") == "sto milionów złotych"

    def test_to_okolo_nominative(self):
        res = normalize_for_tts("Dystans to około 62 mile.", "pl")
        assert res == "Dystans to około sześćdziesiąt dwie mile."

        res2 = normalize_for_tts("Dystans to około 10 kilometrów.", "pl")
        assert res2 == "Dystans to około dziesięć kilometrów."

        res3 = normalize_for_tts("sto kilometrów to około sześćdziesiąt dwie i jedna dziesiąta mili.", "pl")
        assert res3 == "sto kilometrów to około sześćdziesiąt dwie i jedna dziesiąta mili."


class TestTimeOClockDropZero:
    """Tests for dropping :00 in time expressions."""

    def test_o_10_00_mam_spotkanie(self):
        res = normalize_for_tts("O 10:00 mam spotkanie.", "pl")
        assert res == "O dziesiątej mam spotkanie."

    def test_jest_10_00(self):
        res = normalize_for_tts("Jest 10:00", "pl")
        assert res == "Jest dziesiąta"

    def test_od_12_00_do_13_30(self):
        res = normalize_for_tts("Od 12:00 do 13:30.", "pl")
        assert res == "Od dwunastej do trzynastej trzydzieści."

    def test_od_12_00_do_13_00(self):
        res = normalize_for_tts("Od 12:00 do 13:00.", "pl")
        assert res == "Od dwunastej do trzynastej."

    def test_plain_hh_00(self):
        assert normalize_for_tts("10:00", "pl") == "dziesiąta"
        assert normalize_for_tts("12:00", "pl") == "dwunasta"
        assert normalize_for_tts("00:00", "pl") == "północ"
        assert normalize_for_tts("o 00:00", "pl") == "o północy"

    def test_keep_minutes_when_nonzero(self):
        assert normalize_for_tts("7:05", "pl") == "siódma pięć"
        assert normalize_for_tts("07:05", "pl") == "siódma pięć"
        assert normalize_for_tts("o 7:05", "pl") == "o siódmej pięć"
        assert normalize_for_tts("17:59", "pl") == "siedemnasta pięćdziesiąt dziewięć"


class TestPhoneVersusLargeNumber:
    """F4: _sub_phone ran before the large-number collapse and only a fixed noun list kept a
    spaced number from being read as phone groups. A phone now needs a phone cue (tel/telefon/
    numer/zadzwoń/+48) or the bare 3-3-3 shape with a non-zero leading group and no following
    word; a thousands group of zeros marks a magnitude and is read as a number."""

    def test_spaced_magnitude_reads_as_words(self):
        assert normalize_for_tts("Na Ziemi jest 1 412 000 000 ludzi.", "pl") == (
            "Na Ziemi jest miliard czterysta dwanaście milionów ludzi."
        )

    def test_nine_digit_magnitude_before_a_noun(self):
        assert normalize_for_tts("W mieście mieszka 600 700 800 ludzi.", "pl") == (
            "W mieście mieszka sześćset milionów siedemset tysięcy osiemset ludzi."
        )

    def test_eleven_digit_number_is_not_a_phone(self):
        assert normalize_for_tts("Liczba to 12 345 678 901.", "pl") == (
            "Liczba to dwanaście miliardów trzysta czterdzieści pięć milionów "
            "sześćset siedemdziesiąt osiem tysięcy dziewięćset jeden."
        )

    def test_bare_three_groups_stay_a_phone(self):
        assert normalize_for_tts("123 456 789", "pl") == (
            "sto dwadzieścia trzy czterysta pięćdziesiąt sześć siedemset osiemdziesiąt dziewięć"
        )

    def test_phone_cue_still_reads_digit_groups(self):
        assert normalize_for_tts("Mój numer to 123 456 789.", "pl") == (
            "Mój numer to sto dwadzieścia trzy czterysta pięćdziesiąt sześć "
            "siedemset osiemdziesiąt dziewięć."
        )


class TestYearGrammarDeclension:
    """Tests for Polish year grammatical case: locative after w/we/po/o vs genitive after od/do/dates."""

    def test_w_1990_roku(self):
        res = normalize_for_tts("W 1990 roku.", "pl")
        assert res == "W tysiąc dziewięćset dziewięćdziesiątym roku."

    def test_w_2024_r_temperatura(self):
        res = normalize_for_tts("W 2024 r. było -5°C.", "pl")
        assert res == "W dwa tysiące dwudziestym czwartym roku było minus pięć stopni Celsjusza."

    def test_dates_remain_genitive(self):
        res1 = normalize_for_tts("15 maja 2024 r.", "pl")
        assert res1 == "piętnastego maja dwa tysiące dwudziestego czwartego roku"

        res2 = normalize_for_tts("dziewiątego października 2026 roku", "pl")
        assert res2 == "dziewiątego października dwa tysiące dwudziestego szóstego roku"

        res3 = normalize_for_tts("9 października 2026 r.", "pl")
        assert res3 == "dziewiątego października dwa tysiące dwudziestego szóstego roku"

    def test_od_do_year_genitive(self):
        assert normalize_for_tts("od 2020 roku", "pl") == "od dwa tysiące dwudziestego roku"
        assert normalize_for_tts("do 2020 roku", "pl") == "do dwa tysiące dwudziestego roku"

    def test_po_o_year_locative(self):
        assert normalize_for_tts("po 2020 roku", "pl") == "po dwa tysiące dwudziestym roku"
        assert normalize_for_tts("o 2020 roku", "pl") == "o dwa tysiące dwudziestym roku"

    def test_year_2000(self):
        assert normalize_for_tts("w 2000 roku", "pl") == "w dwutysięcznym roku"
        assert normalize_for_tts("od 2000 roku", "pl") == "od dwutysięcznego roku"


class TestStreetAndPlaceAbbreviations:
    """Tests for case-aware abbreviation expansion (ul., al., pl., os.)."""

    def test_locative_after_na_przy_w(self):
        assert normalize_for_tts("na ul. Polnej", "pl") == "na ulicy Polnej"
        assert normalize_for_tts("przy al. Niepodległości", "pl") == "przy alei Niepodległości"
        assert normalize_for_tts("na pl. Zbawiciela", "pl") == "na placu Zbawiciela"
        assert normalize_for_tts("na os. Kolorowym", "pl") == "na osiedlu Kolorowym"
        assert normalize_for_tts("w ul. Krótkiej", "pl") == "w ulicy Krótkiej"

    def test_genitive_after_z_do_od_spod_z_okolic(self):
        assert normalize_for_tts("z ul. Polnej", "pl") == "z ulicy Polnej"
        assert normalize_for_tts("do al. Mickiewicza", "pl") == "do alei Mickiewicza"
        assert normalize_for_tts("z pl. Bankowego", "pl") == "z placu Bankowego"
        assert normalize_for_tts("z os. Piastów", "pl") == "z osiedla Piastów"
        assert normalize_for_tts("spod os. Piastów", "pl") == "spod osiedla Piastów"
        assert normalize_for_tts("z okolic pl. Konstytucji", "pl") == "z okolic placu Konstytucji"
        assert normalize_for_tts("od ul. Długiej", "pl") == "od ulicy Długiej"

    def test_nominative_otherwise(self):
        assert normalize_for_tts("ul. Polna", "pl") == "ulica Polna"
        assert normalize_for_tts("Ul. Polna", "pl") == "Ulica Polna"
        assert normalize_for_tts("al. Jerozolimskie", "pl") == "aleja Jerozolimskie"
        assert normalize_for_tts("pl. Bankowy", "pl") == "plac Bankowy"
        assert normalize_for_tts("os. Kolorowe", "pl") == "osiedle Kolorowe"


class TestOtherAbbreviationsAndQuantities:
    """Tests for titles, quantities (tys., mln), and time abbreviations."""

    def test_dr_and_prof(self):
        assert normalize_for_tts("u dr Kowalskiego", "pl") == "u doktora Kowalskiego"
        assert normalize_for_tts("u dr. Kowalskiego", "pl") == "u doktora Kowalskiego"
        assert normalize_for_tts("do prof. Nowaka", "pl") == "do profesora Nowaka"
        assert normalize_for_tts("z prof. Nowakiem", "pl") == "z profesorem Nowakiem"
        assert normalize_for_tts("dr Kowalski", "pl") == "doktor Kowalski"
        assert normalize_for_tts("prof. Nowak", "pl") == "profesor Nowak"

    def test_tys_and_mln(self):
        assert normalize_for_tts("5 tys. zł", "pl") == "pięć tysięcy złotych"
        assert normalize_for_tts("5 mln zł", "pl") == "pięć milionów złotych"
        assert normalize_for_tts("1 mln zł", "pl") == "jeden milion złotych"
        assert normalize_for_tts("2 mln zł", "pl") == "dwa miliony złotych"
        assert normalize_for_tts("1 tys.", "pl") == "jeden tysiąc"
        assert normalize_for_tts("2 tys.", "pl") == "dwa tysiące"
        assert normalize_for_tts("5 tys.", "pl") == "pięć tysięcy"
        assert normalize_for_tts("1 mln", "pl") == "jeden milion"
        assert normalize_for_tts("5 mln", "pl") == "pięć milionów"

    def test_gr_stays_groszy(self):
        assert normalize_for_tts("15 gr", "pl") == "piętnaście groszy"
        assert normalize_for_tts("Reszta 15 gr.", "pl") == "Reszta piętnaście groszy."

    def test_mld_reads_miliard(self):
        assert normalize_for_tts("2 mld osób", "pl") == "dwa miliardy osób"
        assert normalize_for_tts("1 mld zł", "pl") == "jeden miliard złotych"
        assert normalize_for_tts("5 mld zł", "pl") == "pięć miliardów złotych"
        # a decimal follows the same genitive singular mln already uses ("... miliona złotych")
        assert normalize_for_tts("1,5 mld zł", "pl") == "jeden przecinek pięć miliarda złotych"


class TestBareNumericRanges:
    """F1: "od X do Y" bez jednostki nie może dorabiać "stopni", a zakres lat z "roku" musi
    mieć obie granice w ordinalnej formie dopełniacza (nie kardinalna + ordinalna mieszanina)."""

    def test_bare_range_has_no_invented_unit(self):
        assert normalize_for_tts("Wybierz liczbę od 5 do 10.", "pl") == "Wybierz liczbę od pięciu do dziesięciu."

    def test_bare_year_range_without_roku(self):
        assert (
            normalize_for_tts("od 2026 do 2027", "pl")
            == "od dwóch tysięcy dwudziestu sześciu do dwóch tysięcy dwudziestu siedmiu"
        )

    def test_year_range_with_roku_is_ordinal_genitive(self):
        assert (
            normalize_for_tts("od 2026 do 2027 roku", "pl")
            == "od dwa tysiące dwudziestego szóstego do dwa tysiące dwudziestego siódmego roku"
        )


class TestPolishNounCases:
    """polish_noun_cases: best-effort accusative/genitive of a Polish device label (F7)."""

    def test_feminine_head_with_prepositional_tail(self):
        from gaja.text.normalize import polish_noun_cases

        assert polish_noun_cases("klima u Ani") == ("klimę u Ani", "klimy u Ani")
        assert polish_noun_cases("klima w salonie") == ("klimę w salonie", "klimy w salonie")

    def test_feminine_nia_ka_ola_stems(self):
        from gaja.text.normalize import polish_noun_cases

        assert polish_noun_cases("sypialnia") == ("sypialnię", "sypialni")
        assert polish_noun_cases("drukarka") == ("drukarkę", "drukarki")
        assert polish_noun_cases("konsola") == ("konsolę", "konsoli")

    def test_adjective_phrase_matches_known_forms(self):
        from gaja.text.normalize import polish_noun_cases

        assert polish_noun_cases("ciepła woda") == ("ciepłą wodę", "ciepłej wody")

    def test_masculine_accusative_is_the_nominative_genitive_unknown(self):
        from gaja.text.normalize import polish_noun_cases

        assert polish_noun_cases("piec") == ("piec", "")
        assert polish_noun_cases("telewizor w salonie") == ("telewizor w salonie", "")
        assert polish_noun_cases("stary telewizor") == ("stary telewizor", "")

    def test_sc_noun(self):
        from gaja.text.normalize import polish_noun_cases

        assert polish_noun_cases("wilgotność") == ("wilgotność", "wilgotności")

    def test_indeclinable_tokens_stay_unchanged(self):
        from gaja.text.normalize import polish_noun_cases

        assert polish_noun_cases("ps5") == ("ps5", "ps5")

    def test_unknown_forms_come_back_empty(self):
        from gaja.text.normalize import polish_noun_cases

        assert polish_noun_cases("pokój Ani") == ("", "")
        assert polish_noun_cases("radio") == ("radio", "")
        assert polish_noun_cases("u Ani") == ("", "")
        assert polish_noun_cases("rodzice") == ("", "")
