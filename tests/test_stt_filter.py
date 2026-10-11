from gaja.stt.faster_whisper_stt import keep_segment


def test_ordinary_segment_is_kept() -> None:
    assert keep_segment(" Jaka jest pogoda w Krakowie?", 0.1, -0.3)


def test_confident_silence_segment_is_dropped() -> None:
    assert not keep_segment(" Dziękuję.", 0.9, -1.4)


def test_subtitle_hallucinations_are_dropped() -> None:
    assert not keep_segment(" Napisy stworzone przez społeczność Amara.org", 0.2, -0.5)
    assert not keep_segment(" Thanks for watching!", 0.2, -0.5)
    assert not keep_segment(" ...", 0.2, -0.5)


def test_high_no_speech_but_confident_text_is_kept() -> None:
    assert keep_segment(" Tak.", 0.7, -0.2)
