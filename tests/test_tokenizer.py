from kws_mandarin.tokenizer import PinyinTokenizer, ToneMode
from kws_mandarin.tokenizer.pinyin import split_syllable, text_to_tone3


def test_split_syllable():
    assert split_syllable("zhong1") == ("zh", "ong", "1")
    assert split_syllable("wen2") == ("", "wen", "2")   # zero-initial (w glide stays in final)
    assert split_syllable("a1") == ("", "a", "1")
    assert split_syllable("shi4") == ("sh", "i", "4")
    assert split_syllable("lv3") == ("l", "v", "3")     # ü rendered as v by pypinyin
    assert split_syllable("de5") == ("d", "e", "5")      # neutral tone


def test_text_to_tone3_drops_non_han():
    assert text_to_tone3("中文abc123") == ["zhong1", "wen2"]


def test_final_mode_units():
    tok = PinyinTokenizer(ToneMode.FINAL)
    assert tok.units_for_text("中文") == ["zh", "ong1", "wen2"]


def test_separate_mode_units():
    tok = PinyinTokenizer(ToneMode.SEPARATE)
    assert tok.units_for_text("中") == ["zh", "ong", "1"]


def test_syllable_mode_units():
    tok = PinyinTokenizer(ToneMode.SYLLABLE)
    assert tok.units_for_text("中文") == ["zhong1", "wen2"]


def test_special_token_ids():
    tok = PinyinTokenizer(ToneMode.FINAL)
    assert tok.blank_id == 0
    assert tok.unk_id == 1
    assert tok.id_to_unit[0] == PinyinTokenizer.BLANK
    assert tok.id_to_unit[1] == PinyinTokenizer.UNK


def test_encode_ids_in_range_and_roundtrip_units():
    tok = PinyinTokenizer(ToneMode.FINAL)
    ids = tok.encode("你好世界")
    assert all(0 <= i < tok.vocab_size for i in ids)
    assert tok.decode_units(ids) == tok.units_for_text("你好世界")


def test_no_unk_on_common_text():
    # Every mode must cover common Han text with zero <unk> fallbacks.
    sample = "你好世界中文语音关键词识别打开空调播放音乐今天天气怎么样请帮我预订机票"
    for mode in ToneMode:
        tok = PinyinTokenizer(mode)
        ids = tok.encode(sample)
        assert tok.unk_id not in ids, f"<unk> produced in mode {mode.value}"


def test_neutral_tone_coverage():
    # Regression: contextual neutral tone (tone 5) must be in-vocab. Isolated-character
    # citation forms give 儿=er2 and 宜=yi2, but erhua and 便宜 need er5/yi5 — these were
    # missing until the vocab generator added tone-5 variants for every base.
    tok = PinyinTokenizer()
    assert "er5" in tok.unit_to_id and "yi5" in tok.unit_to_id
    for text in ("便宜", "这儿", "一点儿", "他们以挑刺儿的态度"):
        assert tok.unk_id not in tok.encode(text), f"<unk> for {text}: {tok.units_for_text(text)}"


def test_vocab_sizes_are_sane():
    sizes = {m: PinyinTokenizer(m).vocab_size for m in ToneMode}
    # separate (initials+finals+tones) << final (initials + toned-finals) << syllable
    assert sizes[ToneMode.SEPARATE] < sizes[ToneMode.FINAL] < sizes[ToneMode.SYLLABLE]
