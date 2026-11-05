import math

import pytest

from transcript import Turn, build_transcript, format_time, validate_turns


def transcript(texts, turns, **kwargs):
    return build_transcript(texts=texts, turns=turns, duration=100, language="Chinese", **kwargs)


def test_conversation_format_and_returning_speaker_use_turn_timestamps():
    turns = [
        Turn(0, 0.95, "SPEAKER_07"),
        Turn(1, 21.5, "SPEAKER_02"),
        Turn(22, 34, "SPEAKER_05"),
        Turn(34, 34.8, "SPEAKER_02"),
        Turn(35, 35.4, "SPEAKER_05"),
        Turn(35.5, 36, "SPEAKER_02"),
    ]
    result = transcript(
        ["那咱这边，", "这个东西有人看吗？", "", "嗯。", "就行了。", "嗯嗯。"], turns
    )
    assert result["speaker_count"] == 3
    assert result["formatted_text"] == (
        "发言人1\n00:00\n那咱这边，\n\n"
        "发言人2\n00:01\n这个东西有人看吗？\n\n"
        "发言人2\n00:34\n嗯。\n\n"
        "发言人3\n00:35\n就行了。\n\n"
        "发言人2\n00:35\n嗯嗯。"
    )
    assert result["timestamp_mode"] == "segment" and result["words"] == []
    assert result["segments"][1]["end"] == 21.5


def test_same_speaker_merges_but_long_pause_splits():
    result = transcript(
        ["你，", "好。", "啊"],
        [Turn(0, 0.2, "A"), Turn(0.2, 0.5, "A"), Turn(5, 6, "A")],
    )
    assert [s["text"] for s in result["segments"]] == ["你，好。", "啊"]
    assert [(s["start"], s["end"]) for s in result["segments"]] == [(0, 0.5), (5, 6)]


def test_empty_recognition_of_another_speaker_still_prevents_merging():
    result = transcript(
        ["你好。", "", "", "再见。"],
        [Turn(0, 1, "A"), Turn(1, 1.1, "B"), Turn(1.1, 1.2, "A"), Turn(1.2, 2, "A")],
    )
    assert len(result["segments"]) == 2
    assert result["segments"][0]["end"] == 1
    assert result["segments"][1]["start"] == 1.2


def test_english_turns_keep_word_spaces_and_punctuation():
    result = build_transcript(
        texts=["Hello,", "world!", "OK."],
        turns=[Turn(0, 1, "A"), Turn(1, 2, "A"), Turn(2, 3, "B")],
        duration=3,
        language="English",
    )
    assert result["text"] == "Hello, world! OK."
    assert [s["text"] for s in result["segments"]] == ["Hello, world!", "OK."]


def test_empty_speech():
    result = transcript([], [])
    assert result["text"] == result["formatted_text"] == ""
    assert result["segments"] == result["words"] == []
    assert result["speaker_count"] == 0


def test_unpaired_text_is_never_silently_dropped():
    with pytest.raises(ValueError, match="数量不一致"):
        transcript(["你好", "再见"], [Turn(0, 2, "A")])


@pytest.mark.parametrize(
    "start,end", [(math.nan, 1), (0, math.inf), (-1, 1), (1, 1), (2, 1), (100, 101), (0, 101)]
)
def test_invalid_diarization_timestamps_rejected(start, end):
    with pytest.raises(ValueError, match="无效时间区间"):
        validate_turns([Turn(start, end, "A")], 100)


def test_nonexclusive_diarization_is_rejected():
    with pytest.raises(ValueError, match="exclusive"):
        validate_turns([Turn(0, 2, "A"), Turn(1, 3, "B")], 100)


def test_turns_are_sorted_and_small_end_overshoot_is_clipped():
    assert validate_turns([Turn(9, 10.2, "B"), Turn(0, 1, "A")], 10) == [
        Turn(0, 1, "A"),
        Turn(9, 10, "B"),
    ]


def test_timestamp_minutes_and_hours():
    assert format_time(0.99) == "00:00"
    assert format_time(61.2) == "01:01"
    assert format_time(3661) == "01:01:01"
