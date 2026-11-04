import math

import pytest

from alignment import AlignedToken, SpeakerTimeline, Turn, build_transcript, format_time


def transcript(text, tokens, turns, **kwargs):
    return build_transcript(
        text=text, tokens=tokens, turns=turns, duration=100, language="Chinese", **kwargs
    )


def test_requested_conversation_format_and_returning_speaker():
    text = "那咱这边，这个东西有人看吗？嗯。就行了。嗯嗯。"
    tokens = [
        AlignedToken("那咱这边", 0, 0.9),
        AlignedToken("这个东西有人看吗", 1, 21),
        AlignedToken("嗯", 34, 34.4),
        AlignedToken("就行了", 35, 35.3),
        AlignedToken("嗯嗯", 35.5, 36),
    ]
    turns = [
        Turn(0, 0.95, "SPEAKER_07"),
        Turn(1, 21.5, "SPEAKER_02"),
        Turn(22, 34, "SPEAKER_05"),
        Turn(34, 34.8, "SPEAKER_02"),
        Turn(35, 35.4, "SPEAKER_05"),
        Turn(35.5, 36, "SPEAKER_02"),
    ]
    result = transcript(text, tokens, list(reversed(turns)))
    assert result["speaker_count"] == 3
    assert result["formatted_text"] == (
        "发言人1\n00:00\n那咱这边，\n\n"
        "发言人2\n00:01\n这个东西有人看吗？\n\n"
        "发言人2\n00:34\n嗯。\n\n"
        "发言人3\n00:35\n就行了。\n\n"
        "发言人2\n00:35\n嗯嗯。"
    )


def test_keep_punctuation_spaces_and_short_acknowledgement():
    result = transcript(
        "Hello, world! 嗯，OK。",
        [
            AlignedToken("Hello", 0, 0.4),
            AlignedToken("world", 0.5, 0.9),
            AlignedToken("嗯", 1, 1.1),
            AlignedToken("OK", 1.2, 1.5),
        ],
        [Turn(0, 1, "A"), Turn(1, 1.15, "B"), Turn(1.15, 2, "A")],
    )
    assert [s["text"] for s in result["segments"]] == ["Hello, world!", "嗯，", "OK。"]
    assert "".join(w["text"] for w in result["words"]) == result["text"]


def test_same_speaker_merges_but_long_pause_splits():
    result = transcript(
        "你好啊",
        [AlignedToken("你", 0, 0.2), AlignedToken("好", 0.2, 0.5), AlignedToken("啊", 5, 5.5)],
        [Turn(0, 1, "A"), Turn(5, 6, "A")],
    )
    assert [s["text"] for s in result["segments"]] == ["你好", "啊"]


def test_sum_overlap_for_same_speaker_across_fragments():
    timeline = SpeakerTimeline([Turn(0, 0.4, "A"), Turn(0.4, 1, "B"), Turn(1, 1.4, "A")], 0.75)
    assert timeline.assign(0, 1.4) == ("A", "overlap")


def test_overlap_tie_and_zero_duration_use_midpoint():
    timeline = SpeakerTimeline([Turn(0, 1, "A"), Turn(1, 2, "B")], 0.75)
    assert timeline.assign(0.5, 1.5) == ("B", "overlap")
    assert timeline.assign(1, 1) == ("B", "overlap")


def test_small_gaps_nearest_large_gaps_unknown():
    result = transcript(
        "嗯好",
        [AlignedToken("嗯", 1.2, 1.3), AlignedToken("好", 10, 10.5)],
        [Turn(0, 1, "A")],
    )
    assert [w["assignment"] for w in result["words"]] == ["nearest", "unknown"]
    assert result["segments"][1]["speaker_name"] == "未知发言人"
    assert len(result["warnings"]) == 2


def test_empty_speech():
    result = transcript("", [], [])
    assert result["segments"] == []
    assert result["formatted_text"] == ""
    assert result["speaker_count"] == 0


def test_text_is_never_silently_dropped_when_alignment_mismatches():
    with pytest.raises(RuntimeError, match="不一致"):
        transcript("你好啊", [AlignedToken("你好", 0, 1)], [Turn(0, 2, "A")])
    with pytest.raises(RuntimeError, match="没有有效时间戳"):
        transcript("你好", [], [Turn(0, 2, "A")])


@pytest.mark.parametrize("start,end", [(math.nan, 1), (0, math.inf), (-1, 0), (1, 0), (101, 102)])
def test_invalid_model_timestamps_rejected(start, end):
    with pytest.raises(RuntimeError):
        transcript("好", [AlignedToken("好", start, end)], [Turn(0, 2, "A")])


def test_nonexclusive_diarization_is_rejected():
    with pytest.raises(RuntimeError, match="exclusive"):
        SpeakerTimeline([Turn(0, 2, "A"), Turn(1, 3, "B")], 0.75)


def test_timestamp_minutes_and_hours():
    assert format_time(0.99) == "00:00"
    assert format_time(61.2) == "01:01"
    assert format_time(3661) == "01:01:01"
