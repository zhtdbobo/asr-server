"""Pure Python reconciliation of aligned text with exclusive diarization turns."""

import math
import unicodedata
from bisect import bisect_left, bisect_right
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

UNKNOWN_SPEAKER = "UNKNOWN"


@dataclass(frozen=True)
class Turn:
    start: float
    end: float
    speaker: str


@dataclass(frozen=True)
class AlignedToken:
    text: str
    start: float
    end: float


def format_time(seconds: float) -> str:
    whole = max(0, int(seconds))
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _is_content(char: str) -> bool:
    # Match the official aligner's cleaner: punctuation and whitespace have no timestamps.
    return char == "'" or unicodedata.category(char)[0] in "LN"


def restore_text(text: str, tokens: list[AlignedToken]) -> list[str]:
    """Restore ASR punctuation/spaces discarded by the forced aligner, without losing text."""
    if not tokens:
        if text.strip():
            raise RuntimeError("ASR 返回了正文，但没有有效时间戳，无法生成带说话人的结果")
        return []
    content_positions = [i for i, char in enumerate(text) if _is_content(char)]
    content = "".join(text[i] for i in content_positions)
    cleaned = ["".join(c for c in token.text if _is_content(c)) for token in tokens]
    if any(not value for value in cleaned) or "".join(cleaned) != content:
        raise RuntimeError("对齐文本与 ASR 正文不一致，已停止输出以避免漏字；请检查模型版本或语言")

    starts = []
    cursor = 0
    for token_text in cleaned:
        starts.append(content_positions[cursor])
        cursor += len(token_text)
    starts[0] = 0  # Preserve leading punctuation too.
    stops = starts[1:] + [len(text)]
    return [text[start:stop] for start, stop in zip(starts, stops)]


class SpeakerTimeline:
    def __init__(self, turns: Iterable[Turn], max_distance: float):
        self.turns = sorted(turns, key=lambda t: (t.start, t.end, t.speaker))
        self.max_distance = max_distance
        for i, turn in enumerate(self.turns):
            if not all(math.isfinite(t) for t in (turn.start, turn.end)):
                raise RuntimeError("说话人模型返回了非有限时间戳")
            if turn.start < 0 or turn.end <= turn.start:
                raise RuntimeError("说话人模型返回了无效时间区间")
            if i and turn.start < self.turns[i - 1].end - 1e-6:
                raise RuntimeError("文本对齐需要 Community-1 的 exclusive_speaker_diarization")
        self.starts = [t.start for t in self.turns]
        self.ends = [t.end for t in self.turns]

    def assign(self, start: float, end: float) -> tuple[str, str]:
        if not self.turns:
            return UNKNOWN_SPEAKER, "unknown"
        midpoint = (start + end) / 2
        first = bisect_right(self.ends, start)
        stop = bisect_left(self.starts, end)
        scores: dict[str, float] = {}
        for turn in self.turns[first:stop]:
            overlap = max(0.0, min(end, turn.end) - max(start, turn.start))
            scores[turn.speaker] = scores.get(turn.speaker, 0.0) + overlap
        midpoint_index = bisect_right(self.starts, midpoint) - 1
        midpoint_speaker = None
        if midpoint_index >= 0 and midpoint < self.ends[midpoint_index]:
            midpoint_speaker = self.turns[midpoint_index].speaker
        if scores:
            speaker = max(scores, key=lambda s: (scores[s], s == midpoint_speaker))
            return speaker, "overlap"
        # Zero-duration tokens still have a well-defined speaker when inside a turn.
        if start == end and midpoint_speaker is not None:
            return midpoint_speaker, "overlap"
        # Assign tiny alignment gaps only; distant text stays explicitly unknown.
        candidates = self.turns[max(0, first - 1) : min(len(self.turns), first + 1)]
        nearest = min(candidates, key=lambda t: min(abs(start - t.end), abs(t.start - end)))
        distance = max(0.0, nearest.start - end, start - nearest.end)
        if distance <= self.max_distance:
            return nearest.speaker, "nearest"
        return UNKNOWN_SPEAKER, "unknown"


def build_transcript(
    *,
    text: str,
    tokens: list[AlignedToken],
    turns: list[Turn],
    duration: float,
    language: str,
    merge_gap: float = 1.5,
    max_speaker_distance: float = 0.75,
) -> dict[str, Any]:
    timeline = SpeakerTimeline(turns, max_speaker_distance)
    names = {}
    for turn in timeline.turns:
        if turn.speaker not in names:
            names[turn.speaker] = f"发言人{len(names) + 1}"
    fragments = restore_text(text, tokens)
    words = []
    previous_start = -1.0
    for token, fragment in zip(tokens, fragments):
        if (
            not all(math.isfinite(t) for t in (token.start, token.end))
            or token.start < 0
            or token.end < token.start
            or token.start < previous_start
            or token.start > duration + 0.5
            or token.end > duration + 0.5
        ):
            raise RuntimeError("强制对齐模型返回了无效或乱序时间戳")
        previous_start = token.start
        start, end = min(token.start, duration), min(token.end, duration)
        speaker, assignment = timeline.assign(start, end)
        words.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "speaker": speaker,
                "speaker_name": names.get(speaker, "未知发言人"),
                "text": fragment,
                "assignment": assignment,
            }
        )

    segments = []
    for word in words:
        if (
            segments
            and segments[-1]["speaker"] == word["speaker"]
            and word["start"] - segments[-1]["end"] <= merge_gap
        ):
            segments[-1]["end"] = max(segments[-1]["end"], word["end"])
            segments[-1]["text"] += word["text"]
        else:
            segments.append(
                {key: word[key] for key in ("start", "end", "speaker", "speaker_name", "text")}
            )
    for segment in segments:
        segment["text"] = segment["text"].strip()
        segment["timestamp"] = format_time(segment["start"])
    formatted = "\n\n".join(f"{s['speaker_name']}\n{s['timestamp']}\n{s['text']}" for s in segments)
    warnings = []
    if any(w["assignment"] == "unknown" for w in words):
        warnings.append("部分文字与说话人时间段相距较远，标为未知发言人，请人工核对。")
    if any(w["assignment"] == "nearest" for w in words):
        warnings.append("部分字词位于说话人分段间隙，已按最近时间段分配，请核对边界。")
    return {
        "language": language,
        "duration": round(duration, 3),
        "speaker_count": len(names),
        "speakers": [{"id": key, "name": value} for key, value in names.items()],
        "text": text,
        "formatted_text": formatted,
        "segments": segments,
        "words": words,
        "diarization": [
            {"start": round(t.start, 3), "end": round(t.end, 3), "speaker": t.speaker}
            for t in timeline.turns
        ],
        "warnings": warnings,
    }
