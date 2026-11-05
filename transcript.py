"""Assemble recognized speaker turns using the diarization timestamps."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Turn:
    start: float
    end: float
    speaker: str


def format_time(seconds: float) -> str:
    whole = max(0, int(seconds))
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def join_text(parts: list[str], language: str) -> str:
    separator = "" if language in ("Chinese", "Cantonese", "Japanese", "Korean") else " "
    return separator.join(part.strip() for part in parts if part.strip())


def validate_turns(turns: list[Turn], duration: float) -> list[Turn]:
    """Reject invalid/nonexclusive turns before using them to slice the audio."""
    ordered = sorted(turns, key=lambda turn: (turn.start, turn.end, turn.speaker))
    validated = []
    for turn in ordered:
        if (
            not all(math.isfinite(t) for t in (turn.start, turn.end))
            or turn.start < 0
            or turn.end <= turn.start
            or turn.start >= duration
            or turn.end > duration + 0.5
            or not turn.speaker.strip()
        ):
            raise ValueError(
                f"说话人服务返回了无效时间区间：start={turn.start:g}, end={turn.end:g}，"
                f"录音时长={duration:.3f} 秒，speaker={turn.speaker!r}"
            )
        end = min(turn.end, duration)
        if validated and turn.start < validated[-1].end - 1e-6:
            raise ValueError("需要 Community-1 的 exclusive_speaker_diarization 互斥时间线")
        validated.append(Turn(turn.start, end, turn.speaker))
    return validated


def build_transcript(
    *,
    texts: list[str],
    turns: list[Turn],
    duration: float,
    language: str,
    merge_gap: float = 1.5,
) -> dict:
    if len(texts) != len(turns):
        raise ValueError("说话人片段与转写正文数量不一致")
    names = {}
    for turn in turns:
        if turn.speaker not in names:
            names[turn.speaker] = f"发言人{len(names) + 1}"

    segments = []
    merge_allowed = True
    for turn, text in zip(turns, texts):
        if segments and turn.speaker != segments[-1]["speaker"]:
            merge_allowed = False
        text = text.strip()
        if not text:
            continue
        if (
            segments
            and merge_allowed
            and segments[-1]["speaker"] == turn.speaker
            and turn.start - segments[-1]["end"] <= merge_gap
        ):
            segments[-1]["end"] = turn.end
            segments[-1]["text"] = join_text([segments[-1]["text"], text], language)
        else:
            segments.append(
                {
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": turn.speaker,
                    "speaker_name": names[turn.speaker],
                    "text": text,
                }
            )
        merge_allowed = True
    for segment in segments:
        segment["timestamp"] = format_time(segment["start"])
        segment["start"] = round(segment["start"], 3)
        segment["end"] = round(segment["end"], 3)
    formatted = "\n\n".join(f"{s['speaker_name']}\n{s['timestamp']}\n{s['text']}" for s in segments)
    return {
        "language": language,
        "duration": round(duration, 3),
        "speaker_count": len(names),
        "speakers": [{"id": key, "name": value} for key, value in names.items()],
        "text": join_text(texts, language),
        "formatted_text": formatted,
        "segments": segments,
        "timestamp_mode": "segment",
        "words": [],  # Retain the response field without inventing word timestamps.
        "diarization": [
            {"start": round(t.start, 3), "end": round(t.end, 3), "speaker": t.speaker}
            for t in turns
        ],
        "warnings": [],
    }
