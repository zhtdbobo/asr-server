import math
from pathlib import Path
from threading import BoundedSemaphore
from time import perf_counter

from alignment import AlignedToken, Turn, build_transcript
from audio import SAMPLE_RATE, decode_audio
from clients import ServiceClients, UpstreamError
from config import Settings

ALIGNMENT_LANGUAGES = (
    "Chinese",
    "English",
    "Cantonese",
    "French",
    "German",
    "Italian",
    "Japanese",
    "Korean",
    "Portuguese",
    "Russian",
    "Spanish",
)


class PipelineBusyError(RuntimeError):
    pass


def validate_options(language, num_speakers=None, min_speakers=None, max_speakers=None):
    language = language.strip().capitalize()
    if language not in ALIGNMENT_LANGUAGES:
        raise ValueError(f"时间戳模式支持这些语言: {', '.join(ALIGNMENT_LANGUAGES)}")
    if num_speakers is not None and (min_speakers is not None or max_speakers is not None):
        raise ValueError("num_speakers 不能与 min_speakers / max_speakers 同时使用")
    for value in (num_speakers, min_speakers, max_speakers):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 1
        ):
            raise ValueError("说话人数必须为正整数")
    if min_speakers is not None and max_speakers is not None and min_speakers > max_speakers:
        raise ValueError("min_speakers 不能大于 max_speakers")
    return language, {
        key: value
        for key, value in {
            "num_speakers": num_speakers,
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
        }.items()
        if value is not None
    }


def audio_chunks(waveform, sample_rate: int, max_seconds: float):
    """Disjoint chunks, cut at a low-energy window before the duration limit."""
    import numpy as np

    start = 0
    limit = max(1, int(max_seconds * sample_rate))
    window = max(1, min(int(0.05 * sample_rate), limit // 4))
    while start < len(waveform):
        stop = min(len(waveform), start + limit)
        if stop < len(waveform) and limit > window * 4:
            scan_start = max(start + limit // 2, stop - sample_rate)
            windows = (stop - scan_start) // window
            if windows:
                scan = waveform[scan_start : scan_start + windows * window]
                energy = np.mean(np.square(scan.reshape(-1, window)), axis=1)
                stop = scan_start + int(np.argmin(energy)) * window + window // 2
        yield waveform[start:stop], start / sample_rate
        start = stop


class MeetingPipeline:
    """CPU orchestrator. All three inference stages are called over HTTP."""

    def __init__(self, settings: Settings | None = None, clients=None):
        self.settings = settings or Settings.from_env()
        self.clients = clients if clients is not None else ServiceClients(self.settings)
        self._slots = BoundedSemaphore(self.settings.job_workers)

    @property
    def loaded(self):
        return False  # GPU models live in the two independent services.

    def close(self):
        self.clients.close()

    def transcribe(
        self,
        path: str | Path,
        *,
        language: str = "Chinese",
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> dict:
        language, speaker_options = validate_options(
            language, num_speakers, min_speakers, max_speakers
        )
        if not self._slots.acquire(blocking=False):
            raise PipelineBusyError("处理槽位已满，请使用 POST /v1/jobs 提交排队任务")
        started = perf_counter()
        try:
            if not self.settings.asr_api_url:
                raise UpstreamError(
                    "请先设置 ASR_API_URL，指向已有的 /v1/audio/transcriptions 接口"
                )
            waveform, sample_rate = decode_audio(path, self.settings.max_audio_seconds)
            if sample_rate != SAMPLE_RATE:
                raise RuntimeError("解码器返回了错误的采样率")
            try:
                turns = [
                    Turn(float(t["start"]), float(t["end"]), str(t["speaker"]))
                    for t in self.clients.diarize(Path(path), speaker_options)
                ]
            except (KeyError, TypeError, ValueError) as exc:
                raise UpstreamError("说话人服务返回了无效的时间区间") from exc
            parts, tokens = [], []
            for chunk, offset in audio_chunks(
                waveform, sample_rate, self.settings.asr_chunk_seconds
            ):
                duration = len(chunk) / sample_rate
                if not any(t.end > offset and t.start < offset + duration for t in turns):
                    continue
                text = self.clients.recognize(chunk, sample_rate)
                if not text:
                    continue
                words = self.clients.align(chunk, sample_rate, text, language)
                if not words:
                    raise UpstreamError("ASR 返回了正文，但对齐服务没有返回时间戳")
                if parts and language not in ("Chinese", "Cantonese", "Japanese", "Korean"):
                    parts.append(" ")
                parts.append(text)
                try:
                    for word in words:
                        begin, end = float(word["start"]), float(word["end"])
                        if (
                            not all(math.isfinite(t) for t in (begin, end))
                            or begin < 0
                            or end < begin
                            or end > duration + 0.5
                        ):
                            raise ValueError("invalid timestamps")
                        tokens.append(
                            AlignedToken(
                                str(word["text"]),
                                min(begin, duration) + offset,
                                min(end, duration) + offset,
                            )
                        )
                except (KeyError, TypeError, ValueError) as exc:
                    raise UpstreamError("对齐服务返回了无效的字词时间戳") from exc
            output = build_transcript(
                text="".join(parts),
                tokens=tokens,
                turns=turns,
                duration=len(waveform) / sample_rate,
                language=language,
                merge_gap=self.settings.merge_gap_seconds,
                max_speaker_distance=self.settings.max_speaker_distance,
            )
            output["processing_seconds"] = round(perf_counter() - started, 3)
            if not turns:
                output["warnings"].append("未检测到有效语音。")
            return output
        finally:
            self._slots.release()
