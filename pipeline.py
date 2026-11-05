import logging
import math
import tempfile
from pathlib import Path
from threading import BoundedSemaphore
from time import perf_counter

from alignment import AlignedToken, Turn, build_transcript, restore_text
from audio import SAMPLE_RATE, decode_audio
from clients import ChunkInferenceError, ServiceClients, UpstreamError, alignment_text_limit
from config import Settings

logger = logging.getLogger(__name__)

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


def validate_aligned_words(words: list, text: str, duration: float) -> list[AlignedToken]:
    """Validate an entire chunk before committing any of its text or timestamps."""
    if not words:
        raise ChunkInferenceError("ASR 返回了正文，但对齐服务没有返回时间戳")
    tokens = []
    previous_start = -math.inf
    for index, word in enumerate(words, start=1):
        try:
            begin, end = float(word["start"]), float(word["end"])
            token_text = word["text"]
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ChunkInferenceError(f"对齐服务第 {index} 个字词字段无效") from exc
        if (
            not all(math.isfinite(t) for t in (begin, end))
            or begin < 0
            or end < begin
            or begin < previous_start
            or end > duration + 0.5
            or not isinstance(token_text, str)
        ):
            raise ChunkInferenceError(
                f"对齐服务第 {index} 个字词时间戳无效：start={begin:g}, end={end:g}，"
                f"该音频块时长={duration:.3f} 秒，上一字词开始={previous_start:g}"
            )
        previous_start = begin
        tokens.append(AlignedToken(token_text, min(begin, duration), min(end, duration)))
    try:
        restore_text(text, tokens)
    except RuntimeError as exc:
        raise ChunkInferenceError(str(exc)) from exc
    return tokens


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

    def _transcribe_chunk(self, chunk, sample_rate, language, offset, label, depth=0):
        duration = len(chunk) / sample_rate
        logger.info("%s [%.1f–%.1f 秒] 开始 ASR", label, offset, offset + duration)
        try:
            text = self.clients.recognize(chunk, sample_rate)
            if not text:
                logger.info("%s ASR 返回空文本，继续下一块", label)
                return [], [], []
            limit = alignment_text_limit(duration)
            if len(text) > limit:
                raise ChunkInferenceError(
                    f"ASR 文本异常过长：{len(text)} 字符，音频 {duration:.3f} 秒，"
                    f"本块上限 {limit} 字符；未发送到对齐服务"
                )
            logger.info("%s ASR 完成，%d 字符；开始时间戳对齐", label, len(text))
            words = self.clients.align(chunk, sample_rate, text, language)
            local_tokens = validate_aligned_words(words, text, duration)
        except ChunkInferenceError as exc:
            if depth >= 3 or duration < 10:
                raise UpstreamError(
                    f"{label} [原录音 {offset:.3f}–{offset + duration:.3f} 秒] 失败：{exc}；"
                    "缩块重试已达上限，未生成含猜测时间戳的结果"
                ) from exc
            message = (
                f"{label} [原录音 {offset:.3f}–{offset + duration:.3f} 秒] {exc}；"
                "将音频二等分重新识别并对齐"
            )
            logger.warning(message)
            midpoint = len(chunk) // 2
            left_parts, left_tokens, left_warnings = self._transcribe_chunk(
                chunk[:midpoint], sample_rate, language, offset, f"{label}.1", depth + 1
            )
            right_parts, right_tokens, right_warnings = self._transcribe_chunk(
                chunk[midpoint:],
                sample_rate,
                language,
                offset + midpoint / sample_rate,
                f"{label}.2",
                depth + 1,
            )
            return (
                left_parts + right_parts,
                left_tokens + right_tokens,
                [message, *left_warnings, *right_warnings],
            )
        logger.info("%s 对齐完成，%d 个字词", label, len(local_tokens))
        return (
            [text],
            [AlignedToken(t.text, t.start + offset, t.end + offset) for t in local_tokens],
            [],
        )

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
            name = Path(path).name
            if not self.settings.asr_api_url:
                raise UpstreamError(
                    "请先设置 ASR_API_URL，指向已有的 /v1/audio/transcriptions 接口"
                )
            logger.info("[%s] 开始解码音频", name)
            waveform, sample_rate = decode_audio(path, self.settings.max_audio_seconds)
            if sample_rate != SAMPLE_RATE:
                raise RuntimeError("解码器返回了错误的采样率")
            total_seconds = len(waveform) / sample_rate
            logger.info("[%s] 解码完成，录音时长 %.1f 秒；开始说话人分离", name, total_seconds)
            try:
                import soundfile as sf

                # Keep one full-length request for global speaker clustering. Sending
                # the original 8-channel WAV can exceed the GPU upload limit.
                with tempfile.TemporaryDirectory(prefix="meeting-diarization-") as directory:
                    mono_path = Path(directory) / "mono-16k.wav"
                    sf.write(mono_path, waveform, sample_rate, subtype="PCM_16")
                    logger.info(
                        "[%s] 上传整段 16 kHz 单声道 PCM16，共 %.1f MB",
                        name,
                        mono_path.stat().st_size / 1024**2,
                    )
                    turns = [
                        Turn(float(t["start"]), float(t["end"]), str(t["speaker"]))
                        for t in self.clients.diarize(mono_path, speaker_options)
                    ]
            except (KeyError, TypeError, ValueError) as exc:
                raise UpstreamError("说话人服务返回了无效的时间区间") from exc
            logger.info("[%s] 说话人分离完成，共 %d 个时间段", name, len(turns))
            parts, tokens, retry_warnings = [], [], []
            for index, (chunk, offset) in enumerate(
                audio_chunks(waveform, sample_rate, self.settings.asr_chunk_seconds), start=1
            ):
                duration = len(chunk) / sample_rate
                if not any(t.end > offset and t.start < offset + duration for t in turns):
                    continue
                logger.info(
                    "[%s] 分块 %d 进度 [%.1f–%.1f / %.1f 秒]",
                    name,
                    index,
                    offset,
                    offset + duration,
                    total_seconds,
                )
                chunk_parts, chunk_tokens, warnings = self._transcribe_chunk(
                    chunk, sample_rate, language, offset, f"[{name}] 分块 {index}"
                )
                for text in chunk_parts:
                    if parts and language not in ("Chinese", "Cantonese", "Japanese", "Korean"):
                        parts.append(" ")
                    parts.append(text)
                tokens.extend(chunk_tokens)
                retry_warnings.extend(warnings)
            logger.info("[%s] 开始合并文本、时间戳和说话人", name)
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
            output["warnings"].extend(retry_warnings)
            if not turns:
                output["warnings"].append("未检测到有效语音。")
            logger.info(
                "[%s] 转写完成，共 %d 段，耗时 %.1f 秒",
                name,
                len(output["segments"]),
                output["processing_seconds"],
            )
            return output
        finally:
            self._slots.release()
