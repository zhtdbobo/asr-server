import logging
import tempfile
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from threading import BoundedSemaphore
from time import perf_counter

from audio import SAMPLE_RATE, decode_audio
from clients import ChunkInferenceError, ServiceClients, UpstreamError, asr_text_limit
from config import Settings
from transcript import Turn, build_transcript, join_text, validate_turns

logger = logging.getLogger(__name__)

TRANSCRIPT_LANGUAGES = (
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
    if language not in TRANSCRIPT_LANGUAGES:
        raise ValueError(f"支持这些语言: {', '.join(TRANSCRIPT_LANGUAGES)}")
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
    """CPU orchestrator for diarization followed by ASR of each speaker turn."""

    def __init__(self, settings: Settings | None = None, clients=None):
        self.settings = settings or Settings.from_env()
        self.clients = clients if clients is not None else ServiceClients(self.settings)
        self._slots = BoundedSemaphore(self.settings.job_workers)

    @property
    def loaded(self):
        return False  # GPU models live in the independent services.

    def close(self):
        self.clients.close()

    def _transcribe_chunk(self, chunk, sample_rate, offset, label, depth=0):
        duration = len(chunk) / sample_rate
        logger.info("%s [%.1f–%.1f 秒] 开始 ASR", label, offset, offset + duration)
        try:
            text = self.clients.recognize(chunk, sample_rate)
            if not text:
                logger.info("%s ASR 返回空文本，继续下一块", label)
                return [], []
            limit = asr_text_limit(duration)
            if len(text) > limit:
                raise ChunkInferenceError(
                    f"ASR 文本异常过长：{len(text)} 字符，音频 {duration:.3f} 秒，"
                    f"本块上限 {limit} 字符"
                )
        except ChunkInferenceError as exc:
            if depth >= 3 or duration < 10:
                raise UpstreamError(
                    f"{label} [原录音 {offset:.3f}–{offset + duration:.3f} 秒] 失败：{exc}；"
                    "缩块重试已达上限，未使用不完整正文"
                ) from exc
            message = (
                f"{label} [原录音 {offset:.3f}–{offset + duration:.3f} 秒] {exc}；"
                "将音频二等分重新识别"
            )
            logger.warning(message)
            midpoint = len(chunk) // 2
            left_parts, left_warnings = self._transcribe_chunk(
                chunk[:midpoint], sample_rate, offset, f"{label}.1", depth + 1
            )
            right_parts, right_warnings = self._transcribe_chunk(
                chunk[midpoint:],
                sample_rate,
                offset + midpoint / sample_rate,
                f"{label}.2",
                depth + 1,
            )
            return (
                left_parts + right_parts,
                [message, *left_warnings, *right_warnings],
            )
        logger.info("%s ASR 完成，%d 字符", label, len(text))
        return [text], []

    def _asr_chunks(self, waveform, sample_rate, turns, name):
        for index, turn in enumerate(turns):
            start_sample = round(turn.start * sample_rate)
            end_sample = round(turn.end * sample_rate)
            logger.info(
                "[%s] 说话人片段 %d/%d %s [%.3f–%.3f / %.1f 秒]",
                name,
                index + 1,
                len(turns),
                turn.speaker,
                turn.start,
                turn.end,
                len(waveform) / sample_rate,
            )
            for chunk_index, (chunk, local_offset) in enumerate(
                audio_chunks(
                    waveform[start_sample:end_sample], sample_rate, self.settings.asr_chunk_seconds
                ),
                start=1,
            ):
                yield (
                    index,
                    chunk,
                    start_sample / sample_rate + local_offset,
                    f"[{name}] 片段 {index + 1} 分块 {chunk_index}",
                )

    def _transcribe_turns(self, waveform, sample_rate, turns, name, language):
        tasks = enumerate(self._asr_chunks(waveform, sample_rate, turns, name))
        results = {}
        logger.info("[%s] 开始 ASR，文件内最多 %d 个分块并发", name, self.settings.asr_workers)
        if self.settings.asr_workers == 1:
            for number, (turn_index, chunk, offset, label) in tasks:
                results[number] = (
                    turn_index,
                    self._transcribe_chunk(chunk, sample_rate, offset, label),
                )
        else:
            executor = ThreadPoolExecutor(
                max_workers=self.settings.asr_workers, thread_name_prefix="asr-chunk"
            )
            pending = {}

            def submit_next():
                task = next(tasks, None)
                if task is None:
                    return
                number, (turn_index, chunk, offset, label) = task
                future = executor.submit(self._transcribe_chunk, chunk, sample_rate, offset, label)
                pending[future] = (number, turn_index)

            try:
                # Keep only one task per worker in flight; do not queue the entire recording.
                for _ in range(self.settings.asr_workers):
                    submit_next()
                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        number, turn_index = pending.pop(future)
                        results[number] = (turn_index, future.result())
                    for _ in done:
                        submit_next()
            finally:
                # Drain running requests before releasing the file slot or closing HTTP clients.
                for future in pending:
                    future.cancel()
                executor.shutdown(wait=True, cancel_futures=True)

        texts = [[] for _ in turns]
        retry_warnings = []
        for number in sorted(results):
            turn_index, (parts, warnings) = results[number]
            texts[turn_index].extend(parts)
            retry_warnings.extend(warnings)
        return [join_text(parts, language) for parts in texts], retry_warnings

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
        timings = {}
        phase_name, phase_started = "decode_seconds", started
        total_seconds = None
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
            decoded_at = perf_counter()
            timings[phase_name] = round(decoded_at - phase_started, 3)
            phase_name, phase_started = "diarization_seconds", decoded_at
            logger.info("[%s] 解码完成，录音时长 %.1f 秒；开始说话人分离", name, total_seconds)
            try:
                import soundfile as sf

                # Keep one full-length request for global speaker clustering. Sending
                # the original 8-channel WAV can exceed the GPU upload limit.
                with tempfile.TemporaryDirectory(prefix="meeting-diarization-") as directory:
                    mono_path = Path(directory) / "mono-16k.wav"
                    logger.info("[%s] 开始写入整段单声道 PCM16 临时 WAV", name)
                    sf.write(mono_path, waveform, sample_rate, subtype="PCM_16")
                    logger.info(
                        "[%s] 上传整段 16 kHz 单声道 PCM16，共 %.1f MB",
                        name,
                        mono_path.stat().st_size / 1024**2,
                    )
                    turns = [
                        Turn(float(t["start"]), float(t["end"]), t["speaker"])
                        for t in self.clients.diarize(mono_path, speaker_options)
                    ]
                    logger.info(
                        "[%s] 说话人响应已转换，共 %d 段；开始清理临时 WAV", name, len(turns)
                    )
                logger.info("[%s] 临时 WAV 清理完成；开始校验说话人时间段", name)
                if any(not isinstance(t.speaker, str) for t in turns):
                    raise ValueError("说话人 ID 必须为字符串")
                turns = validate_turns(turns, total_seconds)
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise UpstreamError(f"说话人服务返回了无效的时间区间：{exc}") from exc
            logger.info("[%s] 说话人分离完成，共 %d 个时间段", name, len(turns))
            diarized_at = perf_counter()
            timings[phase_name] = round(diarized_at - phase_started, 3)
            phase_name, phase_started = "asr_seconds", diarized_at
            texts, retry_warnings = self._transcribe_turns(
                waveform, sample_rate, turns, name, language
            )
            recognized_at = perf_counter()
            timings[phase_name] = round(recognized_at - phase_started, 3)
            phase_name, phase_started = "assembly_seconds", recognized_at
            logger.info("[%s] 按说话人时间段合并转写正文", name)
            output = build_transcript(
                texts=texts,
                turns=turns,
                duration=len(waveform) / sample_rate,
                language=language,
                merge_gap=self.settings.merge_gap_seconds,
            )
            assembled_at = perf_counter()
            output["processing_seconds"] = round(assembled_at - started, 3)
            timings[phase_name] = round(assembled_at - phase_started, 3)
            output["timings"] = timings
            output["warnings"].extend(retry_warnings)
            logger.info("[%s] 转写结果组装完成，共 %d 条警告", name, len(output["warnings"]))
            if not turns:
                output["warnings"].append("未检测到有效语音。")
            logger.info(
                "[%s] 转写完成，共 %d 段，耗时 %.1f 秒",
                name,
                len(output["segments"]),
                output["processing_seconds"],
            )
            return output
        except UpstreamError as exc:
            finished = perf_counter()
            timings[phase_name] = round(finished - phase_started, 3)
            exc.timings = timings
            exc.duration_seconds = total_seconds
            exc.processing_seconds = round(finished - started, 3)
            raise
        finally:
            self._slots.release()
