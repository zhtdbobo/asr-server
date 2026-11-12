"""ServiceClients 的 FireRedASR2 适配版。

与原始 clients.py 的唯一区别：ASR 识别 recognize() 使用普通 JSON 接口
（FireRedASR2-AED 服务，如 http://127.0.0.1:8004/v1/audio/transcriptions），
而不是 Qwen3-ASR 的 SSE 流式协议。

diarize() 与原始实现保持一致（调用 Community-1 /v1/diarization，返回 {"turns":[...]}）。
其余 API 与 clients.ServiceClients 相同，可直接注入 MeetingPipeline。
"""

import io
import json
import logging
import math
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import httpx

from config import Settings

logger = logging.getLogger(__name__)
MAX_ASR_TEXT_CHARS = 4096


class UpstreamError(RuntimeError):
    pass


class ChunkInferenceError(UpstreamError):
    """Model output that can be retried by re-recognizing shorter audio."""


def asr_text_limit(duration: float) -> int:
    return min(MAX_ASR_TEXT_CHARS, max(256, math.ceil(duration * 40)))


def wav_bytes(waveform, sample_rate: int) -> bytes:
    import soundfile as sf

    target = io.BytesIO()
    sf.write(target, waveform, sample_rate, format="WAV", subtype="PCM_16")
    return target.getvalue()


class FireRedServiceClients:
    """Async-free HTTP client: FireRedASR2 ASR (JSON) + diarization (unchanged)."""

    def __init__(self, settings: Settings, transport=None):
        self.settings = settings
        self._asr_model = settings.asr_model.strip() or "fireredasr2"
        self.http = httpx.Client(
            timeout=httpx.Timeout(settings.api_timeout_seconds, connect=15),
            limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
            trust_env=False,
            transport=transport,
        )

    def close(self):
        self.http.close()

    def _safe_detail(self, detail):
        detail = str(detail).replace("\n", " ").replace("\r", " ")
        for secret in (self.settings.asr_api_key, self.settings.internal_api_key):
            if secret:
                detail = detail.replace(secret, "[redacted]")
        return detail[:500]

    # ---------- ASR：FireRedASR2 普通 JSON ----------
    def recognize(self, waveform, sample_rate: int) -> str:
        url = self.settings.asr_api_url
        if not url.startswith(("http://", "https://")):
            raise UpstreamError("请配置完整的 ASR_API_URL")
        duration = len(waveform) / sample_rate
        logger.info("FireRed 准备转写 %.3f 秒音频（普通 JSON 接口）", duration)
        payload = wav_bytes(waveform, sample_rate)
        headers = {}
        if self.settings.asr_api_key:
            headers["Authorization"] = f"Bearer {self.settings.asr_api_key}"
        data = {"model": self._asr_model, "response_format": "json"}

        timeout_seconds = min(self.settings.api_timeout_seconds, self.settings.asr_timeout_seconds)
        started = time.perf_counter()
        last_exc = None
        for attempt in range(3):
            try:
                response = self.http.post(
                    url,
                    data=data,
                    files={"file": ("audio.wav", payload, "audio/wav")},
                    headers=headers,
                    timeout=httpx.Timeout(timeout_seconds, connect=min(15, timeout_seconds)),
                )
            except httpx.TransportError as exc:
                last_exc = exc
                logger.warning(
                    "FireRed 第 %d/3 次请求%s（%s）",
                    attempt + 1,
                    "超时" if isinstance(exc, httpx.TimeoutException) else "连接失败",
                    type(exc).__name__,
                )
                if attempt == 2:
                    raise UpstreamError(
                        f"FireRed ASR 连接失败或超时（已尝试 3 次）"
                    ) from exc
                time.sleep(0.5 * (attempt + 1))
                continue

            if response.status_code in (429, 502, 503, 504) and attempt < 2:
                logger.warning("FireRed 第 %d/3 次返回 HTTP %d，重试", attempt + 1, response.status_code)
                time.sleep(0.5 * (attempt + 1))
                continue

            if response.is_error:
                try:
                    body = response.json()
                    detail = body.get("error", body.get("detail", "")) if isinstance(body, dict) else ""
                except ValueError:
                    detail = response.text
                raise UpstreamError(
                    f"FireRed ASR 返回 HTTP {response.status_code}：{self._safe_detail(detail)}"
                )

            try:
                body = response.json()
            except ValueError as exc:
                raise UpstreamError("FireRed ASR 没有返回 JSON") from exc
            if not isinstance(body, dict):
                raise UpstreamError("FireRed ASR JSON 格式无效")

            text = body.get("text")
            if text is None:
                raise UpstreamError(f"FireRed ASR 响应缺少 text 字段：{self._safe_detail(body)}")
            text = str(text).strip()
            # 过长文本保护：交给上层缩块重试
            limit = asr_text_limit(duration)
            if len(text) > limit:
                raise ChunkInferenceError(
                    f"ASR 文本异常过长：{len(text)} 字符，音频 {duration:.3f} 秒，上限 {limit} 字符"
                )
            logger.info(
                "FireRed ASR 完成，%d 字符，耗时 %.2f 秒", len(text), time.perf_counter() - started
            )
            return text

        raise UpstreamError(f"FireRed ASR 请求失败：{last_exc}")

    # ---------- Diarization：与原实现一致 ----------
    def diarize(self, path: Path, speaker_options: dict) -> list:
        url = self.settings.diarization_api_url
        if not url.startswith(("http://", "https://")):
            raise UpstreamError("请配置完整的 DIARIZATION_API_URL，例如 http://127.0.0.1:8002/v1/diarization")
        headers = (
            {"Authorization": f"Bearer {self.settings.internal_api_key}"}
            if self.settings.internal_api_key
            else {}
        )
        data = {key: str(value) for key, value in speaker_options.items()}
        started = time.perf_counter()
        try:
            with path.open("rb") as audio:
                logger.info(
                    "Community-1 上传文件 %.1f MB，等待响应",
                    path.stat().st_size / 1024 ** 2,
                )
                response = self.http.post(
                    url,
                    data=data,
                    files={"file": ("audio.wav", audio)},
                    headers=headers,
                    timeout=httpx.Timeout(self.settings.api_timeout_seconds, connect=15),
                )
        except httpx.TransportError as exc:
            raise UpstreamError(
                f"说话人服务连接失败或超时：{type(exc).__name__}；请检查 DIARIZATION_API_URL 和服务状态"
            ) from exc

        if response.is_error:
            try:
                body = response.json()
                detail = body.get("error", body.get("detail", "")) if isinstance(body, dict) else ""
            except ValueError:
                detail = response.text
            raise UpstreamError(
                f"Community-1 返回 HTTP {response.status_code}：{self._safe_detail(detail)}"
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise UpstreamError("说话人服务没有返回 JSON") from exc
        if not isinstance(body.get("turns"), list):
            raise UpstreamError("说话人服务响应缺少 turns 数组")
        logger.info(
            "Community-1 结果读取完成，共 %d 个说话人时间段，耗时 %.2f 秒",
            len(body["turns"]),
            time.perf_counter() - started,
        )
        return body["turns"]
