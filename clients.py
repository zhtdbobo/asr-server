"""HTTP clients for the existing Qwen API and the two independent GPU services."""

import io
import json
import logging
import math
import time
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit, urlunsplit

import httpx

from config import Settings

logger = logging.getLogger(__name__)
MAX_ALIGNMENT_TEXT_CHARS = 4096


class UpstreamError(RuntimeError):
    pass


class ChunkInferenceError(UpstreamError):
    """Model output that can be retried by re-recognizing shorter audio."""


def alignment_text_limit(duration: float) -> int:
    """A conservative character budget, well above ordinary speech density."""
    return min(MAX_ALIGNMENT_TEXT_CHARS, max(256, math.ceil(duration * 40)))


def sse_data(lines):
    """Read complete SSE data events, including multiline data and a final EOF event."""
    parts = []
    for line in lines:
        if not line:
            if parts:
                yield "\n".join(parts)
                parts = []
        elif line.startswith("data:"):
            parts.append(line[5:].lstrip(" "))
    if parts:
        yield "\n".join(parts)


def wav_bytes(waveform, sample_rate: int) -> bytes:
    import soundfile as sf

    target = io.BytesIO()
    sf.write(target, waveform, sample_rate, format="WAV", subtype="PCM_16")
    return target.getvalue()


class ServiceClients:
    def __init__(self, settings: Settings, transport=None):
        self.settings = settings
        self._asr_model = settings.asr_model.strip()
        self._model_lock = Lock()
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

    def _response_json(self, response, service, url, model=""):
        parts = urlsplit(url)
        safe_url = urlunsplit((parts.scheme, parts.netloc.rsplit("@", 1)[-1], parts.path, "", ""))
        target = f"{service}（{safe_url}" + (f"；model={model}" if model else "") + "）"
        if response.is_error:
            try:
                body = response.json()
            except ValueError:
                body = None
            detail = body.get("error", body.get("detail", "")) if isinstance(body, dict) else ""
            if isinstance(detail, dict):
                detail = detail.get("message", "")
            detail = self._safe_detail(detail or response.text)
            hint = ""
            if response.status_code == 404 and service.startswith("Qwen3-ASR"):
                hint = "；请核对 ASR_API_URL 和 ASR_MODEL，模型名须与 /v1/models 的 id 一致"
            raise UpstreamError(f"{target} 返回 HTTP {response.status_code}：{detail[:500]}{hint}")
        try:
            body = response.json()
        except ValueError as exc:
            raise UpstreamError(f"{target} 没有返回 JSON") from exc
        if not isinstance(body, dict):
            raise UpstreamError(f"{target} JSON 格式无效")
        return body

    def _model_name(self):
        if self._asr_model:
            return self._asr_model
        with self._model_lock:
            if self._asr_model:
                return self._asr_model
            url = urlsplit(self.settings.asr_api_url)
            path = url.path.rstrip("/")
            if url.scheme not in ("http", "https") or not path.endswith("/audio/transcriptions"):
                raise UpstreamError(
                    "自动读取模型名需要完整的 ASR_API_URL，例如 http://服务器:8001/v1/audio/transcriptions"
                )
            models_url = urlunsplit(
                (
                    url.scheme,
                    url.netloc,
                    path.removesuffix("/audio/transcriptions") + "/models",
                    "",
                    "",
                )
            )
            headers = (
                {"Authorization": f"Bearer {self.settings.asr_api_key}"}
                if self.settings.asr_api_key
                else {}
            )
            try:
                response = self.http.get(
                    models_url,
                    headers=headers,
                    timeout=httpx.Timeout(
                        min(self.settings.api_timeout_seconds, self.settings.asr_timeout_seconds),
                        connect=min(
                            15, self.settings.api_timeout_seconds, self.settings.asr_timeout_seconds
                        ),
                    ),
                )
            except httpx.TransportError as exc:
                raise UpstreamError(
                    "无法读取 ASR 的 /v1/models；请检查网络，或将 ASR_MODEL 填为成功请求中的完整 model"
                ) from exc
            body = self._response_json(response, "Qwen3-ASR 模型列表", models_url)
            entries = body.get("data")
            models = (
                {
                    entry["id"]
                    for entry in entries
                    if isinstance(entry, dict)
                    and isinstance(entry.get("id"), str)
                    and entry["id"].strip()
                }
                if isinstance(entries, list)
                else set()
            )
            if len(models) != 1:
                raise UpstreamError(
                    "ASR 的 /v1/models 未返回唯一模型；请将 ASR_MODEL 填为成功请求中的完整 model"
                )
            self._asr_model = models.pop()
            return self._asr_model

    def _post(self, url, source, data, key, service):
        if not url.startswith(("http://", "https://")):
            raise UpstreamError("请配置完整的服务 HTTP 地址，ASR_API_URL 必须指向现有上传接口")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        for attempt in range(3):
            started = time.perf_counter()
            logger.info(
                "%s 请求开始（第 %d/3 次，读写等待超时 %.0f 秒）",
                service,
                attempt + 1,
                self.settings.api_timeout_seconds,
            )
            try:
                if isinstance(source, Path):
                    with source.open("rb") as audio:
                        response = self.http.post(
                            url, data=data, files={"file": ("audio.wav", audio)}, headers=headers
                        )
                else:
                    response = self.http.post(
                        url,
                        data=data,
                        files={"file": ("audio.wav", source, "audio/wav")},
                        headers=headers,
                    )
            except httpx.TransportError as exc:
                logger.warning(
                    "%s 第 %d/3 次请求%s（%s），已耗时 %.1f 秒%s",
                    service,
                    attempt + 1,
                    "超时" if isinstance(exc, httpx.TimeoutException) else "连接失败",
                    type(exc).__name__,
                    time.perf_counter() - started,
                    "，准备重试" if attempt < 2 else "，停止重试",
                )
                if attempt == 2:
                    raise UpstreamError(
                        f"{service} 连接失败或超时（已尝试 3 次，读写等待超时设置 "
                        f"{self.settings.api_timeout_seconds:g} 秒），请检查地址、网络和服务状态"
                    ) from exc
                time.sleep(0.5 * (attempt + 1))
                continue
            if response.status_code in (429, 502, 503, 504) and attempt < 2:
                logger.warning(
                    "%s 第 %d/3 次请求返回 HTTP %d，准备重试",
                    service,
                    attempt + 1,
                    response.status_code,
                )
                time.sleep(0.5 * (attempt + 1))
                continue
            logger.info(
                "%s 请求结束，HTTP %d，耗时 %.1f 秒",
                service,
                response.status_code,
                time.perf_counter() - started,
            )
            return self._response_json(response, service, url, data.get("model", ""))
        raise UpstreamError(f"{service} 请求失败")

    @staticmethod
    def _asr_text(raw):
        return raw.split("<asr_text>", 1)[-1].strip()

    def _asr_stream_text(self, response, duration, started, timeout_seconds):
        def checked_lines():
            for line in response.iter_lines():
                if time.perf_counter() - started > timeout_seconds:
                    raise UpstreamError(
                        f"Qwen3-ASR 生成耗时超过 {timeout_seconds:g} 秒，已关闭响应；"
                        "未自动重复提交，请检查 ASR 服务生成上限和请求取消情况"
                    )
                yield line

        pieces = []
        finished = False
        limit = alignment_text_limit(duration)
        for data in sse_data(checked_lines()):
            if data.strip() == "[DONE]":
                break
            try:
                event = json.loads(data)
            except ValueError as exc:
                raise UpstreamError("Qwen3-ASR 流式响应包含无效 JSON") from exc
            if not isinstance(event, dict):
                raise UpstreamError("Qwen3-ASR 流式响应格式无效")
            if event.get("error"):
                detail = event["error"]
                if isinstance(detail, dict):
                    detail = detail.get("message", detail)
                raise UpstreamError(f"Qwen3-ASR 流式错误：{self._safe_detail(detail)}")
            if event.get("finish_reason") == "length":
                raise ChunkInferenceError("ASR 达到最大生成 token 数，正文被截断")

            event_type = event.get("type")
            if event_type == "transcript.text.delta":
                content = event.get("delta")
                if not isinstance(content, str):
                    raise UpstreamError("Qwen3-ASR 流式 delta 缺少字符串正文")
            elif event_type == "transcript.text.done":
                content = event.get("text")
                if not isinstance(content, str):
                    raise UpstreamError("Qwen3-ASR 流式 done 缺少完整正文")
                pieces = []  # This event contains the complete transcript.
                finished = True
            elif isinstance(event.get("choices"), list):
                choices = event["choices"]
                if not choices and event.get("usage") is not None:
                    continue
                if not choices:
                    raise UpstreamError("Qwen3-ASR 流式 choices 为空且没有用量信息")
                choice = choices[0]
                if not isinstance(choice, dict):
                    raise UpstreamError("Qwen3-ASR 流式 choices 格式无效")
                reason = choice.get("finish_reason")
                if reason == "length":
                    raise ChunkInferenceError("ASR 达到最大生成 token 数，正文被截断")
                if reason not in (None, "stop"):
                    raise UpstreamError(f"Qwen3-ASR 非正常结束：{self._safe_detail(reason)}")
                finished = finished or reason == "stop"
                delta = choice.get("delta")
                content = delta.get("content") if isinstance(delta, dict) else choice.get("text")
            else:
                raise UpstreamError("Qwen3-ASR 未返回支持的转写流式事件，请核对服务版本")
            if content is not None:
                if not isinstance(content, str):
                    raise UpstreamError("Qwen3-ASR 流式正文不是字符串")
                pieces.append(content)
            text = self._asr_text("".join(pieces))
            if len(text) > limit:
                raise ChunkInferenceError(
                    f"ASR 文本异常过长：超过本块 {limit} 字符，已主动关闭生成流，未送去对齐"
                )
        if not finished:
            raise UpstreamError("Qwen3-ASR 流式响应提前结束，未收到正常结束标记；未使用不完整正文")
        return self._asr_text("".join(pieces))

    def recognize(self, waveform, sample_rate: int) -> str:
        url = self.settings.asr_api_url
        if not url.startswith(("http://", "https://")):
            raise UpstreamError("请配置完整的 ASR_API_URL")
        model = self._model_name()
        timeout_seconds = min(self.settings.api_timeout_seconds, self.settings.asr_timeout_seconds)
        headers = {"Accept": "text/event-stream"}
        if self.settings.asr_api_key:
            headers["Authorization"] = f"Bearer {self.settings.asr_api_key}"
        data = {
            "model": model,
            "response_format": "json",
            "stream": "true",
            "temperature": "0",
            "max_completion_tokens": str(self.settings.asr_max_completion_tokens),
        }
        started = time.perf_counter()
        logger.info(
            "Qwen3-ASR 流式请求开始，生成上限 %d tokens，等待及流式耗时上限 %.0f 秒",
            self.settings.asr_max_completion_tokens,
            timeout_seconds,
        )
        try:
            with self.http.stream(
                "POST",
                url,
                data=data,
                files={"file": ("audio.wav", wav_bytes(waveform, sample_rate), "audio/wav")},
                headers=headers,
                timeout=httpx.Timeout(timeout_seconds, connect=min(15, timeout_seconds)),
            ) as response:
                if response.is_error:
                    response.read()
                    self._response_json(response, "Qwen3-ASR", url, model)
                if "text/event-stream" not in response.headers.get("content-type", "").lower():
                    response.read()
                    body = self._response_json(response, "Qwen3-ASR", url, model)
                    if body.get("finish_reason") == "length":
                        raise ChunkInferenceError("ASR 达到最大生成 token 数，正文被截断")
                    raise UpstreamError(
                        "Qwen3-ASR 未返回请求的流式响应，无法确认完整结束；"
                        "请核对 vLLM 对 stream=true 的支持"
                    )
                return self._asr_stream_text(
                    response, len(waveform) / sample_rate, started, timeout_seconds
                )
        except httpx.TransportError as exc:
            reason = "超时" if isinstance(exc, httpx.TimeoutException) else "连接或读取失败"
            logger.warning("Qwen3-ASR %s（%s），不自动重复提交", reason, type(exc).__name__)
            raise UpstreamError(
                f"Qwen3-ASR {reason}（读写等待上限 {timeout_seconds:g} 秒），未自动重复提交；"
                "请检查 ASR 服务生成上限和请求取消情况"
            ) from exc
        finally:
            logger.info("Qwen3-ASR 流式请求结束，耗时 %.1f 秒", time.perf_counter() - started)

    def diarize(self, path: Path, speaker_options: dict) -> list:
        body = self._post(
            self.settings.diarization_api_url,
            Path(path),
            {key: str(value) for key, value in speaker_options.items()},
            self.settings.internal_api_key,
            "Community-1",
        )
        if not isinstance(body.get("turns"), list):
            raise UpstreamError("说话人服务响应缺少 turns 数组")
        return body["turns"]

    def align(self, waveform, sample_rate: int, text: str, language: str) -> list:
        limit = alignment_text_limit(len(waveform) / sample_rate)
        if len(text) > limit:
            raise ChunkInferenceError(
                f"对齐文本异常过长：{len(text)} 字符，当前音频块上限 {limit} 字符"
            )
        body = self._post(
            self.settings.aligner_api_url,
            wav_bytes(waveform, sample_rate),
            {"text": text, "language": language},
            self.settings.internal_api_key,
            "ForcedAligner",
        )
        if not isinstance(body.get("words"), list):
            raise UpstreamError("对齐服务响应缺少 words 数组")
        return body["words"]
