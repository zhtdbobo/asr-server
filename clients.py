"""HTTP clients for the existing Qwen API and the two independent GPU services."""

import io
import time
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit, urlunsplit

import httpx

from config import Settings


class UpstreamError(RuntimeError):
    pass


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
            detail = str(detail or response.text).replace("\n", " ").replace("\r", " ")
            for secret in (self.settings.asr_api_key, self.settings.internal_api_key):
                if secret:
                    detail = detail.replace(secret, "[redacted]")
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
                response = self.http.get(models_url, headers=headers)
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
                if attempt == 2:
                    raise UpstreamError(
                        f"{service} 连接失败或超时，请检查地址、网络和服务状态"
                    ) from exc
                time.sleep(0.5 * (attempt + 1))
                continue
            if response.status_code in (429, 502, 503, 504) and attempt < 2:
                time.sleep(0.5 * (attempt + 1))
                continue
            return self._response_json(response, service, url, data.get("model", ""))
        raise UpstreamError(f"{service} 请求失败")

    def recognize(self, waveform, sample_rate: int) -> str:
        body = self._post(
            self.settings.asr_api_url,
            wav_bytes(waveform, sample_rate),
            {"model": self._model_name(), "response_format": "json", "stream": "false"},
            self.settings.asr_api_key,
            "Qwen3-ASR",
        )
        text = body.get("text")
        if not isinstance(text, str):
            raise UpstreamError("Qwen3-ASR 响应缺少字符串 text 字段")
        if body.get("finish_reason") == "length":
            raise UpstreamError("ASR 正文被截断，请缩短 ASR_CHUNK_SECONDS 或调整 ASR 服务生成上限")
        if "<asr_text>" in text:
            text = text.split("<asr_text>", 1)[1]
        return text.strip()

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
