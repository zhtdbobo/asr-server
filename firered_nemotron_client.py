"""FireRedASR2(ASR) + Nemotron-3-Diarization 适配客户端。

与 fireRed_client.FireRedServiceClients 的唯一区别：
  diarize() 调用 Nemotron-3-Diarization 服务（默认 http://127.0.0.1:8006/v1/audio/diarizations），
  其响应格式为 {"duration":..,"segments":[{"start","end","speaker"(int),"speaker_label"}]}。
  这里把 segments 转成 pipeline 期望的 [{"start","end","speaker"(str)}]（speaker 转字符串）。

ASR 仍走 FireRedASR2 普通 JSON 接口（同 fireRed_client）。
用法：
  ASR_API_URL=http://127.0.0.1:8004/v1/audio/transcriptions
  ASR_MODEL=fireredasr2
  DIARIZATION_API_URL=http://127.0.0.1:8006/v1/audio/diarizations
然后 batch_transcribe_nemotron.py 使用本客户端。
"""

import logging
import time
from pathlib import Path

import httpx

from config import Settings
from firered_client import FireRedServiceClients, UpstreamError

logger = logging.getLogger(__name__)


class FireRedNemotronClients(FireRedServiceClients):
    """ASR 复用 FireRedServiceClients.recognize；diarize 换成 Nemotron segments 格式。"""

    def diarize(self, path: Path, speaker_options: dict) -> list:
        url = self.settings.diarization_api_url
        if not url.startswith(("http://", "https://")):
            raise UpstreamError(
                "请配置完整的 DIARIZATION_API_URL，例如 http://127.0.0.1:8006/v1/audio/diarizations"
            )
        headers = (
            {"Authorization": f"Bearer {self.settings.internal_api_key}"}
            if self.settings.internal_api_key
            else {}
        )
        # Nemotron 服务支持 num_speakers/min_speakers/max_speakers（与社区-1 同名表单字段）
        data = {key: str(value) for key, value in speaker_options.items()}
        started = time.perf_counter()
        try:
            with path.open("rb") as audio:
                logger.info(
                    "Nemotron-Diarization 上传文件 %.1f MB，等待响应",
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
                f"Nemotron-Diarization 服务连接失败或超时：{type(exc).__name__}；"
                "请检查 DIARIZATION_API_URL 和服务状态"
            ) from exc

        if response.is_error:
            try:
                body = response.json()
                detail = body.get("error", body.get("detail", "")) if isinstance(body, dict) else ""
            except ValueError:
                detail = response.text
            raise UpstreamError(
                f"Nemotron-Diarization 返回 HTTP {response.status_code}：{self._safe_detail(detail)}"
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise UpstreamError("Nemotron-Diarization 没有返回 JSON") from exc

        segments = body.get("segments")
        if not isinstance(segments, list):
            raise UpstreamError("Nemotron-Diarization 响应缺少 segments 数组")

        # 归一化为 pipeline 期望的 [{"start","end","speaker"(str)}]
        turns = []
        for s in segments:
            if not isinstance(s, dict):
                continue
            try:
                start = float(s["start"])
                end = float(s["end"])
            except (KeyError, TypeError, ValueError):
                continue
            spk = s.get("speaker_label", s.get("speaker"))
            spk = str(spk) if spk is not None else "speaker_0"
            turns.append({"start": start, "end": end, "speaker": spk})

        raw_count = len(turns)
        turns = _make_exclusive(turns)
        logger.info(
            "Nemotron-Diarization 结果读取完成，共 %d 段原文 → 消解重叠后 %d 段互斥时间段，耗时 %.2f 秒",
            raw_count,
            len(turns),
            time.perf_counter() - started,
        )
        return turns


def _make_exclusive(turns: list, eps: float = 1e-6) -> list:
    """把含重叠的说话人时间段转成互斥（exclusive）时间线。

    pipeline.validate_turns 要求相邻段不可重叠（Community-1 的 exclusive 口径）。
    Nemotron 允许重叠（多人同时说话）。这里按时间边界切分：
      对每个小时间片，分配给「在该片内有覆盖、且起始最早」的说话人；
    多人重叠时采用「先到者优先」（arrival-order，与 Sortformer 的说话人排序一致）。
    相邻同说话人片段合并。
    """
    if not turns:
        return []
    # 收集所有边界点
    points = sorted({round(t["start"], 3) for t in turns} | {round(t["end"], 3) for t in turns})
    out = []
    for a, b in zip(points, points[1:]):
        if b - a <= eps:
            continue
        # 找覆盖 [a,b] 的说话人，选"起始最早"的（先到者优先）
        candidates = [t for t in turns if t["start"] <= a + eps and t["end"] >= b - eps]
        if not candidates:
            continue
        chosen = min(candidates, key=lambda t: (t["start"], t["end"]))
        spk = chosen["speaker"]
        # 与上一段合并（同说话人且时间连续）
        if out and out[-1]["speaker"] == spk and abs(out[-1]["end"] - a) <= eps:
            out[-1]["end"] = b
        else:
            out.append({"start": a, "end": b, "speaker": spk})
    return out
