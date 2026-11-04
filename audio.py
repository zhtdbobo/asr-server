import shutil
import subprocess
import tempfile
from pathlib import Path

SAMPLE_RATE = 16000


class AudioInputError(ValueError):
    pass


def decode_audio(path: str | Path, max_seconds: float):
    """Decode audio/video once; both models receive exactly the same mono 16 kHz samples."""
    import numpy as np
    import soundfile as sf

    source = Path(path).resolve()
    if not source.is_file():
        raise AudioInputError(f"音频文件不存在: {source}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("未找到 ffmpeg，请先安装并加入 PATH，或使用 Docker 部署")
    with tempfile.TemporaryDirectory(prefix="meeting-audio-") as directory:
        target = Path(directory) / "decoded.wav"
        try:
            result = subprocess.run(
                [
                    ffmpeg,
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-protocol_whitelist",
                    "file,pipe",
                    "-i",
                    str(source),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    str(SAMPLE_RATE),
                    "-t",
                    str(max_seconds + 1),
                    "-c:a",
                    "pcm_f32le",
                    str(target),
                ],
                capture_output=True,
                timeout=300,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise AudioInputError("音频解码超时，请检查文件或缩短录音") from exc
        if result.returncode:
            raise AudioInputError("无法解码音频，请上传含有效音轨的 WAV、MP3、M4A 或 MP4 文件")
        waveform, sample_rate = sf.read(target, dtype="float32")
    if waveform.size == 0:
        raise AudioInputError("音频为空")
    if waveform.size / sample_rate > max_seconds:
        raise AudioInputError(f"音频时长超过 {max_seconds:g} 秒限制")
    if not np.isfinite(waveform).all():
        raise AudioInputError("音频含无效采样值")
    peak = float(np.max(np.abs(waveform)))
    if peak > 1:
        waveform /= peak
    return np.ascontiguousarray(waveform), sample_rate
