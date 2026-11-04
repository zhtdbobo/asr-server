import shutil

import numpy as np
import pytest
import soundfile as sf

from audio import AudioInputError, decode_audio

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def test_real_ffmpeg_stereo_resampling(tmp_path):
    path = tmp_path / "中文录音.wav"
    source = np.full((48000, 2), [0.1, 0.3], dtype=np.float32)
    sf.write(path, source, 48000, subtype="FLOAT")
    waveform, sr = decode_audio(path, 2)
    assert sr == 16000
    assert waveform.shape == (16000,)
    assert waveform.dtype == np.float32
    # FFmpeg downmixes stereo to mono using its default channel mix.
    assert np.all(waveform > 0)
    assert np.isfinite(waveform).all()


def test_duration_limit_is_checked_without_silent_truncation(tmp_path):
    path = tmp_path / "long.wav"
    sf.write(path, np.zeros(16000 * 2, dtype=np.float32), 16000)
    with pytest.raises(AudioInputError, match="超过"):
        decode_audio(path, 1)


def test_bad_and_missing_audio(tmp_path):
    path = tmp_path / "bad.wav"
    path.write_bytes(b"not an audio file")
    with pytest.raises(AudioInputError, match="无法解码"):
        decode_audio(path, 2)
    with pytest.raises(AudioInputError, match="不存在"):
        decode_audio(tmp_path / "missing.wav", 2)
