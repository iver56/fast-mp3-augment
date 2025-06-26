import time
from pathlib import Path

import fast_align_audio
import numpy as np
import pytest
import soundfile
import scipy.signal
import fast_mp3_augment

TEST_FIXTURES_PATH = Path(__file__).resolve().parent.parent / "test_fixtures"


def get_chirp_test(sample_rate, duration):
    """Create a `duration` seconds chirp from 0 Hz to Nyquist frequency"""
    n = np.arange(0, duration, 1 / sample_rate)
    samples = scipy.signal.chirp(n, 0, duration, sample_rate // 2, method="linear")
    return samples.astype(np.float32)


def test_mono_1d():
    audio, sample_rate = soundfile.read(
        TEST_FIXTURES_PATH / "p286_011.wav", dtype=np.float32
    )
    assert audio.ndim == 1
    augmented_audio = fast_mp3_augment.compress_roundtrip(
        audio, sample_rate, bitrate_kbps=64
    )
    assert augmented_audio.shape == audio.shape
    assert augmented_audio.dtype == augmented_audio.dtype

    offset, mse = fast_align_audio.find_best_alignment_offset(
        reference_signal=audio,
        delayed_signal=augmented_audio,
        max_offset_samples=3000,
    )
    assert mse <= 5e-5
    assert offset == 0


def test_mono_1d_preserve_delay():
    audio, sample_rate = soundfile.read(
        TEST_FIXTURES_PATH / "p286_011.wav", dtype=np.float32
    )
    assert audio.ndim == 1
    augmented_audio = fast_mp3_augment.compress_roundtrip(
        audio,
        sample_rate,
        bitrate_kbps=64,
        preserve_delay=True,
    )

    assert augmented_audio.shape[-1] > audio.shape[-1]
    assert augmented_audio.dtype == augmented_audio.dtype

    offset, mse = fast_align_audio.find_best_alignment_offset(
        reference_signal=audio,
        delayed_signal=augmented_audio,
        max_offset_samples=3000,
    )
    assert offset > 0

    padding = augmented_audio.shape[-1] - audio.shape[-1] - offset
    assert padding > 0


def test_mono_2d():
    audio, sample_rate = soundfile.read(
        TEST_FIXTURES_PATH / "p286_011.wav", dtype=np.float32
    )
    audio = np.expand_dims(audio, 0)
    assert audio.ndim == 2
    augmented_audio = fast_mp3_augment.compress_roundtrip(
        audio, sample_rate, bitrate_kbps=64
    )
    assert augmented_audio.shape == audio.shape
    assert augmented_audio.dtype == augmented_audio.dtype


def test_stereo():
    audio, sample_rate = soundfile.read(
        TEST_FIXTURES_PATH / "perfect-alley1.ogg", dtype=np.float32
    )
    audio = np.ascontiguousarray(audio.T)
    augmented_audio = fast_mp3_augment.compress_roundtrip(
        audio, sample_rate, bitrate_kbps=64
    )

    assert augmented_audio.shape == audio.shape
    assert augmented_audio.dtype == augmented_audio.dtype

    offset, mse = fast_align_audio.find_best_alignment_offset(
        reference_signal=audio[0],
        delayed_signal=augmented_audio[0],
        max_offset_samples=3000,
    )
    assert mse <= 5e-5
    assert offset == 0


def test_stereo_preserve_delay():
    audio, sample_rate = soundfile.read(
        TEST_FIXTURES_PATH / "perfect-alley1.ogg", dtype=np.float32
    )
    audio = np.ascontiguousarray(audio.T)
    augmented_audio = fast_mp3_augment.compress_roundtrip(
        audio, sample_rate, bitrate_kbps=64, preserve_delay=True
    )

    assert augmented_audio.shape[0] == audio.shape[0]
    assert augmented_audio.shape[1] > audio.shape[1]
    assert augmented_audio.dtype == augmented_audio.dtype

    offset, mse = fast_align_audio.find_best_alignment_offset(
        reference_signal=audio[0],
        delayed_signal=augmented_audio[0],
        max_offset_samples=3000,
    )
    assert offset > 0

    padding = augmented_audio.shape[1] - audio.shape[1] - offset
    assert padding > 0


def test_stereo_wrong_dimension_order():
    audio, sample_rate = soundfile.read(
        TEST_FIXTURES_PATH / "perfect-alley1.ogg", dtype=np.float32
    )
    assert audio.shape[1] == 2
    with pytest.raises(fast_mp3_augment.WrongMultichannelAudioShape):
        fast_mp3_augment.compress_roundtrip(audio, sample_rate, bitrate_kbps=64)


def test_3_channels():
    sample_rate = 48000
    audio = np.random.uniform(-1, 1, (3, 48000)).astype("float32")
    with pytest.raises(fast_mp3_augment.UnsupportedNumberOfChannels):
        fast_mp3_augment.compress_roundtrip(audio, sample_rate, bitrate_kbps=64)


def test_too_short():
    sample_rate = 44100
    audio = np.random.uniform(-1, 1, (1, 100)).astype("float32")
    with pytest.raises(fast_mp3_augment.TooShortAudio):
        fast_mp3_augment.compress_roundtrip(audio, sample_rate, bitrate_kbps=64)


def test_transposed_but_not_contiguous_audio():
    audio, sample_rate = soundfile.read(
        TEST_FIXTURES_PATH / "perfect-alley1.ogg", dtype=np.float32
    )
    transposed_non_contiguous_array = audio.T
    with pytest.raises(fast_mp3_augment.NonContiguousAudio):
        fast_mp3_augment.compress_roundtrip(
            transposed_non_contiguous_array, sample_rate, bitrate_kbps=64
        )


def test_supported_sample_rates():
    sample_rates = (8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000)
    duration_s = 2

    for sample_rate in sample_rates:
        for bitrate_kbps in [
            8,
            16,
            24,
            32,
            40,
            48,
            56,
            64,
            80,
            96,
            112,
            128,
            144,
            160,
            192,
            224,
            256,
            320,
        ]:
            for num_channels in (1, 2):
                sig = get_chirp_test(sample_rate, duration_s)
                sig = np.ascontiguousarray(sig.reshape((num_channels, -1)))
                augmented_sig = fast_mp3_augment.compress_roundtrip(
                    sig, sample_rate, bitrate_kbps=bitrate_kbps
                )
                assert augmented_sig.shape == sig.shape
                offset, mse = fast_align_audio.find_best_alignment_offset(
                    reference_signal=sig[0],
                    delayed_signal=augmented_sig[0],
                    max_offset_samples=3000,
                )
                assert offset == 0

                corr = fast_align_audio.alignment.fast_autocorr(
                    sig.flatten(), augmented_sig.flatten(), t=0
                )
                assert not np.any(np.isnan(augmented_sig))
                if bitrate_kbps == 96:
                    assert corr > 0.75


def test_unsupported_sample_rate():
    sig = np.zeros(4500, dtype=np.float32)
    with pytest.raises(fast_mp3_augment.UnsupportedSampleRate):
        fast_mp3_augment.compress_roundtrip(sig, sample_rate=1337, bitrate_kbps=64)


def test_supported_bitrates():
    bitrates = {
        8,
        16,
        24,
        32,
        40,
        48,
        56,
        64,
        80,
        96,
        112,
        128,
        144,
        160,
        192,
        224,
        256,
        320,
    }
    duration_s = 2
    sample_rate = 44100
    for num_channels in (1, 2):
        for bitrate_kbps in bitrates:
            sig = 0.5 * np.sin(
                2
                * np.pi
                * 440
                * np.linspace(0, 5, sample_rate * duration_s, dtype=np.float32)
            )
            sig = sig.reshape((num_channels, -1))
            augmented_sig = fast_mp3_augment.compress_roundtrip(
                sig, sample_rate, bitrate_kbps=bitrate_kbps
            )
            assert augmented_sig.shape == sig.shape
            assert not np.any(np.isnan(augmented_sig))


def test_unsupported_bitrate():
    sig = np.zeros(4500, dtype=np.float32)
    with pytest.raises(fast_mp3_augment.UnsupportedBitrate):
        fast_mp3_augment.compress_roundtrip(sig, sample_rate=44100, bitrate_kbps=42)


def test_quality_values():
    quality_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    sig = np.random.uniform(-1, 1, 20000).astype("float32")
    execution_times = []
    for quality in quality_values:
        start_t = time.time()
        augmented_sig = fast_mp3_augment.compress_roundtrip(
            sig, sample_rate=44100, bitrate_kbps=64, quality=quality
        )
        execution_time = time.time() - start_t
        execution_times.append(execution_time)
        assert augmented_sig.shape == sig.shape

    # quality = 0 is slower than quality = 9
    assert execution_times[0] > execution_times[-1]


def test_unsupported_quality_value():
    sig = np.zeros(4500, dtype=np.float32)
    with pytest.raises(fast_mp3_augment.UnsupportedQuality):
        fast_mp3_augment.compress_roundtrip(
            sig, sample_rate=44100, bitrate_kbps=64, quality=42
        )
