from typing import Final

import numpy as np
from numpy.typing import NDArray
import numpy_minmax

from ._mp3augment import compress_roundtrip as _compress_roundtrip


class WrongMultichannelAudioShape(Exception):
    pass


class UnsupportedNumberOfChannels(Exception):
    pass


class TooShortAudio(Exception):
    pass


class NonContiguousAudio(Exception):
    pass


class UnsupportedSampleRate(Exception):
    pass


class UnsupportedBitrate(Exception):
    pass


class UnsupportedQuality(Exception):
    pass


SUPPORTED_SAMPLE_RATES = {8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000}
SUPPORTED_BITRATES = {
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
SUPPORTED_QUALITY_VALUES = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}


def compress_roundtrip(
    samples: NDArray[np.float32],
    sample_rate: int,
    bitrate_kbps: int,
    preserve_delay: bool = False,
    quality: int = 7,
) -> NDArray[np.float32]:
    """
    Degrade the given float32 audio by encoding it as MP3 and then decoding it back to a float32 audio signal.

    :param samples: 1D or 2D samples with shape (channels, samples)
    :param sample_rate: Sample rate of the audio in Hertz
    :param bitrate_kbps: Constant bitrate (in kbps) to use when encoding the audio
    :param preserve_delay:
        If False (default), the output length and timing will match the input.
        If True, include LAME encoder delay + filter delay (a few tens of milliseconds) and padding in the output.
        This makes the output
        1) longer than the input
        2) delayed (out of sync) relative to the input
        Normally, it makes sense to set preserve_delay to False, but if you want outputs that include the
        short, almost silent part in the beginning, you here have the option to get that.
    :param quality: An int in range [0, 9].
        0: higher quality audio at the cost of slower processing
        9: fast processing at the cost of lower quality audio
    """
    if sample_rate not in SUPPORTED_SAMPLE_RATES:
        raise UnsupportedSampleRate(
            f"Expected sample_rate to be one of {SUPPORTED_SAMPLE_RATES}, but received {sample_rate}"
        )
    if bitrate_kbps not in SUPPORTED_BITRATES:
        raise UnsupportedBitrate(
            f"Expected bitrate_kbps to be one of {SUPPORTED_BITRATES}, but received {bitrate_kbps}"
        )
    if quality not in SUPPORTED_QUALITY_VALUES:
        raise UnsupportedQuality(
            f"Expected quality to be one of {SUPPORTED_QUALITY_VALUES}, but received {quality}"
        )

    original_ndim = samples.ndim
    if samples.ndim == 2:
        if samples.shape[0] > samples.shape[1]:
            raise WrongMultichannelAudioShape(
                "Multichannel audio must have channels first, not channels"
                " last. In other words, the shape must be (channels, samples),"
                " not (samples, channels). See"
                " https://iver56.github.io/audiomentations/guides/multichannel_audio_array_shapes/"
                " for more info."
            )
        if samples.shape[0] > 2:
            raise UnsupportedNumberOfChannels(
                f"Expected mono or stereo audio, but received {samples.shape[0]} channels, which is unsupported."
            )

        if not samples.flags.c_contiguous:
            raise NonContiguousAudio(
                f"The given NDArray must be C-contiguous. You can make it C-contiguous by using `np.ascontiguousarray(samples)`"
            )

    if samples.shape[-1] < 0.1 * sample_rate:
        raise TooShortAudio(
            f"The input audio {samples.shape[-1]} samples is too short. It needs to be at least 100 ms."
        )

    # Possibly gain down to avoid clipping
    min_amplitude, max_amplitude = numpy_minmax.minmax(samples)
    max_abs_amplitude = max(abs(min_amplitude), abs(max_amplitude))
    post_gain = 1.0
    if max_abs_amplitude > 1.0:
        factor = 1.0 / max_abs_amplitude
        samples = samples * factor
        post_gain = max_abs_amplitude

    result = _compress_roundtrip(
        samples,
        sample_rate,
        bitrate_kbps,
        post_gain=post_gain,
        preserve_delay=preserve_delay,
        quality=quality,
    )
    if result.ndim == 1 and original_ndim == 2:
        result = result.reshape((1, -1))
    return result


__all__: Final = ["compress_roundtrip"]
