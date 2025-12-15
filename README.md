# fast-mp3-augment

A fast Python library for MP3 encoder + decoder data augmentation. Made for integration with [audiomentations](https://github.com/iver56/audiomentations/). Intentionally applying audio degradation by lossy compression help machine learning models learn to deal with audio that gets streamed from various internet services, which is commonly lossy/compressed.

# Installation

[![PyPI version](https://img.shields.io/pypi/v/fast-mp3-augment.svg?style=flat)](https://pypi.org/project/fast-mp3-augment/)
![python 3.10, 3.11, 3.12, 3.13, 3.14](https://img.shields.io/badge/Python-3.10%20|%203.11%20|%203.12%20|%203.13%20|%203.14-blue)
![os: Linux, macOS, Windows](https://img.shields.io/badge/OS-Linux%20%28arm%20%26%20x86--64%29%20|%20macOS%20%28arm%29%20|%20Windows%20%28x86--64%29-blue)

```
$ pip install fast-mp3-augment
```

## Code example

```
import numpy as np

import fast_mp3_augment

audio = np.random.uniform(-1, 1, (2, 2 * 48000)).astype("float32")
augmented_audio = fast_mp3_augment.compress_roundtrip(
    audio, sample_rate=48000, bitrate_kbps=64, preserve_delay=False, quality=7
)
```

## Features

* The output is perfectly aligned (no delay/offset and padding) with the input by default, but this trimming behavior can be disabled (with `preserve_delay=True`)
* Supports mono and stereo
* Supports standard MP3 bitrates (8-320 kbps)
* Supports common sample rates (8-48 kHz)
* Inputs and outputs float32 numpy array
* Adjustable `quality` parameter for various tradeoffs between speed and audio quality

## Performance

This library is largely developed with Rust under the hood (via pyo3 & maturin), and applies a few nice little tricks for achieving speedy execution (which is important during large-scale audio ML training!), such as:

* Fast numpy array interop between Python and rust
* In-memory computations (no disk I/O)
* SIMD-optimized max abs calculation (for avoiding clipping distortion)
* Pipelining/streaming (LAME encoder and minimp3 decoder in separate threads)

A quick performance benchmark (based on demo.py in audiomentations), which augmented 3 short (~7-9 sec) audio snippets (2 mono, 1 stereo) on a laptop with i7-13700HX and a 2 TB Samsung PM9A1 NVMe shows that fast-mp3-augment is superior when it comes to speed:

![perf_benchmark_results.png](https://raw.githubusercontent.com/iver56/fast-mp3-augment/master/images/perf_benchmark_results.png)

## Changelog

## [0.2.0] - 2025-12-15

* Bump min Python version to 3.10
* Officially add support for Python 3.14
* Make builds slightly slimmer

For the complete changelog, go to [CHANGELOG.md](CHANGELOG.md)

## Development setup

* `conda create --name fast-mp3-augment python=3.11`
* `conda activate fast-mp3-augment`
* `pip install -r dev_requirements.txt`
* `maturin develop`
* `pytest`

## LAME note

fast_mp3_augment statically links libmp3lame 3.100 (LGPL-2.1-or-later). Full source is available [here](https://downloads.sourceforge.net/project/lame/lame/3.100/lame-3.100.tar.gz). To rebuild the wheel against a modified LAME, see [mp3lame-sys](https://crates.io/crates/mp3lame-sys)
