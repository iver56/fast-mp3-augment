mod channel_reader;
mod encoder;

use channel_reader::ChannelReader;
use encoder::encode_segment_stream;
use minimp3::{Decoder as Mp3Dec, Error as Mp3Err, Frame};

use numpy::{IntoPyArray, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn};
use pyo3::{exceptions::PyValueError, prelude::*, Bound};

use std::sync::mpsc;
use std::thread;

fn interleaved_to_planar(src: &[f32], ch: usize) -> Vec<f32> {
    let frames = src.len() / ch;
    let mut dst = vec![0.0; src.len()];
    for f in 0..frames {
        for c in 0..ch {
            dst[c * frames + f] = src[f * ch + c];
        }
    }
    dst
}

/// Down-mix interleaved multichannel to mono (take left)
fn downmix_left(src: &[f32], ch: usize) -> Vec<f32> {
    src.chunks_exact(ch).map(|frm| frm[0]).collect()
}

#[pyfunction]
#[pyo3(text_signature = "(samples, sample_rate, bitrate_kbps, post_gain, preserve_delay, /)")]
fn compress_roundtrip<'py>(
    py: Python<'py>,
    samples: PyReadonlyArrayDyn<'py, f32>,
    sample_rate: u32,
    bitrate_kbps: u32,
    post_gain: f32,
    preserve_delay: bool,
    quality: u32,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let pcm = samples.as_array();
    let (in_ch, frames) = match pcm.ndim() {
        1 => (1usize, pcm.len()),
        2 => (pcm.shape()[0] as usize, pcm.shape()[1]),
        _ => return Err(PyValueError::new_err("array must be 1-D or 2-D")),
    };
    let buf = pcm.to_owned().into_raw_vec();

    // Spawn streaming encoder (separate thread)
    let (tx, rx) = mpsc::channel::<Vec<u8>>();
    let enc_buf = buf.clone();
    let enc_handle = thread::spawn(move || {
        encode_segment_stream(
            &enc_buf,
            in_ch as i32,
            sample_rate as i32,
            bitrate_kbps as i32,
            quality as i32,
            tx,
        )
    });

    // Streaming decode
    let mut dec = Mp3Dec::new(ChannelReader::new(rx));
    let mut dec_inter = Vec::<f32>::with_capacity(frames * in_ch);
    let mut dec_ch = 0usize;

    loop {
        match dec.next_frame() {
            Ok(Frame { data, channels, .. }) => {
                dec_ch = channels;
                dec_inter.extend(
                    data.iter()
                        .map(|&s| (s as f32 / i16::MAX as f32) * post_gain),
                );
            }
            Err(Mp3Err::Eof) => break,
            Err(e) => return Err(PyValueError::new_err(e.to_string())),
        }
    }
    if dec_ch == 0 {
        return Err(PyValueError::new_err("no audio frames"));
    }

    // Collect encoder stats
    let (delay, pad) = match enc_handle.join() {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => return Err(PyValueError::new_err(e)),
        Err(_) => return Err(PyValueError::new_err("encoder thread panicked")),
    };

    const FILTER_DELAY: usize = 529;

    let processed = if preserve_delay {
        dec_inter
    } else {
        let mut v = dec_inter;

        // Drop encoder delay + synthesis filter pre-roll in decoder
        let start = (delay as usize + FILTER_DELAY) * dec_ch;
        if v.len() < start {
            return Err(PyValueError::new_err("decoded stream shorter than delay"));
        }
        v.drain(..start);

        let end_pad = (pad as usize) * dec_ch;
        if end_pad != 0 && v.len() > end_pad {
            v.truncate(v.len() - end_pad);
        }

        let need = frames * dec_ch;

        // If output is longer than needed, chop off one frame at the front, for alignment
        if v.len() > need {
            let frame_samples = if sample_rate >= 32_000 { 1152 } else { 576 };
            let extra = frame_samples * dec_ch;
            if v.len() > extra {
                v.drain(..extra);
            }
        }

        // Resize to exact length
        if v.len() < need {
            v.resize(need, 0.0);
        } else if v.len() > need {
            v.truncate(need);
        }
        v
    };

    let frames_out = processed.len() / dec_ch;

    let out_planar = match (in_ch, dec_ch) {
        (1, 1) => processed,
        (1, _) => downmix_left(&processed, dec_ch),
        (_, _) => interleaved_to_planar(&processed, dec_ch),
    };

    // Create NumPy view
    let arr = if in_ch == 1 {
        // Mono/1D
        out_planar.into_pyarray(py).to_dyn().clone()
    } else {
        // Stereo/2D
        out_planar
            .into_pyarray(py)
            .reshape([in_ch, frames_out])?
            .to_dyn()
            .clone()
    };
    Ok(arr)
}

#[pymodule]
fn _mp3augment(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compress_roundtrip, m)?)?;
    Ok(())
}
