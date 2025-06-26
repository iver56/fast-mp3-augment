use lame::{lame_t, MPEG_mode};
use lame_sys as lame;
use std::sync::mpsc::Sender;

const BLOCK_FRAMES: usize = 1152;

pub fn encode_segment_stream(
    pcm: &[f32],
    channels: i32,
    sample_rate: i32,
    bitrate_kbps: i32,
    quality: i32,
    tx: Sender<Vec<u8>>,
) -> Result<(i32, i32), String> {
    unsafe {
        let gfp: lame_t = lame::lame_init();
        if gfp.is_null() {
            return Err("lame_init failed".into());
        }
        lame::lame_set_in_samplerate(gfp, sample_rate);
        lame::lame_set_out_samplerate(gfp, sample_rate);
        lame::lame_set_num_channels(gfp, channels);
        lame::lame_set_brate(gfp, bitrate_kbps);
        lame::lame_set_quality(gfp, quality);

        // TODO: Instead of disabling the bit reservoir, implement proper support for it.
        lame::lame_set_disable_reservoir(gfp, 1);

        let mode = if channels == 1 { MPEG_mode::MONO } else { MPEG_mode::STEREO };
        lame::lame_set_mode(gfp, mode);

        if lame::lame_init_params(gfp) < 0 {
            return Err("lame_init_params failed".into());
        }

        let total_frames = pcm.len() / channels as usize;
        let mut offset = 0usize;

        while offset < total_frames {
            let nframes = (total_frames - offset).min(BLOCK_FRAMES);
            let mut mp3_buf = vec![0u8; ((nframes as f32 * 1.25) as usize) + 7200];

            let left = pcm.as_ptr().add(offset);
            let right = if channels == 1 {
                std::ptr::null()
            } else {
                pcm.as_ptr().add(total_frames + offset)
            };

            let used = lame::lame_encode_buffer_ieee_float(
                gfp,
                left,
                right,
                nframes as i32,
                mp3_buf.as_mut_ptr(),
                mp3_buf.len() as i32,
            );
            if used < 0 {
                return Err(format!("encode error {used}"));
            }
            if used > 0 {
                mp3_buf.truncate(used as usize);
                let _ = tx.send(mp3_buf);
            }
            offset += nframes;
        }

        // Flush any remaining samples
        let mut tail = vec![0u8; 7200];
        let written =
            lame::lame_encode_flush(gfp, tail.as_mut_ptr(), tail.len() as i32);
        if written < 0 {
            return Err(format!("flush error {written}"));
        }
        tail.truncate(written as usize);
        if !tail.is_empty() {
            let _ = tx.send(tail);
        }

        let delay = lame::lame_get_encoder_delay(gfp);
        let padding = lame::lame_get_encoder_padding(gfp);
        lame::lame_close(gfp);
        drop(tx); // close the channel

        Ok((delay, padding))
    }
}
