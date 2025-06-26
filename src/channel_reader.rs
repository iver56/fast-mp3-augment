use std::{
    collections::VecDeque,
    io::{Read, Result},
    sync::mpsc::Receiver,
};

pub struct ChannelReader {
    rx: Receiver<Vec<u8>>,
    buf: VecDeque<u8>,
}

impl ChannelReader {
    #[inline]
    pub fn new(rx: Receiver<Vec<u8>>) -> Self {
        Self {
            rx,
            buf: VecDeque::new(),
        }
    }
}

impl Read for ChannelReader {
    fn read(&mut self, out: &mut [u8]) -> Result<usize> {
        // Refill internal buffer if needed
        while self.buf.is_empty() {
            match self.rx.recv() {
                Ok(chunk) => self.buf.extend(chunk),
                Err(_) => break, // sender closed -> EOF once buffer empty
            }
        }

        let n = out.len().min(self.buf.len());
        for (dst, src) in out.iter_mut().take(n).zip(self.buf.drain(..n)) {
            *dst = src;
        }
        Ok(n)
    }
}
