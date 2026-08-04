use std::cmp::min;

pub trait Chunker: Debug + Send + Async {
    fn chunk(&self, text: &str) -> Vec<String>;
}

pub struct FixedChunkSize {
    pub chunk_size: usize,
    pub overlap: usize,
}

impl FixedChunkSize {
    pub fn new(chunk_size: usize, overlap: usize) -> Self {
        Self {
            chunk_size,
            overlap,
        }
    }
}

impl Chunker for FixedChunkSize {
    fn chunk(&self, text: &str) -> Vec<String> {
        if text.is_empty() {
            return Vec::new();
        }

        let mut chunks = Vec::new();
        let l = text.len();
        let step = self.chunk_size - self.overlap;

        let mut start = 0;

        while start < l {
            let end = min(start + self.chunk_size, l);

            let chunk = text[start..end].to_string();
            chunks.push(chunk);

            if end == l {
                break;
            }

            start += step;
        }

        chunks
    }
}