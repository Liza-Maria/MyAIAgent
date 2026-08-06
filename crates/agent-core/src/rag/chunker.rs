use std::cmp::min;
use std::fmt::Debug;

pub trait Chunker: Debug + Send + Sync {
    fn chunk(&self, text: &str) -> Vec<String>;
}

#[derive(Debug)]
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
        if text.is_empty() || self.chunk_size == 0 {
            return Vec::new();
        }

        let mut chunks = Vec::new();

        let characters: Vec<char> = text.chars().collect();
        let l = characters.len();

        assert!(self.chunk_size > self.overlap, "Chunk_size should be grater than overlap");
        let step = self.chunk_size - self.overlap;

        let mut start = 0;
        
        while start < l {
            let end = min(start + self.chunk_size, l);
            
            let chunk: String = characters[start..end].iter().collect();
            chunks.push(chunk);
            
            if end == l {
                break;
            }

            start += step;
        }

        chunks
    }
}