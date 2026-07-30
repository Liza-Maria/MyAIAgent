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
    
}