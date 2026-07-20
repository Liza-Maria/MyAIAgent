use thiserror::Error;

pub mod store;
pub mod embedder;
pub use store::cosine_similarity;

#[derive(Error, Debug)]
pub enum EmbedError {
    #[error("embedding transport failed: {0}")]
    Transport(String),

    #[error("embedding response was empty or invalid")]
    InvalidResponse,

    #[error("embedding API returned status {status}: {body}")]
    Api {
        status: u16,
        body: String,
    },

    #[error("Unable to decode embedding response {0}")]
    Decode(String),
}

pub enum StoreError {
    #[error("Embedding cannot be empty.")]
    EmptyEmbedding,

    #[error("Embedding dimentions: expected {expected}, got {actual}")]
    DimensionMismatch {
        expected: usize,
        actual: usize,
    }
}

#[async_trait::async_trait]
pub trait Embedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError>;
}