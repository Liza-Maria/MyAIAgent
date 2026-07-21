use thiserror::Error;

pub mod store;
pub mod embedder;
pub mod retriever;

pub use retriever::Retriever;
pub use embedder::{OllamaEmbedder};
pub use store::{Document, VectorStore, cosine_similarity};

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

#[derive(Debug, Error)]
pub enum RetrieveError {
    #[error("embedding failed: {0}")]
    Embed(#[from] EmbedError),

    #[error("store failed: {0}")]
    Store(#[from] StoreError),
}

#[derive(Debug, Error)]
pub enum StoreError {
    #[error("Embedding cannot be empty.")]
    EmptyEmbedding,

    #[error("Embedding dimensions: expected {expected}, got {actual}")]
    DimensionMismatch {
        expected: usize,
        actual: usize,
    }
}

pub struct SearchResult {
    pub id: String,
    pub text: String,
    pub score: f32,
}

#[async_trait::async_trait]
pub trait Embedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError>;
}