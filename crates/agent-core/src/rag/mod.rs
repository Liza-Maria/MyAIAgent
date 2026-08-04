use thiserror::Error;
use std::fmt::Debug;

pub mod store;
pub mod embedder;
pub mod retriever;
pub mod chunker;
pub mod persist;

pub use retriever::Retriever;
pub use embedder::{OllamaEmbedder};
pub use store::{Document, VectorStore, cosine_similarity};
pub use persist::{ PersistentIndex, INDEX_VERSION };

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

    #[error("Could not read path: {path} source: {source}")]
    Io {
        path: String,
        source: std::io::Error,
    }
}

#[derive(Debug, Error)]
pub enum StoreError {
    #[error("Embedding cannot be empty.")]
    EmptyEmbedding,

    #[error("Embedding dimensions: expected {expected}, got {actual}")]
    DimensionMismatch {
        expected: usize,
        actual: usize,
    },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("embedding model mismatch: expected {expected} actual {actual}")]
    ModelMismatch {
        expected: String,
        actual: String,
    },

    #[error("Unsupported index version: {version}")]
    UnsupportedVersion {
        version: u32,
    }
}

pub struct SearchResult {
    pub id: String,
    pub text: String,
    pub score: f32,
}

#[async_trait::async_trait]
pub trait Embedder: Debug + Send + Sync {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError>;
    fn model(&self) -> &str;
}