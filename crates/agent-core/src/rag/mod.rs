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
}

#[async_trait::async_trait]
pub trait Embedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError>;
}