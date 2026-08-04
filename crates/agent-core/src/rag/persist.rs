use serde::{Deserialize, Serialize};
use super::Document;

pub const INDEX_VERSION: u32 = 1;

#[derive(Debug, Serialize, Deserialize)]
pub struct PersistentIndex {
    pub version: u32,
    pub embedding_model: String,
    pub dimensions: usize,
    pub documents: Vec<Document>,
}

