use std::cmp::Ordering;
use std::path::Path;
use serde::{ Serialize, Deserialize };
use super::{ INDEX_VERSION, PersistentIndex };

use super::{ StoreError, SearchResult };

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub text: String,
    pub embedding: Vec<f32>,
}

#[derive(Debug)]
pub struct VectorStore {
    pub documents: Vec<Document>,
}

impl VectorStore {
    pub fn new() -> Self {
        Self {
            documents: Vec::new(),
        }
    }

    pub fn add(&mut self, document: Document) -> Result<(), StoreError> {
        let actual_len = document.embedding.len();

        if actual_len == 0 {
            return Err(StoreError::EmptyEmbedding);
        }

        if let Some(first_doc) = self.documents.first() {
            let expected_len = first_doc.embedding.len();

            if actual_len != expected_len {
                return Err(StoreError::DimensionMismatch {
                    expected: expected_len,
                    actual: actual_len,
                });
            }
        }

        self.documents.push(document);

        Ok(())
    }

    pub fn search(&self, query_embedding: &[f32], top_k: usize) -> Vec<SearchResult> {
        let mut res: Vec<SearchResult> = self.documents
            .iter()
            .map(|document| SearchResult {
                id: document.id.clone(),
                text: document.text.clone(),
                score: cosine_similarity(query_embedding, &document.embedding),
            })
            .collect();

        res.sort_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(Ordering::Equal)
                .reverse()
        });

        res.truncate(top_k);

        res
    }

    pub fn save(&self, path: &Path, embedding_model: &str) -> Result<(), StoreError> {
        let dimensions = self.documents
                            .first()
                            .map(|doc| doc.embedding.len())
                            .unwrap_or(0);

        let index = PersistentIndex {
            version: INDEX_VERSION,
            embedding_model: embedding_model.to_string(),
            dimensions,
            documents: self.documents.clone(),
        };

        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }

        let json = serde_json::to_vec_pretty(&index)?;

        let name_path = path.with_extension("json.temp");

        std::fs::write(&name_path, json)?;

        std::fs::rename(&name_path, path)?;

        Ok(())
    }

    pub fn load(path: &Path, expected_model: &str) -> Result<Self, StoreError> {
        let bytes = std::fs::read(path)?;

        let index: PersistentIndex = serde_json::from_slice(&bytes)?;

        if index.embedding_model != expected_model {
            return Err(StoreError::ModelMismatch {
                expected: expected_model.to_string(),
                actual: index.embedding_model,
            });
        }

        if index.version != INDEX_VERSION {
            return Err(StoreError::UnsupportedVersion {
                version: index.version
            });
        }

        let mut store = VectorStore::new();

        for document in index.documents {
            store.add(document)?;
        }

        Ok(store)
    }
}
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }

    let dot = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>();

    let magnitude_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let magnitude_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if magnitude_a == 0.0 || magnitude_b == 0.0 {
        return 0.0;
    }

    dot / (magnitude_a * magnitude_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_vectors_have_similaryty_one() {
        let a = vec![0.1, 0.2, 0.3];
        let b = vec![0.1, 0.2, 0.3];

        let result = cosine_similarity(&a, &b);

        assert!((result - 1.0).abs() < 1e-6);
    }

    #[test]
    fn orthogonal_vectors_with_similarity_zero() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];

        let result = cosine_similarity(&a, &b);

        assert!(result.abs() < 1e-6);
    }

    #[test]
    fn orthogonal_vectors_with_similarity_minus_one() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];

        let result = cosine_similarity(&a, &b);

        assert!((result + 1.0).abs() < 1e-6);
    }

    #[test]
    fn search_returns_best_match_first() {
        let mut store = VectorStore::new();

        store.add(Document {
            id: "a".to_string(),
            text: "test 1".to_string(),
            embedding: vec![1.0, 0.0],
        }).unwrap();

        store.add(Document {
            id: "b".to_string(),
            text: "test 2".to_string(),
            embedding: vec![0.9, 0.1],
        }).unwrap();

        store.add(Document {
            id: "c".to_string(),
            text: "test 3".to_string(),
            embedding: vec![0.0, 1.0],
        }).unwrap();

        let results = store.search(&[1.0, 0.0], 3);

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].id, "a");
        assert_eq!(results[1].id, "b");
        assert_eq!(results[2].id, "c");
    }

    #[test]
    fn search_on_empty_store_returns_empty() {
        let store = VectorStore::new();

        let results = store.search(&[1.0, 0.0], 3);

        assert_eq!(results.len(), 0);
    }

    #[test]
    fn search_respects_top_k() {
        let mut store = VectorStore::new();

        store.add(Document {
            id: "a".to_string(),
            text: "test 1".to_string(),
            embedding: vec![1.0, 0.0],
        }).unwrap();

        store.add(Document {
            id: "b".to_string(),
            text: "test 2".to_string(),
            embedding: vec![0.9, 0.1],
        }).unwrap();

        store.add(Document {
            id: "c".to_string(),
            text: "test 3".to_string(),
            embedding: vec![0.0, 1.0],
        }).unwrap();

        let results = store.search(&[1.0, 0.0], 2);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "a");
        assert_eq!(results[1].id, "b");
    }

    #[test]
    fn add_rejects_mismatched_dimensions() {
        let mut store = VectorStore::new();

        store.add(Document {
            id: "a".to_string(),
            text: "three dimensions".to_string(),
            embedding: vec![1.0, 0.0, 1.0],
        }).unwrap();

        let res = store.add(Document {
            id: "b".to_string(),
            text: "2 dimensions".to_string(),
            embedding: vec![0.9, 0.1],
        });

        assert!(matches!(
            res,
            Err(StoreError::DimensionMismatch {
                expected: 3,
                actual: 2,
            })
        ));
    }
}