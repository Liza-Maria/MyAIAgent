use std::cmp::Ordering;

use super::{ StoreError, SearchResult };

pub struct Document {
    pub id: String,
    pub text: String,
    pub embedding: Vec<f32>,
}

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