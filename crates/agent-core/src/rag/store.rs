use super::StoreError;

pub struct Document {
    pub id: String,
    pub text: String,
    pub embedding: Vec<32>,
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
        let actual_len = document.len();

        if actual == 0 {
            return Err(StoreError::EmptyEmbedding);
        }

        if let Some(first_doc) = self.documents.first() {
            let expected_len = first_doc.len();

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
}