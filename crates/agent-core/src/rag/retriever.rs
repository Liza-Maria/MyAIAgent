use super::{
    RetrieveError,
    SearchResult,
    Document,
    VectorStore,
    Embedder,
    EmbedError};

pub struct Retriever {
    embedder: Box<dyn Embedder>,
    store: VectorStore,
}

impl Retriever {
    pub fn new(embedder: Box<dyn Embedder>) -> Self {
        Self {
            embedder,
            store: VectorStore::new(),
        }
    }

    pub async fn index(&mut self, id: &str, text: &str) -> Result<(), RetrieveError> {
        let embedding = self.embedder.embed(&text).await?;

        let document = Document {
            id: id.to_string(),
            text: text.to_string(),
            embedding,
        };

        self.store.add(document)?;

        Ok(())
    }

    pub async fn retrieve(&self, query: &str, top_k: usize)
            -> Result<Vec<SearchResult>, RetrieveError> {
        let embedding = self.embedder.embed(&query).await?;

        let result = self.store.search(&embedding, top_k);

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FakeEmbedder;

    #[async_trait::async_trait]
    impl Embedder for FakeEmbedder {
        async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
            if text.contains("cat") {
                return Ok(vec![1.0, 0.0]);
            } else if text.contains("dog") {
                return Ok(vec![0.0, 1.0]);
            }

            Err(EmbedError::InvalidResponse)
        }
    }

    #[tokio::test]
    async fn retrieve_returns_most_similar() {
        let embedder = Box::new(FakeEmbedder);
        let mut retriever = Retriever::new(embedder);

        retriever.index("c", "cat").await.unwrap();
        retriever.index("d", "dog").await.unwrap();

        let results = retriever.retrieve("cat query", 2).await.unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "c");
    }

    #[tokio::test]
    async fn retrieve_on_empty_index_returns_empty() {
        let embedder = Box::new(FakeEmbedder);
        let retriever = Retriever::new(embedder);

        let results = retriever.retrieve("cat query", 3).await.unwrap();

        assert!(results.is_empty());
    }
}