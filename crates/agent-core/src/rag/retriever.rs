use super::{
    RetrieveError,
    SearchResult,
    Document,
    VectorStore,
    Embedder,
    EmbedError};

    #[derive(Debug, Default, Clone, PartialEq)]
    pub struct IngestReport {
        pub files_indexed: usize,
        pub chunks_indexed: usize,
        pub files_skipped: usize,
    }

    #[derive(Debug)]
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

    pub fn save(&self, path: &Path) -> Result<(), RetriveError> {
        self.store.save(path, self.embedder.model())?;

        Ok(())
    }

    pub fn load(&mut self, path: &Path) -> Result<u32, RetrieveError> {
        let res = self.load(path, self.embedder.model())?;
        self.store = res;

        Ok(res.documents.len())
    }

    pub async fn index_file(&mut self, path: &Path) -> Result<(), RetriveError> {
        let text = std::fs::read_to_string(path)?;

        let id = path.to_string_lossy().to_string();

        self.index(&id, &text).await?;

        Ok(())
    }

    pub async fn index_directory(&mut self, root: &Path, extensions: &[&str]) 
        -> Result<IngestReport, RetriveError> 
        {
        let mut report = IngestReport::default();

        let metadata = std::fs::metadata(root)?;

        if metadata.is_dir() {
            let entries = std::fs::read_dir(root)?;

            for entry in entries {
                let file_path = entry.path();

                self.index_file(file_path).await?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
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
        assert_eq!(results[1].id, "d");
    }

    #[tokio::test]
    async fn retrieve_returns_invalid_response() {
        let embedder = Box::new(FakeEmbedder);
        let mut retriever = Retriever::new(embedder);

        let result = retriever.retrieve("query", 3).await;

        assert!(matches!(result, Err(RetrieveError::Embed(EmbedError::InvalidResponse))));
    }

    #[tokio::test]
    async fn retrieve_on_empty_index_returns_empty() {
        let embedder = Box::new(FakeEmbedder);
        let retriever = Retriever::new(embedder);

        let results = retriever.retrieve("cat query", 3).await.unwrap();

        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn retrieve_top_k() {
        let embedder = Box::new(FakeEmbedder);
        let mut retriever = Retriever::new(embedder);

        retriever
            .index("cat 1", "cat document")
            .await
            .expect("cat document should be added");

        retriever
            .index("cat 2", "cat document 2")
            .await
            .expect("cat document 2 should be added");

        retriever
            .index("dog 1", "dog document 1")
            .await
            .expect("dog document 1 should be added");

        let results = retriever.retrieve("cat query", 1).await.unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "cat 1");
    }
}