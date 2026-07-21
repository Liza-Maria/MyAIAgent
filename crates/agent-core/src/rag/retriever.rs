use super::{
    RetrieveError,
    SearchResult,
    Document,
    VectorStore,
    Embedder};

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