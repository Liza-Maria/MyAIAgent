use std::fs::Metadata;

use super::{
    RetrieveError,
    SearchResult,
    Document,
    VectorStore,
    Embedder,
    EmbedError,
    Chunker};

use std::path::{ Path, PathBuf };

    #[derive(Debug, Default, Clone, PartialEq)]
    pub struct IngestReport {
        pub files_indexed: usize,
        pub chunks_indexed: usize,
        pub files_skipped: usize,
    }

    #[derive(Debug)]
pub struct Retriever {
    embedder: Box<dyn Embedder>,
    chunker: Box<dyn Chunker>,
    store: VectorStore,
}

fn print_type<T>(_: &T) {
      println!("{}", std::any::type_name::<T>());
  }

impl Retriever {
    pub fn new(embedder: Box<dyn Embedder>, chunker: Box<dyn Chunker>) -> Self {
        Self {
            embedder,
            chunker,
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

    pub fn save(&self, path: &Path) -> Result<(), RetrieveError> {
        self.store.save(path, self.embedder.model())?;

        Ok(())
    }

    pub fn load(&mut self, path: &Path) -> Result<usize, RetrieveError> {
        let res = VectorStore::load(path, self.embedder.model())?;
        let length = res.documents.len();
        self.store = res;

        Ok(length)
    }

    pub async fn index_document(&mut self, doc_id: &str, text: &str) -> Result<usize, RetrieveError> {
        let chunks = self.chunker.chunk(text);
        
        let mut docs: Vec<Document> = Vec::new();

        for (id, chunk) in chunks.iter().enumerate() {
            let chunk_id = format!("{doc_id}#chunk_{id}");

            self.index(&chunk_id, text).await?;
        }

        Ok(chunks.len())
    }

    pub async fn index_file(&mut self, path: &Path) -> Result<usize, RetrieveError> {
        let text = std::fs::read_to_string(path)?;

        let id = path.to_string_lossy().to_string();

        Ok(self.index_document(&id, &text).await?)
    }

    pub async fn index_directory(&mut self, root: &Path, extensions: &[&str]) 
        -> Result<IngestReport, RetrieveError> 
        {
        let mut report = IngestReport::default();

        let mut pending: Vec<PathBuf> = vec![root.to_path_buf()];
        
        while let Some(element) = pending.pop() {
            let metadata = std::fs::metadata(&element)
                            .map_err(|err| RetrieveError::Io(err))?;
            if metadata.is_dir() {
                let mut children: Vec<PathBuf> = Vec::new();

                let entries = std::fs::read_dir(&element)
                                .map_err(|err| RetrieveError::Io(err))?;
    
                for entry in entries {
                    let entry = entry.map_err(|err| RetrieveError::Io(err))?;
                    children.push(entry.path());
                }

                children.sort();
                children.reverse();
                pending.extend(children);
            } else if metadata.is_file() {
                let matches = element.extension()
                    .and_then(|ext| ext.to_str())
                    .map(|exten| { extensions
                                .iter()
                                .any(|allowed_ext| allowed_ext
                                                    .eq_ignore_ascii_case(exten))})
                                                    .unwrap_or(false);

                if !matches {
                    report.files_skipped += 1;
                    continue;
                } 

                let indexed_chunks = self.index_file(&element).await?;

                report.chunks_indexed += indexed_chunks;
                report.files_indexed += 1;
            }
        }

        Ok(report)
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