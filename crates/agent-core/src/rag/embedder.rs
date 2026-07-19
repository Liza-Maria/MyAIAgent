pub struct OllamaEmbedder {
    http: request::Client,
    base_url: String,
    model: String,
}

impl OllamaEmbedder {
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: request::Client::new(),
            base_url: base_url.into(),
            model: model.into(),
        }
    }
}

impl Embedder for OllamaEmbedder {

}