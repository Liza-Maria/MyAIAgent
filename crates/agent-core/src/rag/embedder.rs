use crate::rag::{ Embedder, EmbedError };
use serde::{ Serialize, Deserialize };

#[derive(Debug, Serialize)]
pub struct OllamaEmbeddingRequest {
    model: String,
    prompt: String,
}

#[derive(Debug, Deserialize)]
pub struct OllamaEmbeddingResponse {
    embedding: Vec<f32>,
}

pub struct OllamaEmbedder {
    http: reqwest::Client,
    base_url: String,
    model: String,
}

impl OllamaEmbedder {
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: base_url.into(),
            model: model.into(),
        }
    }
}

#[async_trait::async_trait]
impl Embedder for OllamaEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        let ollama_request = OllamaEmbeddingRequest {
            model: self.model.clone(),
            prompt: text.to_string(),
        };

        let url = format!("{}/api/embeddings", &self.base_url.trim_end_matches('/'));
        println!("{}", url);
        let response = self
            .http
            .post(url)
            .json(&ollama_request)
            .send()
            .await
            .map_err(|error| EmbedError::Transport(error.to_string()))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|error| EmbedError::Transport(error.to_string()))?;

        if !status.is_success() {
            return Err(EmbedError::Api {
                status: status.as_u16(),
                body,
            });
        }

        let ollama_response: OllamaEmbeddingResponse = serde_json::from_str(&body)
            .map_err(|error| EmbedError::Decode(error.to_string()))?;

        Ok(ollama_response.embedding)
    }
}

#[cfg(test)]

mod tests {
    use super::*;

    use wiremock::{
        matchers::{method, path},
        Mock,
        MockServer,
        ResponseTemplate,
    };

    #[tokio::test]
    async fn test_ollama_embedder() {
        let server = MockServer::start().await;

        let response = serde_json::json!({"embedding": [0.012345, -0.056789, 0.034567, 0.078912, -0.045678, 0.023456, -0.067890, 0.056789, 0.012345, -0.034567]});

        Mock::given(method("POST"))
            .and(path("/api/embeddings"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(response))
            .mount(&server)
            .await;
        
        let ollama_embedder = OllamaEmbedder::new(
            server.uri(),
            "test embedding");


        let embedding_result = ollama_embedder.embed("Artificial intelligence is transforming the world.")
            .await
            .expect("deserialized");

        assert_eq!(embedding_result.len(), 10);
        assert_eq!(embedding_result[0], 0.012345);
    }
}