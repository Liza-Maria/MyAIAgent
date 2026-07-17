use reqwest::Client;
use tracing::instrument;
use super::LlmError;
use super::types::{ ChatRequest, ChatResponse, Message };
use super::config::LlmConfig;

#[derive(Debug)]
pub struct LlmClient {
    pub http: Client,
    pub config: LlmConfig,
}

impl LlmClient {
    pub fn new(config: LlmConfig) -> Self {
        Self {
            http: Client::new(),
            config,
        }
    }

    pub async fn chat_with_tool(&self, messages: Vec<Message>, tools: Vec<serde_json::Value>)
        -> Result<ChatResponse, LlmError> {
        
        let request = ChatRequest {
            model: self.config.model.clone(),
            messages,
            tools: Some(tools),
            temperature: None,
        };

        let url = format!("{}/chat/completions", self.config.base_url);

        tracing::debug!("sending chat with tools request");

        let response = self
                        .http
                        .post(&url)
                        .bearer_auth(&self.config.api_key)
                        .json(&request)
                        .send()
                        .await?;

        let status = response.status();

        let text = response.text().await?;

        if !status.is_success() {
            return Err(LlmError::Api {
                status: status.as_u16(),
                body: text,
            });
        }

        let parsed: ChatResponse = serde_json::from_str(&text)
                                    .map_err(|error| LlmError::Decode(error.to_string()))?;
        
        tracing::info!(total_tokens = parsed.usage.as_ref().map(|u| u.total_tokens));

        Ok(parsed)
    }

    #[instrument(skip(self, messages))]
    async fn chat(&self, messages: Vec<Message>) -> Result<ChatResponse, LlmError> {
        let request = ChatRequest {
            model: self.config.model.clone(),
            messages,
            tools: None,
            temperature: None,
        };

        let url = format!("{}/chat/completions", self.config.base_url);

        tracing::debug!("sending chat completions request");

        let response = self
                        .http
                        .post(&url)
                        .bearer_auth(&self.config.api_key)
                        .json(&request)
                        .send()
                        .await?;

        let status = response.status();

        let text = response.text().await?;

        if !status.is_success() {
            return Err(LlmError::Api {
                status: status.as_u16(),
                body: text,
            });
        }

        let parsed: ChatResponse = serde_json::from_str(&text)
                    .map_err(|err| LlmError::Decode(err.to_string()))?;

        tracing::info!(total_tokens = parsed.usage.as_ref().map(|u| u.total_tokens));

        Ok(parsed)
    }

    pub async fn complete(&self, messages: Vec<Message>) -> Result<String, LlmError> {
        let response = self.chat(messages).await?;

        response.choices.into_iter().next()
            .and_then(|choice| choice.message.content)
            .ok_or(LlmError::NoChoices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use wiremock:: {
        matchers:: {method, path},
        MockServer, Mock, ResponseTemplate
    };

    #[tokio::test]
    async fn test_api() {
        let server = MockServer::start().await;

        Mock::given(method("POST")).and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200)
            .set_body_json(json!({
                "choices": [ {
                    "message": {
                        "role": "assistant",
                        "content": "Hello from the mock server",
                    },
                    "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 1,
                    "total_tokens": 15
                }
            })))
            .mount(&server)
            .await;

        let config = LlmConfig {
            api_key: "asdfgh".into(),
            base_url: server.uri(),
            model: "gpt-4o-mini".to_string(),
        };

        let messages = vec![Message::user("Which is the capital of Romania?")];

        let response = LlmClient::new(config).complete(messages)
            .await.expect("Success");

        assert_eq!(response, "Hello from the mock server");
    }
}