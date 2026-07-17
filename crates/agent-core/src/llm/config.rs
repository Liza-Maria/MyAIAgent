use super::error;
use std::env;

#[derive(Debug, Clone)]
pub struct LlmConfig {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
}

impl LlmConfig {
    pub fn from_environment() -> Result<Self, error::LlmError> {
        let api_key = env::var("OPENAI_API_KEY")
                .map_err(|_| error::LlmError::MissingEnv("OPENAI_API_KEY".to_string()))?;

        let base_url = std::env::var("OPENAI_BASE_URL")
            .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());

        let model = std::env::var("OPENAI_MODEL")
            .unwrap_or_else(|_| "gpt-4o-mini".to_string());

        Ok(Self {
            api_key,
            base_url,
            model,
        })
    }
}