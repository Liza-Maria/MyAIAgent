
#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    #[error("Missing required environment variable: {0}")]
    MissingEnv(String),
    #[error("Http Transport error: {0}")]
    Transport(#[from] reqwest::Error),
    #[error("API return status: {status}: {body}")]
    Api {
        status: u16, 
        body: String,
    },
    #[error("Could not decode request body: {0}")]
    Decode(String),
    #[error("Response contains no choices")]
    NoChoices,
}
