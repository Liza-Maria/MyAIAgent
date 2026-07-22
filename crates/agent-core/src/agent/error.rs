use super::LlmError;
use super::ToolError;
use crate::rag::RetrieveError;

#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("Llm call failed.")]
    Llm(#[from] LlmError),

    #[error("Model has reached maximum iteration without the final answear.")]
    MaxIterations(usize),

    #[error("Model has returned no choice")]
    NoChoice,

    #[error("Response is empty")]
    EmptyResponse,

    #[error("Invalid argument")]
    InvalidArgument,

    #[error("Tool failed")]
    Tool(#[from] ToolError),

    #[error("Retrieve call failed")]
    Retriever(#[from] RetrieveError),
}