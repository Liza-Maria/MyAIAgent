#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub max_iterations: usize,
    pub system_prompt: String,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_iterations: 5,
            system_prompt: "Default prompt".into(),
        }
    }
}