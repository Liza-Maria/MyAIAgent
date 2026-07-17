use tracing::instrument;
use super::{ AgentConfig, AgentError };
use super::{ LlmClient, LlmConfig, LlmError, Message };
use crate::tools::registry::ToolRegistry;

#[derive(Debug)]
pub struct Agent {
    pub llm_client: LlmClient,
    pub config: AgentConfig,
    pub tool: ToolRegistry,
}

impl Agent {
    pub fn new(llm_config: LlmConfig, config: AgentConfig, tool: ToolRegistry) -> Self {
        Self {
            llm_client: LlmClient::new(llm_config),
            config,
            tool,
        }
    }

    #[instrument(skip(goal), fields(goal = %goal))]
    pub async fn run(&self, goal: &str) -> Result<String, AgentError> {
        let current_message: String = goal.into();
        let mut history = vec![Message::system(self.config.system_prompt.clone()),
                            Message::user(current_message)];
        
        for _iteration in 0..self.config.max_iterations {
            let response = self.llm_client.chat_with_tool(
                                                history.clone(),
                                                self.tool.definitions()).await?;
            
            // complete(history.clone()).await?;

            let choice = response.choices
                                .into_iter()
                                .next()
                                .ok_or(AgentError::EmptyResponse)?;

            let message = choice.message;

            if let Some(tool_calls) = message.tool_calls.clone() {
                history.push(message);
                
                for tool_call in tool_calls {
                    let arguments = serde_json::from_str(&tool_call.function.arguments)
                            .map_err(|error| { AgentError::InvalidArgument })?;

                    let result = self.tool.execute(&tool_call.function.name, arguments)?;

                    history.push(Message::tool(result, tool_call.id));
                }

                continue;
            }

            return message.content.ok_or(AgentError::EmptyResponse);
            
        }
        
        Err(AgentError::MaxIterations(self.config.max_iterations))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use wiremock::{
        matchers::{method, path},
        Mock,
        MockServer,
        ResponseTemplate,
    };
    use crate::tools::calculator::CalculatorTool;

    #[tokio::test]
    async fn agent_returns_final_answer_without_tool_call() {
        let server = MockServer::start().await;

        let response = json!({
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "hello"
                    },
                    "finish_reason": "stop"
                }
            ]
        });

        Mock::given(method("POST"))
            .and(path("chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(response))
            .mount(&server)
            .await;

        let llmConfig = LlmConfig {
            base_url: server.uri(),
            model: "test-model".to_string(),
            api_key: "test_key".to_string(),
        };

        let agent_config = AgentConfig {
            system_prompt: "Test agent".to_string(),
            max_iterations: 5,
        };

        let mut tool_registry = ToolRegistry::new();
        tool_registry.register(CalculatorTool);

        let agent = Agent::new(llmConfig, agent_config, tool_registry);

        let answer = agent
                        .run("say hello")
                        .await
                        .expect("agent should return final answer");

        assert_eq!(answer, "hello");
    }

    #[tokio::test]
    async fn run_executes_calculator_tool() {
        let server = MockServer::start().await;

        let tool_call_response = json!({
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_1",
            "call_type": "function",
            "function": { "name": "Calculator", "arguments": "{\"expression\":\"240 * 0.15\"}" }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ]
});

    let final_response = json!({
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "The answer is 36.",
                },
                "finish_reason": "stop",
            }
        ]
    });

        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(tool_call_response))
            .up_to_n_times(1)
            .mount(&server)
            .await;

        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(final_response))
            .mount(&server)
            .await;

        let llmConfig = LlmConfig {
            base_url: server.uri(),
            model: "test-model".to_string(),
            api_key: "test_key".to_string(),
        };

        let agent_config = AgentConfig {
            system_prompt: "Test agent".to_string(),
            max_iterations: 5,
        };

        let mut tool_registry = ToolRegistry::new();
        tool_registry.register(CalculatorTool);

        let agent = Agent::new(llmConfig, agent_config, tool_registry);

        let answer = agent.run("What is 240 * 0.15?")
                        .await
                        .expect("agent should return a result");

        assert_eq!(answer, "The answer is 36.");
    }
}

