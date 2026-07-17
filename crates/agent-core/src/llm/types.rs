use serde::{Deserialize, Serialize};

/// Who authored a message. Serializes as "system" / "user" / "assistant".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// One message in the conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
                role: Role::System,
                content: Some(content.into()),
                tool_calls: None,
                tool_call_id: None,
            }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self {
                role: Role::User,
                content: Some(content.into()),
                tool_calls: None,
                tool_call_id: None,
            }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
                role: Role::Assistant,
                content: Some(content.into()),
                tool_calls: None,
                tool_call_id: None,
            }
    }

    pub fn tool(content: impl Into<String>, tool_call_id: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
        }
    }
}

/// The request body we POST. Send-only, so `Serialize` only.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<serde_json::Value>>,
    pub temperature: Option<f32>,
}

/// The response body we receive. Receive-only, so `Deserialize` only.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatResponse {
    pub choices: Vec<Choice>,
    #[serde(default)]
    pub usage: Option<Usage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<serde_json::Value>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Choice {
    pub message: Message,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename="type", default)]
    pub call_type: String,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_serialize() {
        let chat_request = ChatRequest {
            model: "Gpt 4.0 mini".to_string(),
            temperature: None,
            messages: vec![Message::system("Which is the capital of Romania?".to_string())],
            tools: None,
        };

        let json = serde_json::to_value(&chat_request).expect("serialize to Json");

        /*"{
            "model": "Gpt 4.0 mini",
            "messages": [{ "role": "system",
                            "content": "Which is the capital of Romania?"
                        }]
        }"*/

        assert_eq!(json["model"], "Gpt 4.0 mini");
        assert_eq!(json["messages"][0]["role"], "system");
        assert_eq!(json["messages"][0]["content"], "Which is the capital of Romania?");
    }

    #[test]
fn response_deserialize() {
    let json = r#"
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Bucharest."
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 18,
                "completion_tokens": 2,
                "total_tokens": 20
            }
        }
    "#;

        let response: ChatResponse = serde_json::from_str(json)
                        .expect("deserialize ChatResponse from JSON");

        assert_eq!(response.choices.len(), 1);
        assert_eq!(response.choices[0].message.role, Role::Assistant);
        assert_eq!(response.choices[0].message.content, Some("Bucharest.".to_string()));
        assert_eq!(response.choices[0].finish_reason, Some("stop".to_string()));

        let usage = response.usage.expect("usage should exist");
        assert_eq!(usage.prompt_tokens, 18);
        assert_eq!(usage.completion_tokens, 2);
        assert_eq!(usage.total_tokens, 20);
    }
}