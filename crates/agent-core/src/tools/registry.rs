use std::collections::HashMap;
use crate::tools::tool::Tool;
use crate::tools::error::ToolError;
use serde_json::Value;

#[derive(Debug)]
pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    pub fn register<T: Tool + 'static>(&mut self, tool: T) {
        let name = tool.name();
        self.tools.insert(name.to_string(), Box::new(tool));
    }

    pub fn definitions(&self) -> Vec<Value> {
        self.tools
            .values()
            .map(|tool| tool.definition())
            .collect()
    }

    pub fn execute(&self, toolName: &str, args: Value) -> Result<String, ToolError> {
        match self.tools.get(toolName) {
            Some(tool) => tool.execute(args),
            None => Err(ToolError::NotFound(toolName.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::calculator::CalculatorTool;
    use serde_json::json;

    #[test]
    fn register_and_execute() {
        let mut registry = ToolRegistry::new();
        registry.register(CalculatorTool);

        let definitions = registry.definitions();
        assert_eq!(definitions.len(), 1);

        let result = registry.execute("Calculator", json!({"expression": "2 + 2"}))
                            .expect("should execute");

        assert_eq!(result, "4".to_string());
    }

    #[test]
    fn execute_nonexistent_tool() {
        let registry = ToolRegistry::new();

        let result = registry.execute("nonexistent", json!({"expression": "2 + 2"}))
                    .expect_err("should return error");

        match result {
            ToolError::NotFound(name) => {
                assert_eq!(name, "nonexistent");
            },
            error => {
                panic!("expected NotFound but found {error}");
            },
        }
    }
}


