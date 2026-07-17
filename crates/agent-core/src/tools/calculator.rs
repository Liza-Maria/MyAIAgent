use meval;
use crate::tools::error::ToolError;
use crate::tools::tool::Tool;  

#[derive(Debug)]
pub struct CalculatorTool;

impl Tool for CalculatorTool {
    fn name(&self) -> &str {
        "Calculator"
    }

    fn definition(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "Calculator",
                "description": "Evaluate a math expression. Use this for any arithmetic.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "The math expression to evaluate, e.g. '2 + 2'"
                        }
                    },
                    "required": ["expression"]
                }
            }
        })
    }

    fn execute(&self, args: serde_json::Value) -> Result<String, ToolError> {
        let expression = args["expression"]
                            .as_str()
                            .ok_or_else(|| ToolError::InvalidArguments{
                                tool: "calulator".to_string(),
                                args: "missing string field `expression`".to_string(),
                            })?;

        let result = meval::eval_str(expression)
                            .map_err(|err| {ToolError::Execution(err.to_string())})?;

        Ok(result.to_string())
    }
}