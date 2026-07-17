use crate::tools::error::ToolError;
use std::fmt::Debug;

pub trait Tool: Debug {
    fn name(&self) -> &str;
    fn definition(&self) -> serde_json::Value;
    fn execute(&self, args: serde_json::Value) -> Result<String, ToolError>;
}