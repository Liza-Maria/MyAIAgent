pub mod telemetry;
pub mod llm;
pub mod agent;
pub mod tools;

pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}