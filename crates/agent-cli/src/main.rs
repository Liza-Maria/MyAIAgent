use agent_core::telemetry;
use anyhow::Result;
use clap:: { Parser, Subcommand };
use dotenvy::dotenv;

use agent_core::llm::{ LlmClient, LlmConfig , Role, Message };
use agent_core::agent::{ Agent, AgentConfig };
use agent_core::tools::{ ToolRegistry, CalculatorTool };
use agent_core::memory::SlidingWindow;
use agent_core::rag::{ OllamaEmbedder, Retriever };

#[derive(Debug, Parser)]
#[command(name = "agent", version, about = "Rust AI Agent Runtime")]
struct Cli {

    #[arg(long, global = true, default_value = "warn")]
    log: String,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Doctor,
    Chat {
        prompt: String,
    },
    Run {
        goal: String,
    }
}

#[tracing::instrument]
async fn doctor() -> Result<()> {
    println!("Agent is running healthy");
    
    Ok(())
}

async fn chat(prompt: String) -> Result<()> {
    let config = LlmConfig::from_environment()?;

    let llm_client = LlmClient::new(config);

    let messages = vec![Message {
        role: Role::User,
        content: Some(prompt),
        tool_calls: None,
        tool_call_id: None,
    }];

    let _response = llm_client.complete(messages).await?;
    println!("{:?}", _response);

    Ok(())
}

async fn run(goal: String) -> Result<()> {
    let llm_config = LlmConfig::from_environment()?;
    let agent_config = AgentConfig::default();

    let mut tools = ToolRegistry::new();
    tools.register(CalculatorTool);

    let memory = SlidingWindow {
        max_messages: 20,
    };

    let url = std::env::var("OPENAI_BASE_URL")?;
    let model = std::env::var("OPENAI_EMBEDDER_MODEL")?;
    //let embedder = OllamaEmbedder::new(url, model);
    let embedder = OllamaEmbedder::new("http://localhost:11434", "nomic-embed-text");
    let mut retriever = Retriever::new(Box::new(embedder));

    retriever.index("fact-1", "The project's mascot is a rusty crab named Ferris.").await?;
    retriever.index("fact-2", "The agent runtime was built in eight stages.").await?;

    let agent = Agent::new(llm_config, agent_config, tools, Box::new(memory))
                        .with_retriever(retriever);

    let result = agent.run(&goal).await?;
    println!("{}", result);

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let _ = dotenv();

    let cli = Cli::parse();

    telemetry::init(&cli.log)?;

    match cli.command {
        Command::Doctor => {
            doctor().await
        },
        Command::Chat { prompt } => {
            chat(prompt).await
        },
        Command::Run { goal } => {
            run(goal).await
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn cli_definition() {
        Cli::command().debug_assert();
    }

    #[test]
    fn parse_doctor() {
        let cli = Cli::try_parse_from(["myaiagent", "--log", "debug", "doctor"]).expect("Should parse");
        assert_eq!(cli.log, "debug");
        assert!(matches!(cli.command, Command::Doctor));
    }

    #[test]
    fn test_chat() {
        let cli = Cli::try_parse_from([
            "myaiagent",
            "--log",
            "debug",
            "chat", 
            "Which is the capital of Romania"])
            .expect("Should parse");

        assert_eq!(cli.log, "debug");
        assert!(matches!(cli.command, Command::Chat { .. }));
    }
}