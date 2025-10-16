use std::env;
use std::fs;
use serde_json::{self, Value};
use agentic_rag_framework::{agent::{PipelineConfig, AgentOrchestrator, OrchestratorContext, AgentInput, AgentOutput, agent_registry_with_defaults}, RAG};

fn parse_agent_input(json_val: &Value) -> Result<AgentInput, String> {
    let t = json_val.get("type").and_then(|x| x.as_str()).ok_or_else(|| "Missing 'type' in input JSON".to_string())?;
    match t {
        "Email" => {
            let email = json_val.get("email").and_then(|x| x.as_str()).or_else(|| json_val.get("value").and_then(|x| x.as_str()))
                .ok_or_else(|| "Missing 'email' (or 'value') for Email input".to_string())?;
            Ok(AgentInput::Email(email.to_string()))
        }
        "EmailWithContext" => {
            let email = json_val.get("email").and_then(|x| x.as_str()).ok_or_else(|| "Missing 'email' for EmailWithContext input".to_string())?;
            let context = json_val.get("context").and_then(|x| x.as_str()).ok_or_else(|| "Missing 'context' for EmailWithContext input".to_string())?;
            Ok(AgentInput::EmailWithContext { email: email.to_string(), context: context.to_string() })
        }
        other => Err(format!("Unsupported input type '{}'. Supported: 'Email', 'EmailWithContext'", other)),
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // CLI: <bin> <workflow_config.json>
    let args: Vec<String> = env::args().collect();
    let config_path = if args.len() > 1 { &args[1] } else { "agent_flow.json" };
    println!("[INFO] Using workflow pipeline: {}", config_path);

    let config_json = fs::read_to_string(config_path)?;
    let config: PipelineConfig = serde_json::from_str(&config_json)?;

    // derive initial input from the first agent node that contains input
    let first_with_input = config.agents.iter().find_map(|n| n.input.as_ref());
    let input_val = first_with_input.ok_or("Workflow config is missing an initial 'input' on any agent node")?;
    let input = match parse_agent_input(input_val) {
        Ok(i) => i,
        Err(e) => { eprintln!("[ERROR] {}", e); std::process::exit(1); }
    };

    // Set up context (dummy RAG and LLM config for now)
    let ollama = ollama_rs::Ollama::new("http://localhost".to_string(), 11434);
    let rag = std::sync::Arc::new(RAG::new(ollama, "mxbai-embed-large:latest".to_string(), "documents").await?);
    let ctx = OrchestratorContext {
        rag: Some(rag.clone()),
        llm_config: Some("llama3.2".to_string()),
    };

    let registry = agent_registry_with_defaults();
    let orchestrator = AgentOrchestrator::from_json_registry(config, ctx, &registry);

    let result = orchestrator.handle_job(input).await?;
    match result {
        AgentOutput::ReplyDraft(reply) => println!("Suggested Reply: {}", reply),
        AgentOutput::Other(v) => println!("Pipeline final result: {}", v),
        other => println!("Pipeline result: {:?}", other),
    }
    Ok(())
}