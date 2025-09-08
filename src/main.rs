use std::io::{self, Write};
use ollama_rs::Ollama;
use ollama_rs::generation::completion::request::GenerationRequest;
use agentic_rag_framework::{RAG, Config, run_server_mode};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Check for server mode
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 && args[1] == "server" {
        let config = Config::from_env();
        return run_server_mode(config).await;
    }

    // Initialize Ollama client (assumes Ollama is running locally)
    let ollama = Ollama::new("http://localhost".to_string(), 11434);
    let embed_model = "mxbai-embed-large:latest";
    let llama_model = "llama3.2";

    // --- Initialize RAG system ---
    let docs_dir = "documents";
    let rag = RAG::new(ollama, embed_model.to_string(), docs_dir).await?;

    // --- Simple chat REPL ---
    println!("Interactive chat. Type ':exit' to quit.\n");
    let mut history: Vec<(String, String)> = Vec::new(); // (role, content) pairs

    loop {
        print!("You: ");
        io::stdout().flush().ok();
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();
        if input.is_empty() { continue; }
        if input == ":exit" { break; }

        // --- Retrieve context using RAG ---
        let k = 2;
        let context_block = rag.retrieve_context(input, k).await?;

        // --- Build prompt with history + context ---
        let mut prompt = String::new();
        prompt.push_str("You are a helpful assistant. Use the provided context when helpful.\n\n");
        prompt.push_str("Context documents:\n");
        prompt.push_str(&context_block);
        prompt.push_str("\n\nConversation so far:\n");
        for (role, msg) in &history {
            prompt.push_str(&format!("{}: {}\n", role, msg));
        }
        prompt.push_str(&format!("User: {}\nAssistant:", input));

        // --- Generate with Llama 3.2 ---
        let request = GenerationRequest::new(llama_model.to_string(), prompt);
        let response = rag.ollama.generate(request).await?;
        let assistant_reply = response.response.clone();
        println!("Assistant: {}\n", assistant_reply);

        // Update history
        history.push(("User".to_string(), input.to_string()));
        history.push(("Assistant".to_string(), assistant_reply));
    }

    Ok(())
}
