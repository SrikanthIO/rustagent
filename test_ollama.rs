use ollama_rs::Ollama;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ollama = Ollama::new("http://localhost".to_string(), 11434);
    
    println!("Testing Ollama connection...");
    
    // Test embeddings
    match ollama.generate_embeddings("mxbai-embed-large:latest".to_string(), "test".to_string(), None).await {
        Ok(response) => {
            println!("✅ Embeddings test successful");
            println!("Response: {:?}", response);
        },
        Err(e) => {
            println!("❌ Embeddings test failed: {}", e);
        }
    }
    
    Ok(())
}
