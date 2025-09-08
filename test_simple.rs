use ollama_rs::Ollama;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing Ollama connection...");
    
    let ollama = Ollama::new("http://localhost".to_string(), 11434);
    
    // Test with a simple text
    let test_text = "Hello, this is a test.";
    println!("Testing with text: {}", test_text);
    
    match ollama.generate_embeddings("mxbai-embed-large:latest".to_string(), test_text.to_string(), None).await {
        Ok(response) => {
            println!("✅ Success! Got {} embeddings", response.embeddings.len());
            println!("First embedding length: {}", response.embeddings[0].len());
        },
        Err(e) => {
            println!("❌ Error: {}", e);
        }
    }
    
    Ok(())
}
