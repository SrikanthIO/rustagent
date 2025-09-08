use axum::Router;
use std::sync::Arc;
use tokio::net::TcpListener;
use tracing::{info, error};

use crate::{RAG, Config, create_router};

pub async fn start_server(rag: Arc<RAG<'static>>, config: &Config) -> Result<(), Box<dyn std::error::Error>> {
    let app = create_router(rag);
    
    let addr = format!("{}:{}", config.server_host, config.server_port);
    let listener = TcpListener::bind(&addr).await?;
    
    info!("🚀 Server starting on {}", addr);
    info!("📚 Health check: http://{}/health", addr);
    info!("💬 Chat API: http://{}/chat", addr);
    
    axum::serve(listener, app).await?;
    
    Ok(())
}

pub async fn run_server_mode(config: Config) -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();
    
    // Initialize Ollama client
    let ollama = ollama_rs::Ollama::new(config.ollama_host.clone(), config.ollama_port);
    
    // Initialize RAG system
    info!("🔄 Initializing RAG system...");
    let rag = RAG::new(ollama, config.embed_model.clone(), &config.docs_dir).await?;
    let rag = Arc::new(rag);
    
    info!("✅ RAG system initialized with {} documents", rag.doc_texts.len());
    
    // Start server
    start_server(rag, &config).await?;
    
    Ok(())
}
