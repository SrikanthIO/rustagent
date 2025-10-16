use axum::{
    extract::State,
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

use crate::RAG;

#[derive(Serialize, Deserialize)]
pub struct ChatRequest {
    pub message: String,
    pub history: Option<Vec<(String, String)>>,
    pub k: Option<usize>,
}

#[derive(Serialize, Deserialize)]
pub struct ChatResponse {
    pub response: String,
    pub context: String,
}

#[derive(Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub message: String,
}

pub async fn health_check() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy".to_string(),
        message: "Agentic RAG Framework is running".to_string(),
    })
}

pub async fn chat_handler(
    State(rag): State<Arc<RAG>>,
    Json(payload): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, StatusCode> {
    let k = payload.k.unwrap_or(2);
    
    // Retrieve context using RAG
    let context = rag.retrieve_context(&payload.message, k).await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    // Build prompt with history + context
    let mut prompt = String::new();
    prompt.push_str("You are a helpful assistant. Use the provided context when helpful.\n\n");
    prompt.push_str("Context documents:\n");
    prompt.push_str(&context);
    
    if let Some(history) = &payload.history {
        prompt.push_str("\n\nConversation so far:\n");
        for (role, msg) in history {
            prompt.push_str(&format!("{}: {}\n", role, msg));
        }
    }
    
    prompt.push_str(&format!("User: {}\nAssistant:", payload.message));

    // Generate with Llama 3.2
    let request = ollama_rs::generation::completion::request::GenerationRequest::new(
        "llama3.2".to_string(), 
        prompt
    );
    let response = rag.ollama.generate(request).await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    let assistant_reply = response.response;

    Ok(Json(ChatResponse {
        response: assistant_reply,
        context,
    }))
}

pub fn create_router(rag: Arc<RAG>) -> Router {
    Router::new()
        .route("/health", get(health_check))
        .route("/chat", post(chat_handler))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(rag)
}
