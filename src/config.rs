use serde::{Deserialize, Serialize};
use std::env;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Config {
    pub ollama_host: String,
    pub ollama_port: u16,
    pub embed_model: String,
    pub llm_model: String,
    pub docs_dir: String,
    pub server_host: String,
    pub server_port: u16,
    pub max_context_docs: usize,
    pub ef_search: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            ollama_host: "http://localhost".to_string(),
            ollama_port: 11434,
            embed_model: "mxbai-embed-large:latest".to_string(),
            llm_model: "llama3.2".to_string(),
            docs_dir: "documents".to_string(),
            server_host: "0.0.0.0".to_string(),
            server_port: 3000,
            max_context_docs: 2,
            ef_search: 64,
        }
    }
}

impl Config {
    pub fn from_env() -> Self {
        Self {
            ollama_host: env::var("OLLAMA_HOST")
                .unwrap_or_else(|_| "http://localhost".to_string()),
            ollama_port: env::var("OLLAMA_PORT")
                .unwrap_or_else(|_| "11434".to_string())
                .parse()
                .unwrap_or(11434),
            embed_model: env::var("EMBED_MODEL")
                .unwrap_or_else(|_| "mxbai-embed-large:latest".to_string()),
            llm_model: env::var("LLM_MODEL")
                .unwrap_or_else(|_| "llama3.2".to_string()),
            docs_dir: env::var("DOCS_DIR")
                .unwrap_or_else(|_| "documents".to_string()),
            server_host: env::var("SERVER_HOST")
                .unwrap_or_else(|_| "0.0.0.0".to_string()),
            server_port: env::var("SERVER_PORT")
                .unwrap_or_else(|_| "3000".to_string())
                .parse()
                .unwrap_or(3000),
            max_context_docs: env::var("MAX_CONTEXT_DOCS")
                .unwrap_or_else(|_| "2".to_string())
                .parse()
                .unwrap_or(2),
            ef_search: env::var("EF_SEARCH")
                .unwrap_or_else(|_| "64".to_string())
                .parse()
                .unwrap_or(64),
        }
    }

    pub fn from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let config: Config = toml::from_str(&content)?;
        Ok(config)
    }
}
