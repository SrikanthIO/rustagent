use std::fs;
use ollama_rs::Ollama;
use qdrant_client::prelude::*;
use std::collections::HashMap;
use qdrant_client::qdrant::{PointStruct, Value, Vectors, SearchPoints};
use qdrant_client::client::QdrantClient;

pub mod api;
pub mod config;
pub mod agent;
pub mod server;
pub mod pdf_reader;

pub use api::*;
pub use config::*;
pub use agent::*;
pub use server::*;
pub use pdf_reader::*;

pub struct RAG {
    pub ollama: Ollama,
    embed_model: String,
    qdrant: QdrantClient,
    collection_name: String,
}

impl RAG {
    pub async fn new(ollama: Ollama, embed_model: String, docs_dir: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let qdrant = QdrantClient::from_url("http://localhost:6334").build()?;
        let collection_name = "documents".to_string();
        let mut points = Vec::new();
        let mut dim = None;

        for entry in std::fs::read_dir(docs_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() {
                println!("Processing document: {}", path.display());
                let text = crate::pdf_reader::read_document_content(&path).unwrap_or_else(|_| String::new());
                if text.trim().is_empty() { continue; }
                let embed_response = ollama
                    .generate_embeddings(embed_model.clone(), text.clone(), None)
                    .await;
                let embed_response = match embed_response {
                    Ok(r) => r,
                    Err(e) => { eprintln!("Embedding error: {e}"); continue; }
                };
                let embedding_f32 = Self::f64_to_f32_vec(&embed_response.embeddings);
                if dim.is_none() { dim = Some(embedding_f32.len()); }
                let title = path.file_name().unwrap().to_string_lossy().to_string();
                let mut payload = HashMap::new();
                payload.insert("title".to_string(), Value::from(title.clone()));
                payload.insert("body".to_string(), Value::from(text.clone()));
                points.push(PointStruct {
                    id: Some((points.len() as u64).into()),
                    vectors: Some(Vectors::from(embedding_f32)),
                    payload,
                });
            }
        }
        if let Some(real_dim) = dim {
            let _ = qdrant.create_collection(&qdrant_client::qdrant::CreateCollection {
                collection_name: collection_name.clone(),
                vectors_config: Some(qdrant_client::qdrant::VectorsConfig {
                    config: Some(qdrant_client::qdrant::vectors_config::Config::Params(
                        qdrant_client::qdrant::VectorParams {
                            size: real_dim as u64,
                            distance: qdrant_client::qdrant::Distance::Cosine.into(),
                            ..Default::default()
                        },
                    )),
                }),
                ..Default::default()
            }).await; // ignore already exists
            if !points.is_empty() {
                qdrant.upsert_points(collection_name.clone(), None, points, None).await?;
            }
        }
        Ok(RAG {
            ollama,
            embed_model,
            qdrant,
            collection_name
        })
    }

    pub async fn retrieve_context(&self, query: &str, k: usize) -> Result<String, Box<dyn std::error::Error>> {
        let query_embed_f64 = self.ollama
            .generate_embeddings(self.embed_model.clone(), query.to_string(), None)
            .await?;
        let query_vec = Self::f64_to_f32_vec(&query_embed_f64.embeddings);
        let search_points = SearchPoints {
            collection_name: self.collection_name.clone(),
            vector: query_vec,
            limit: k as u64,
            with_payload: Some(true.into()),
            ..Default::default()
        };
        let search_result = self.qdrant.search_points(&search_points).await?;
        let mut contexts = Vec::new();
        for pt in search_result.result {
            let payload = &pt.payload;
            let title = payload.get("title").and_then(|v| v.as_str()).map_or("?", |v| v);
            let body = payload.get("body").and_then(|v| v.as_str()).map_or("", |v| v);
            contexts.push(format!("[{}]\n{}", title, body));
        }
        Ok(contexts.join("\n\n"))
    }
    
    fn f64_to_f32_vec(v: &Vec<f64>) -> Vec<f32> {
        v.iter().map(|x| *x as f32).collect()
    }
}
