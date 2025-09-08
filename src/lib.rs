use std::fs;
use ollama_rs::Ollama;
use hnsw_rs::prelude::*;

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

pub struct RAG<'a> {
    pub ollama: Ollama,
    embed_model: String,
    doc_texts: Vec<String>,
    doc_titles: Vec<String>,
    hnsw: Hnsw<'a, f32, DistCosine>,
}

impl<'a> RAG<'a> {
    pub async fn new(ollama: Ollama, embed_model: String, docs_dir: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let mut doc_texts: Vec<String> = Vec::new();
        let mut doc_titles: Vec<String> = Vec::new();
        let mut doc_embeddings: Vec<Vec<f32>> = Vec::new();

        // Load and embed documents
        for entry in fs::read_dir(docs_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() {
                println!("Processing document: {}", path.display());
                
                // Read document content (PDF or text)
                let text = match read_document_content(&path) {
                    Ok(content) => content,
                    Err(e) => {
                        eprintln!("Error reading document {}: {}", path.display(), e);
                        continue;
                    }
                };
                
                // Skip empty documents
                if text.trim().is_empty() {
                    eprintln!("Skipping empty document: {}", path.display());
                    continue;
                }
                
                // Add retry logic for Ollama calls
                let embed_response = match ollama
                    .generate_embeddings(embed_model.clone(), text.clone(), None)
                    .await
                {
                    Ok(response) => response,
                    Err(e) => {
                        eprintln!("Error generating embeddings for {}: {}", path.display(), e);
                        continue;
                    }
                };
                
                let embedding_f32 = Self::f64_to_f32_vec(&embed_response.embeddings);
                doc_titles.push(path.file_name().unwrap().to_string_lossy().to_string());
                doc_texts.push(text);
                doc_embeddings.push(embedding_f32);
            }
        }

        // Check if we have any documents
        if doc_embeddings.is_empty() {
            return Err("No documents found or all documents failed to process".into());
        }

        // Build HNSW index
        let dim = doc_embeddings[0].len();
        let max_nb_connection = 16;
        let ef_construction = 200;
        let mut hnsw: Hnsw<f32, DistCosine> = Hnsw::new(max_nb_connection, doc_embeddings.len(), dim, ef_construction, DistCosine);

        for (i, emb) in doc_embeddings.iter().enumerate() {
            hnsw.insert((&emb[..], i));
        }

        Ok(RAG {
            ollama,
            embed_model,
            doc_texts,
            doc_titles,
            hnsw,
        })
    }

    pub async fn retrieve_context(&self, query: &str, k: usize) -> Result<String, Box<dyn std::error::Error>> {
        // Embed the query with error handling
        let query_embed_f64 = match self.ollama
            .generate_embeddings(self.embed_model.clone(), query.to_string(), None)
            .await
        {
            Ok(response) => response,
            Err(e) => {
                eprintln!("Error generating query embeddings: {}", e);
                return Err(format!("Failed to generate embeddings: {}", e).into());
            }
        };
        let query_embed: Vec<f32> = Self::f64_to_f32_vec(&query_embed_f64.embeddings);

        // Retrieve top-k documents with HNSW
        let k = k.min(self.doc_texts.len());
        let ef_search = 64;
        let neighborhood: Vec<Neighbour> = self.hnsw.search(&query_embed[..], k, ef_search);

        let mut contexts: Vec<String> = Vec::new();
        for neighbor in neighborhood {
            let idx = neighbor.d_id;
            let title = &self.doc_titles[idx];
            let body = &self.doc_texts[idx];
            contexts.push(format!("[{}]\n{}", title, body));
        }

        Ok(contexts.join("\n\n"))
    }

    fn f64_to_f32_vec(v: &Vec<f64>) -> Vec<f32> {
        v.iter().map(|x| *x as f32).collect()
    }
}
