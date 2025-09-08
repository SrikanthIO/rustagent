use std::fs;
use ollama_rs::Ollama;
use hnsw_rs::prelude::*;

pub struct RAG {
    pub ollama: Ollama,
    embed_model: String,
    doc_texts: Vec<String>,
    doc_titles: Vec<String>,
    hnsw: Hnsw<f32, DistCosine>,
}

impl RAG {
    pub async fn new(ollama: Ollama, embed_model: String, docs_dir: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let mut doc_texts: Vec<String> = Vec::new();
        let mut doc_titles: Vec<String> = Vec::new();
        let mut doc_embeddings: Vec<Vec<f32>> = Vec::new();

        // Load and embed documents
        for entry in fs::read_dir(docs_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() {
                let text = fs::read_to_string(&path)?;
                let embed_response = ollama
                    .generate_embeddings(embed_model.clone(), text.clone(), None)
                    .await?;
                let embedding_f32 = Self::f64_to_f32_vec(&embed_response.embeddings);
                doc_titles.push(path.file_name().unwrap().to_string_lossy().to_string());
                doc_texts.push(text);
                doc_embeddings.push(embedding_f32);
            }
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
        // Embed the query
        let query_embed_f64 = self.ollama
            .generate_embeddings(self.embed_model.clone(), query.to_string(), None)
            .await?;
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
