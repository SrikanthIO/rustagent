# Agentic RAG Framework (Rust + Ollama + HNSW)

This project implements an agentic Retrieval-Augmented Generation (RAG) pipeline in Rust using:

- Ollama for local LLMs and embeddings (Llama 3.2 + mxbai-embed-large:latest)
- HNSW (pure Rust) for fast vector search
- Axum for an optional REST API server
- PDF/text ingestion from the `documents/` directory

## Prerequisites

- Rust (stable)
- Ollama installed and running
- Models:
  - `llama3.2`
  - `mxbai-embed-large:latest`

Install/run Ollama and pull models:

```bash
# Ensure Ollama is installed and running
ollama serve &

# Pull/prepare models
ollama run llama3.2
ollama run mxbai-embed-large:latest
```

## Project Setup

```bash
cd agentic_rag_framework
cargo build
```

### Documents
Place your `.txt` or `.pdf` files in:

```
agentic_rag_framework/documents/
```

Example provided: sample HR policy PDFs and text files.

## Run - Interactive Chat (Terminal)

```bash
cargo run
```

- Loads documents (PDF/text), generates embeddings via Ollama, builds an HNSW index.
- Starts an interactive REPL. Type your question. Type `:exit` to quit.

## Run - REST API Server

```bash
cargo run -- server
```

Server defaults:
- Host: `0.0.0.0`
- Port: `3000`
- Health: `GET /health`
- Chat: `POST /chat`

Environment overrides:

```bash
export OLLAMA_HOST="http://localhost"
export OLLAMA_PORT=11434
export EMBED_MODEL="mxbai-embed-large:latest"
export LLM_MODEL="llama3.2"
export DOCS_DIR="documents"
export SERVER_HOST="0.0.0.0"
export SERVER_PORT=3000
export MAX_CONTEXT_DOCS=3
export EF_SEARCH=64
cargo run -- server
```

## API

### Health
```
GET http://localhost:3000/health
```
Response:
```json
{
  "status": "healthy",
  "message": "Agentic RAG Framework is running"
}
```

### Chat
```
POST http://localhost:3000/chat
Content-Type: application/json
```
Body:
```json
{
  "message": "What are the key HR policies?",
  "k": 2,
  "history": [["User", "hello"], ["Assistant", "hi!"]]
}
```
Response:
```json
{
  "response": "... model reply ...",
  "context": "... retrieved documents ..."
}
```

## Postman Collection

A Postman collection is provided under `docs/postman_collection.json`.

### Import & Use
1. Open Postman -> Import -> Upload `docs/postman_collection.json`.
2. Ensure the server is running: `cargo run -- server`.
3. Use the `Health` and `Chat` requests in the collection.

## Postman Collection JSON

Save the following as `docs/postman_collection.json`:

```json
{
  "info": {
    "name": "Agentic RAG Framework",
    "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
  },
  "item": [
    {
      "name": "Health",
      "request": {
        "method": "GET",
        "url": {
          "raw": "http://localhost:3000/health",
          "protocol": "http",
          "host": ["localhost"],
          "port": "3000",
          "path": ["health"]
        }
      }
    },
    {
      "name": "Chat",
      "request": {
        "method": "POST",
        "header": [{"key": "Content-Type", "value": "application/json"}],
        "body": {
          "mode": "raw",
          "raw": "{\n  \"message\": \"What are the key HR policies?\",\n  \"k\": 2,\n  \"history\": [[\"User\", \"hello\"], [\"Assistant\", \"hi!\"]]\n}"
        },
        "url": {
          "raw": "http://localhost:3000/chat",
          "protocol": "http",
          "host": ["localhost"],
          "port": "3000",
          "path": ["chat"]
        }
      }
    }
  ]
}
```

## Troubleshooting
- Ensure Ollama is running and the models are pulled.
- If you see an embeddings error, verify `OLLAMA_HOST/PORT` and that `mxbai-embed-large:latest` is available.
- For PDF parsing issues, confirm the PDFs are not encrypted and readable.

## License
MIT
