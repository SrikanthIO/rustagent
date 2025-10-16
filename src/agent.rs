use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub parameters: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMessage {
    pub role: String,
    pub content: String,
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    pub arguments: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResponse {
    pub message: String,
    pub tool_calls: Option<Vec<ToolCall>>,
    pub reasoning: Option<String>,
}

#[async_trait]
pub trait Agent {
    async fn process(&self, message: &str, context: &str, history: &[AgentMessage]) -> Result<AgentResponse, Box<dyn std::error::Error>>;
    fn get_available_tools(&self) -> Vec<Tool>;
}

pub struct ReasoningAgent {
    pub tools: Vec<Tool>,
    pub max_iterations: usize,
}

impl ReasoningAgent {
    pub fn new() -> Self {
        Self {
            tools: vec![
                Tool {
                    name: "search_documents".to_string(),
                    description: "Search for relevant documents in the knowledge base".to_string(),
                    parameters: HashMap::new(),
                },
                Tool {
                    name: "generate_response".to_string(),
                    description: "Generate a response based on context and reasoning".to_string(),
                    parameters: HashMap::new(),
                },
            ],
            max_iterations: 3,
        }
    }

    pub async fn multi_step_reasoning(
        &self,
        query: &str,
        rag_context: &str,
        history: &[AgentMessage],
    ) -> Result<AgentResponse, Box<dyn std::error::Error>> {
        let mut current_context = rag_context.to_string();
        let mut reasoning_steps = Vec::new();
        
        for iteration in 0..self.max_iterations {
            let step = format!("Step {}: Analyzing query and context", iteration + 1);
            reasoning_steps.push(step.clone());
            
            // Simulate reasoning process
            if query.contains("compare") || query.contains("difference") {
                reasoning_steps.push("Detected comparison query - gathering relevant information".to_string());
            } else if query.contains("best") || query.contains("recommend") {
                reasoning_steps.push("Detected recommendation query - evaluating options".to_string());
            } else {
                reasoning_steps.push("Processing general query - extracting key information".to_string());
            }
            
            if iteration == self.max_iterations - 1 {
                break;
            }
        }
        
        let reasoning = reasoning_steps.join("\n");
        
        Ok(AgentResponse {
            message: format!("Based on my analysis: {}", self.generate_final_response(query, &current_context)),
            tool_calls: None,
            reasoning: Some(reasoning),
        })
    }
    
    fn generate_final_response(&self, query: &str, context: &str) -> String {
        if context.is_empty() {
            "I don't have enough context to provide a comprehensive answer. Please provide more specific information.".to_string()
        } else {
            format!("Here's what I found based on the available context: {}", context)
        }
    }
}

#[async_trait]
impl Agent for ReasoningAgent {
    async fn process(&self, message: &str, context: &str, history: &[AgentMessage]) -> Result<AgentResponse, Box<dyn std::error::Error>> {
        self.multi_step_reasoning(message, context, history).await
    }
    
    fn get_available_tools(&self) -> Vec<Tool> {
        self.tools.clone()
    }
}

#[derive(Deserialize, Debug)]
pub struct AgentStepConfig {
    pub agent_type: String,
    pub params: serde_json::Value,
}

#[derive(Deserialize, Debug)]
pub struct AgentNodeConfig {
    pub agent_type: String,
    pub params: serde_json::Value,
    pub input: Option<serde_json::Value>,
}

#[derive(Deserialize, Debug)]
pub struct PipelineConfig {
    pub agents: Vec<AgentNodeConfig>,
}

#[async_trait]
pub trait AgentJob: Send + Sync {
    async fn run(&self, input: AgentInput) -> Result<AgentOutput, Box<dyn std::error::Error>>;
    fn from_params(params: &serde_json::Value, ctx: &OrchestratorContext) -> Box<dyn AgentJob + Send + Sync> where Self: Sized;
}

pub type AgentConstructor = fn(&serde_json::Value, &OrchestratorContext) -> Box<dyn AgentJob + Send + Sync>;

pub struct AgentRegistry {
    registry: HashMap<String, AgentConstructor>,
}
impl AgentRegistry {
    pub fn new() -> Self {
        Self { registry: HashMap::new() }
    }
    pub fn register<T: AgentJob + 'static>(&mut self, name: &str) {
        self.registry.insert(
            name.to_string(),
            |params, ctx| T::from_params(params, ctx),
        );
    }
    pub fn get(&self, name: &str, params: &serde_json::Value, ctx: &OrchestratorContext) -> Option<Box<dyn AgentJob + Send + Sync>> {
        self.registry.get(name).map(|ctor| ctor(params, ctx))
    }
}

/// Standard registry with framework-provided agent types
pub fn agent_registry_with_defaults() -> AgentRegistry {
    let mut reg = AgentRegistry::new();
    reg.register::<EmailReaderAgent>("EmailReaderAgent");
    reg.register::<RAGContextAgent>("RAGContextAgent");
    reg.register::<ReplyGeneratorAgent>("ReplyGeneratorAgent");
    reg.register::<LeadDataReaderAgent>("LeadDataReaderAgent");
    reg.register::<FeatureEngineerAgent>("FeatureEngineerAgent");
    reg.register::<ModelOptimizeAgent>("ModelOptimizeAgent");
    reg.register::<ReportWriterAgent>("ReportWriterAgent");
    reg.register::<RemoteHttpAgent>("RemoteHttpAgent");
    reg
}

#[derive(Debug, Clone)]
pub enum AgentInput {
    Email(String),
    EmailWithContext { email: String, context: String },
}

#[derive(Debug, Clone)]
pub enum AgentOutput {
    StructEmail { from: String, subject: String, body: String },
    ContextResult(String),
    ReplyDraft(String),
    // fallback generic
    Other(serde_json::Value),
}

// Dummy RAG + LLM context holder for stubs
#[derive(Clone)]
pub struct OrchestratorContext {
    pub rag: Option<std::sync::Arc<crate::RAG>>,
    pub llm_config: Option<String>, // Simple
}

// Each agent impls from_params and AgentJob
pub struct EmailReaderAgent;
impl EmailReaderAgent {
    pub fn from_params(_params: &serde_json::Value, _ctx: &OrchestratorContext) -> Box<dyn AgentJob + Send + Sync> { Box::new(Self) }
}
#[async_trait]
impl AgentJob for EmailReaderAgent {
    async fn run(&self, input: AgentInput) -> Result<AgentOutput, Box<dyn std::error::Error>> {
        Ok(AgentOutput::StructEmail { from: "from@demo.com".into(), subject: "Test Subject".into(), body: "Email Body".into() })
    }
    fn from_params(params: &serde_json::Value, ctx: &OrchestratorContext) -> Box<dyn AgentJob + Send + Sync> { Self::from_params(params, ctx) }
}

pub struct RAGContextAgent { pub rag: std::sync::Arc<crate::RAG> }
impl RAGContextAgent {
    pub fn from_params(_params: &serde_json::Value, ctx: &OrchestratorContext) -> Box<dyn AgentJob + Send + Sync> {
        Box::new(Self { rag: ctx.rag.clone().unwrap() })
    }
}
#[async_trait]
impl AgentJob for RAGContextAgent {
    async fn run(&self, input: AgentInput) -> Result<AgentOutput, Box<dyn std::error::Error>> {
        let query = match input {
            AgentInput::Email(body) | AgentInput::EmailWithContext { email: body, .. } => body,
        };
        let ctx = self.rag.retrieve_context(&query, 2).await?;
        Ok(AgentOutput::ContextResult(ctx))
    }
    fn from_params(params: &serde_json::Value, ctx: &OrchestratorContext) -> Box<dyn AgentJob + Send + Sync> {
        Self::from_params(params, ctx)
    }
}

pub struct ReplyGeneratorAgent { pub llm_config: String }
impl ReplyGeneratorAgent {
    pub fn from_params(params: &serde_json::Value, ctx: &OrchestratorContext) -> Box<dyn AgentJob + Send + Sync> {
        let config = ctx.llm_config.clone()
            .or_else(|| params.get("model").and_then(|v| v.as_str().map(|s| s.to_string())))
            .unwrap_or_else(|| "llama3.2".to_string());
        Box::new(Self { llm_config: config })
    }
}
#[async_trait]
impl AgentJob for ReplyGeneratorAgent {
    async fn run(&self, _input: AgentInput) -> Result<AgentOutput, Box<dyn std::error::Error>> {
        let reply = "Thank you for your email. [demo reply]".to_string();
        let tokens = estimate_tokens(&reply);
        Ok(AgentOutput::ReplyDraft(format!("{} [tokens:{}]", reply, tokens)))
    }
    fn from_params(params: &serde_json::Value, ctx: &OrchestratorContext) -> Box<dyn AgentJob + Send + Sync> {
        Self::from_params(params, ctx)
    }
}

pub struct LeadDataReaderAgent;
impl LeadDataReaderAgent {
    pub fn from_params(_params: &serde_json::Value, _ctx: &OrchestratorContext) -> Box<dyn AgentJob + Send + Sync> { Box::new(Self) }
}
#[async_trait]
impl AgentJob for LeadDataReaderAgent {
    async fn run(&self, _input: AgentInput) -> Result<AgentOutput, Box<dyn std::error::Error>> {
        // Returns AgentOutput::Other with mock lead data
        Ok(AgentOutput::Other(serde_json::json!({
            "campaign": "Spring Promo",
            "leads": [{"id": 1, "score": 0.8}, {"id": 2, "score": 0.6}]
        })))
    }
    fn from_params(params: &serde_json::Value, ctx: &OrchestratorContext) -> Box<dyn AgentJob + Send + Sync> {
        Self::from_params(params, ctx)
    }
}

pub struct FeatureEngineerAgent;
impl FeatureEngineerAgent {
    pub fn from_params(_params: &serde_json::Value, _ctx: &OrchestratorContext) -> Box<dyn AgentJob + Send + Sync> { Box::new(Self) }
}
#[async_trait]
impl AgentJob for FeatureEngineerAgent {
    async fn run(&self, input: AgentInput) -> Result<AgentOutput, Box<dyn std::error::Error>> {
        // Pretend to add features
        let engineered = serde_json::json!({ "features": [1.0, 0.5, 0.7] });
        Ok(AgentOutput::Other(engineered))
    }
    fn from_params(params: &serde_json::Value, ctx: &OrchestratorContext) -> Box<dyn AgentJob + Send + Sync> {
        Self::from_params(params, ctx)
    }
}

pub struct ModelOptimizeAgent;
impl ModelOptimizeAgent {
    pub fn from_params(_params: &serde_json::Value, _ctx: &OrchestratorContext) -> Box<dyn AgentJob + Send + Sync> { Box::new(Self) }
}
#[async_trait]
impl AgentJob for ModelOptimizeAgent {
    async fn run(&self, input: AgentInput) -> Result<AgentOutput, Box<dyn std::error::Error>> {
        let mock_report = serde_json::json!({ "model": "random_forest", "roc_auc": 0.87 });
        Ok(AgentOutput::Other(mock_report))
    }
    fn from_params(params: &serde_json::Value, ctx: &OrchestratorContext) -> Box<dyn AgentJob + Send + Sync> {
        Self::from_params(params, ctx)
    }
}

pub struct ReportWriterAgent;
impl ReportWriterAgent {
    pub fn from_params(_params: &serde_json::Value, _ctx: &OrchestratorContext) -> Box<dyn AgentJob + Send + Sync> { Box::new(Self) }
}
#[async_trait]
impl AgentJob for ReportWriterAgent {
    async fn run(&self, input: AgentInput) -> Result<AgentOutput, Box<dyn std::error::Error>> {
        // Output summary report string
        Ok(AgentOutput::ReplyDraft("Campaign optimization completed and report generated.".to_string()))
    }
    fn from_params(params: &serde_json::Value, ctx: &OrchestratorContext) -> Box<dyn AgentJob + Send + Sync> {
        Self::from_params(params, ctx)
    }
}

pub struct RemoteHttpAgent {
    pub url: String,
    pub method: String,
    pub headers: Option<std::collections::HashMap<String, String>>,
    pub timeout_ms: Option<u64>,
    pub bearer_token: Option<String>,
    pub retry_count: u32,
    pub retry_backoff_ms: u64,
}
impl RemoteHttpAgent {
    pub fn from_params(params: &serde_json::Value, _ctx: &OrchestratorContext) -> Box<dyn AgentJob + Send + Sync> {
        let url = params.get("url").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let method = params.get("method").and_then(|v| v.as_str()).unwrap_or("POST").to_string();
        let headers = params.get("headers").and_then(|v| v.as_object()).map(|m| {
            m.iter().map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").to_string())).collect()
        });
        let timeout_ms = params.get("timeout_ms").and_then(|v| v.as_u64());
        let bearer_token = params.get("bearer_token").and_then(|v| v.as_str()).map(|s| s.to_string());
        let retry_count = params.get("retry_count").and_then(|v| v.as_u64()).unwrap_or(2) as u32;
        let retry_backoff_ms = params.get("retry_backoff_ms").and_then(|v| v.as_u64()).unwrap_or(500);
        Box::new(Self { url, method, headers, timeout_ms, bearer_token, retry_count, retry_backoff_ms })
    }
}
#[async_trait]
impl AgentJob for RemoteHttpAgent {
    async fn run(&self, input: AgentInput) -> Result<AgentOutput, Box<dyn std::error::Error>> {
        if self.url.is_empty() { return Err("RemoteHttpAgent missing required 'url'".into()); }
        let client = reqwest::Client::builder()
            .timeout(self.timeout_ms.map(std::time::Duration::from_millis))
            .build()?;
        let body = match &input {
            AgentInput::Email(s) => serde_json::json!({"type":"Email","email":s}),
            AgentInput::EmailWithContext { email, context } => serde_json::json!({"type":"EmailWithContext","email":email,"context":context}),
        };

        let mut attempt = 0u32;
        let max_attempts = self.retry_count.saturating_add(1);
        loop {
            attempt += 1;
            let mut req = match self.method.as_str() {
                "GET" => client.get(&self.url),
                "PUT" => client.put(&self.url).json(&body),
                "POST" => client.post(&self.url).json(&body),
                "DELETE" => client.delete(&self.url).json(&body),
                _ => client.post(&self.url).json(&body),
            };
            if let Some(h) = &self.headers {
                for (k, v) in h {
                    req = req.header(k, v);
                }
            }
            if let Some(token) = &self.bearer_token {
                req = req.bearer_auth(token);
            }

            let resp_result = req.send().await;
            match resp_result {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        // Try to parse structured response
                        let json_val: serde_json::Value = resp.json().await.unwrap_or(serde_json::json!({"status":"ok"}));
                        // Map known fields to AgentOutput variants when possible
                        if let Some(reply) = json_val.get("reply").and_then(|v| v.as_str()) {
                            let tokens = estimate_tokens(reply);
                            return Ok(AgentOutput::ReplyDraft(format!("{} [tokens:{}]", reply, tokens)));
                        }
                        if let Some(ctx) = json_val.get("context").and_then(|v| v.as_str()) {
                            return Ok(AgentOutput::ContextResult(ctx.to_string()));
                        }
                        return Ok(AgentOutput::Other(json_val));
                    } else {
                        // Retry on 429/5xx; otherwise treat as terminal error
                        if status.as_u16() == 429 || status.is_server_error() {
                            if attempt < max_attempts {
                                tokio::time::sleep(std::time::Duration::from_millis(self.retry_backoff_ms * attempt as u64)).await;
                                continue;
                            }
                        }
                        return Err(format!("RemoteHttpAgent error: status {}", status).into());
                    }
                }
                Err(e) => {
                    if attempt < max_attempts {
                        tokio::time::sleep(std::time::Duration::from_millis(self.retry_backoff_ms * attempt as u64)).await;
                        continue;
                    }
                    return Err(format!("RemoteHttpAgent request failed: {}", e).into());
                }
            }
        }
    }
    fn from_params(params: &serde_json::Value, ctx: &OrchestratorContext) -> Box<dyn AgentJob + Send + Sync> {
        Self::from_params(params, ctx)
    }
}

fn estimate_tokens(text: &str) -> usize {
    // Simple heuristic: ~1 token per 4 chars (fallback to whitespace split)
    let by_len = (text.len() as f64 / 4.0).ceil() as usize;
    let by_ws = text.split_whitespace().count();
    std::cmp::max(by_len, by_ws)
}

pub struct TraceRecorder {
    pool: sqlx::SqlitePool,
}
impl TraceRecorder {
    pub async fn new(database_url: &str) -> Result<Self, sqlx::Error> {
        let pool = sqlx::SqlitePool::connect(database_url).await?;
        // Create table if not exists
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS traces (\n                id INTEGER PRIMARY KEY AUTOINCREMENT,\n                ts TEXT NOT NULL,\n                workflow TEXT NOT NULL,\n                agent_type TEXT NOT NULL,\n                step_index INTEGER NOT NULL,\n                input_len INTEGER NOT NULL,\n                output_kind TEXT NOT NULL,\n                duration_ms INTEGER NOT NULL\n            )"
        ).execute(&pool).await?;
        Ok(Self { pool })
    }
    pub async fn record(&self, workflow: &str, agent_type: &str, step_index: usize, input_len: usize, output_kind: &str, duration_ms: i64) -> Result<(), sqlx::Error> {
        let ts: DateTime<Utc> = Utc::now();
        sqlx::query("INSERT INTO traces (ts, workflow, agent_type, step_index, input_len, output_kind, duration_ms) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)")
            .bind(ts.to_rfc3339())
            .bind(workflow)
            .bind(agent_type)
            .bind(step_index as i64)
            .bind(input_len as i64)
            .bind(output_kind)
            .bind(duration_ms)
            .execute(&self.pool)
            .await?;
        Ok(())
    }
}

pub struct AgentOrchestrator {
    pub steps: Vec<Box<dyn AgentJob + Send + Sync>>,
    pub tracer: Option<std::sync::Arc<TraceRecorder>>,
    pub workflow_name: String,
}
impl AgentOrchestrator {
    pub fn from_json_registry(conf: PipelineConfig, ctx: OrchestratorContext, registry: &AgentRegistry) -> Self {
        let steps = conf.agents
            .into_iter()
            .map(|step| registry.get(&step.agent_type, &step.params, &ctx).expect("Unknown agent type"))
            .collect();
        Self { steps, tracer: None, workflow_name: "default".to_string() }
    }
    pub fn with_tracer(mut self, tracer: std::sync::Arc<TraceRecorder>, workflow_name: impl Into<String>) -> Self {
        self.tracer = Some(tracer);
        self.workflow_name = workflow_name.into();
        self
    }
    pub async fn handle_job(&self, initial: AgentInput) -> Result<AgentOutput, Box<dyn std::error::Error>> {
        let mut data = initial;
        for (idx, agent) in self.steps.iter().enumerate() {
            let start = std::time::Instant::now();
            let input_len = match &data { AgentInput::Email(s) => s.len(), AgentInput::EmailWithContext { email, context } => email.len() + context.len() };
            let out = agent.run(data).await?;
            let duration = start.elapsed().as_millis() as i64;
            let output_kind = match &out { AgentOutput::StructEmail { .. } => "StructEmail", AgentOutput::ContextResult(_) => "ContextResult", AgentOutput::ReplyDraft(_) => "ReplyDraft", AgentOutput::Other(_) => "Other" };
            if let Some(rec) = &self.tracer {
                let _ = rec.record(&self.workflow_name, std::any::type_name::<&Box<dyn AgentJob + Send + Sync>>(), idx, input_len, output_kind, duration).await;
            }
            data = match out {
                AgentOutput::StructEmail { from, subject, body } => AgentInput::EmailWithContext { email: body, context: "".to_string() },
                AgentOutput::ContextResult(ctx) => AgentInput::EmailWithContext { email: "".to_string(), context: ctx },
                AgentOutput::ReplyDraft(reply) => {
                    // Ensure token annotation even if upstream forgot
                    let annotated = if reply.contains("[tokens:") { reply } else { format!("{} [tokens:{}]", reply, estimate_tokens(&reply)) };
                    return Ok(AgentOutput::ReplyDraft(annotated));
                },
                AgentOutput::Other(v) => { println!("Unrecognized output: {v:?}"); return Ok(AgentOutput::Other(v)); }
            };
        }
        Ok(AgentOutput::Other(serde_json::json!({ "status": "No reply draft returned" })))
    }
}
