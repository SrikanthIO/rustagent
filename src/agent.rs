use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
