//! Tool Types (§7.2)
//!
//! ToolCall and ToolResult types for the code agent extension.
//! ToolResult::CodeRun is the primary evidence-gathering mechanism.

use serde::{Deserialize, Serialize};

/// Tool call types — model proposes these during Phase 3 Execute
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ToolCall {
    Search {
        query: String,
    },
    Visit {
        url: String,
    },
    CodeRun {
        command: String,
        working_dir: String,
        #[serde(default = "default_timeout")]
        timeout_secs: u32,
    },
}

fn default_timeout() -> u32 {
    30
}

/// Tool result types — returned by tool executor, consumed by Phase 4 Compress
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ToolResult {
    Search {
        snippets: Vec<String>,
    },
    Visit {
        content: String,
        /// Structured tree representation of the page (interactables, headings, etc.)
        tree: Option<Vec<PageElement>>,
    },
    CodeRun {
        stdout: String,
        stderr: String,
        exit_code: i32,
        tests_passed: Vec<String>,
        tests_failed: Vec<String>,
        duration_ms: u32,
    },
}

/// Structured element from a web page tree decomposition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageElement {
    pub tag: String,
    pub text: Option<String>,
    pub href: Option<String>,
    pub children: Vec<PageElement>,
    pub is_interactable: bool,
}

/// Inferred claims from a tool execution (model-proposed, goes to Compress)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolExecutionResult {
    pub tool_call: ToolCall,
    pub result: ToolResult,
    pub inferred_claim_texts: Vec<String>,
}

impl ToolResult {
    /// Check if a CodeRun result represents all tests passing
    pub fn all_tests_pass(&self) -> bool {
        match self {
            ToolResult::CodeRun {
                exit_code,
                tests_passed,
                tests_failed,
                ..
            } => *exit_code == 0 && tests_failed.is_empty() && !tests_passed.is_empty(),
            _ => false,
        }
    }

    /// Check if this is a compile/build error (exit_code != 0, no tests parsed)
    pub fn is_build_error(&self) -> bool {
        match self {
            ToolResult::CodeRun {
                exit_code,
                tests_passed,
                tests_failed,
                ..
            } => *exit_code != 0 && tests_passed.is_empty() && tests_failed.is_empty(),
            _ => false,
        }
    }
}
