//! Audit Trail Logger (§9.4)
//!
//! Append-only JSONL at snapshots/audit_{run_id}.jsonl.
//! Every phase transition, graph mutation, validator rejection, and LLM call
//! is logged.

use chrono::Utc;
use serde::Serialize;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Debug, Serialize)]
pub struct AuditEntry {
    pub timestamp: String,
    pub round_id: u32,
    pub phase_id: u32,
    pub event_type: String,
    pub node_ids_affected: Vec<String>,
    pub token_count_delta: Option<u32>,
    pub error_code: Option<String>,
    pub detail: Option<String>,
}

impl AuditEntry {
    pub fn new(round_id: u32, phase_id: u32, event_type: &str) -> Self {
        Self {
            timestamp: Utc::now().to_rfc3339(),
            round_id,
            phase_id,
            event_type: event_type.to_string(),
            node_ids_affected: Vec::new(),
            token_count_delta: None,
            error_code: None,
            detail: None,
        }
    }

    pub fn with_nodes(mut self, ids: Vec<String>) -> Self {
        self.node_ids_affected = ids;
        self
    }

    pub fn with_tokens(mut self, count: u32) -> Self {
        self.token_count_delta = Some(count);
        self
    }

    pub fn with_error(mut self, code: &str) -> Self {
        self.error_code = Some(code.to_string());
        self
    }

    pub fn with_detail(mut self, detail: &str) -> Self {
        self.detail = Some(detail.to_string());
        self
    }
}

pub struct AuditTrail {
    path: PathBuf,
}

impl AuditTrail {
    pub fn new(snapshot_dir: &Path, run_id: &str) -> Self {
        fs::create_dir_all(snapshot_dir).ok();
        Self {
            path: snapshot_dir.join(format!("audit_{}.jsonl", run_id)),
        }
    }

    /// Append an audit entry to the trail
    pub fn log(&self, entry: &AuditEntry) {
        let line = serde_json::to_string(entry).unwrap_or_else(|e| {
            format!(
                r#"{{"error":"serialize_failed","msg":"{}"}}"#,
                e
            )
        });

        if let Ok(mut file) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
        {
            let _ = writeln!(file, "{}", line);
        }
    }

    /// Log a phase transition
    pub fn log_phase_transition(&self, round_id: u32, from_phase: u32, to_phase: u32) {
        self.log(
            &AuditEntry::new(round_id, from_phase, "phase_transition")
                .with_detail(&format!("Phase {} -> Phase {}", from_phase, to_phase)),
        );
    }

    /// Log a graph mutation
    pub fn log_mutation(&self, round_id: u32, phase_id: u32, node_ids: Vec<String>) {
        self.log(
            &AuditEntry::new(round_id, phase_id, "graph_mutation").with_nodes(node_ids),
        );
    }

    /// Log a validator rejection
    pub fn log_rejection(
        &self,
        round_id: u32,
        phase_id: u32,
        check_id: &str,
        message: &str,
    ) {
        self.log(
            &AuditEntry::new(round_id, phase_id, "validator_rejection")
                .with_error(check_id)
                .with_detail(message),
        );
    }

    /// Log an LLM call
    pub fn log_llm_call(
        &self,
        round_id: u32,
        phase_id: u32,
        role: &str,
        tokens: u32,
    ) {
        self.log(
            &AuditEntry::new(round_id, phase_id, "llm_call")
                .with_tokens(tokens)
                .with_detail(role),
        );
    }

    /// Log EIG scores and budget allocations (Phase 2)
    pub fn log_eig_allocation(
        &self,
        round_id: u32,
        allocations: &[(String, f32, u32)], // (hypothesis_id, eig, budget)
    ) {
        let detail = allocations
            .iter()
            .map(|(id, eig, budget)| format!("{}:EIG={:.3},budget={}", &id[..8], eig, budget))
            .collect::<Vec<_>>()
            .join("; ");
        self.log(
            &AuditEntry::new(round_id, 2, "eig_allocation").with_detail(&detail),
        );
    }

    /// Log convergence routing decision (Phase 7)
    pub fn log_routing(&self, round_id: u32, decision: &str, reason: &str) {
        self.log(
            &AuditEntry::new(round_id, 7, "routing_decision")
                .with_detail(&format!("{}: {}", decision, reason)),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::temp_dir;

    #[test]
    fn test_audit_trail_writes() {
        let dir = temp_dir().join("great_sage_test_audit");
        fs::create_dir_all(&dir).ok();

        let trail = AuditTrail::new(&dir, "test_run");
        trail.log_phase_transition(0, 1, 2);
        trail.log_rejection(0, 4, "V-04", "belief_score in model delta");
        trail.log_llm_call(0, 3, "Worker", 150);

        let content = fs::read_to_string(&trail.path).unwrap();
        let lines: Vec<&str> = content.trim().lines().collect();
        assert_eq!(lines.len(), 3);

        // Verify each line is valid JSON
        for line in &lines {
            let parsed: serde_json::Value = serde_json::from_str(line).unwrap();
            assert!(parsed.get("timestamp").is_some());
        }

        // Clean up
        fs::remove_dir_all(&dir).ok();
    }
}
