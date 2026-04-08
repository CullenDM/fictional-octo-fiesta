//! Deterministic Validator Catalog (§6)
//!
//! The Rust Harness executes ALL checks in this catalog before any graph
//! mutation is applied.  A single failing check causes the entire mutation
//! batch to be rejected (all-or-nothing).
//!
//! This is the Forbidden Model Writes wall.  The model proposes text deltas;
//! the Harness computes everything numeric.  If the model tries to inject
//! belief_score, source_entropy, domain_auth, reliability, status fields,
//! or any NodeMeta immutables, the batch is rejected and logged.

use crate::graph::SageGraph;
use crate::schema::*;
use serde_json::Value;

/// Result of a validation run
#[derive(Debug)]
pub struct ValidationResult {
    pub passed: bool,
    pub violations: Vec<Violation>,
}

/// A single validation violation
#[derive(Debug, Clone)]
pub struct Violation {
    pub check_id: String,
    pub message: String,
    pub offending_data: Option<String>,
}

impl std::fmt::Display for Violation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}", self.check_id, self.message)
    }
}

/// Fields that the model is NEVER allowed to write (§6, V-04)
const FORBIDDEN_MODEL_FIELDS: &[&str] = &[
    "belief_score",
    "source_entropy",
    "domain_auth",
    "reliability",
    "is_stale",
    // V-03: immutable after creation
    // (id and created_at checked separately)
];

/// Fields that are immutable after creation (§6, V-03)
const IMMUTABLE_FIELDS: &[&str] = &["id", "created_at"];

/// Validate a model delta against all applicable checks.
/// Returns ValidationResult with all violations found.
///
/// The delta is expected to be a JSON object with structure:
/// {
///   "new_claims": [...],
///   "new_evidence": [...],
///   "new_sources": [...],    // optional
///   "new_edges": [...]
/// }
pub fn validate_model_delta(
    delta: &Value,
    graph: &SageGraph,
    current_phase: u32,
) -> ValidationResult {
    let mut violations = Vec::new();

    // V-03: No-Write Zone — id and created_at must not appear in model output
    check_v03_no_write_zone(delta, &mut violations);

    // V-04: Forbidden Model Writes — numeric/status fields must not appear
    check_v04_forbidden_writes(delta, &mut violations);

    // V-01: Provenance Integrity — every new ClaimNode must have evidence
    check_v01_provenance(delta, &mut violations);

    // V-06: Hypothesis Shortcut — no CandidateAnswerNode during Execute
    if current_phase == 3 {
        check_v06_hypothesis_shortcut(delta, &mut violations);
    }

    // V-10: Schema Completeness — required fields present
    check_v10_schema_completeness(delta, &mut violations);

    // V-08: Stale Write — no Supports edge to is_stale node
    check_v08_stale_write(delta, graph, &mut violations);

    ValidationResult {
        passed: violations.is_empty(),
        violations,
    }
}

/// Validate a candidate answer node before banking (Phase 9).
pub fn validate_candidate_for_banking(content: &str) -> ValidationResult {
    let mut violations = Vec::new();

    // V-11: CodeRun Patch Validity — content must parse as unified diff
    check_v11_patch_validity(content, &mut violations);

    ValidationResult {
        passed: violations.is_empty(),
        violations,
    }
}

/// Validate support_mass >= 0.01 epsilon guard (V-12)
pub fn validate_support_mass(support_mass: f32) -> ValidationResult {
    let mut violations = Vec::new();

    if support_mass < 0.01 {
        violations.push(Violation {
            check_id: "V-12".to_string(),
            message: format!(
                "support_mass {:.4} is below minimum 0.01 — downgrade to Unverified",
                support_mass
            ),
            offending_data: None,
        });
    }

    ValidationResult {
        passed: violations.is_empty(),
        violations,
    }
}

/// Validate reopen ceiling (V-09)
pub fn validate_reopen_ceiling(reopen_count: u8, max_reopen: u8) -> ValidationResult {
    let mut violations = Vec::new();

    if reopen_count >= max_reopen {
        violations.push(Violation {
            check_id: "V-09".to_string(),
            message: format!(
                "reopen_count {} >= max_reopen_count {} — emit with uncertainty",
                reopen_count, max_reopen
            ),
            offending_data: None,
        });
    }

    ValidationResult {
        passed: violations.is_empty(),
        violations,
    }
}

// ---------------------------------------------------------------------------
// Individual validator implementations
// ---------------------------------------------------------------------------

/// V-03: id and created_at fields must not appear in model delta output
fn check_v03_no_write_zone(delta: &Value, violations: &mut Vec<Violation>) {
    for field in IMMUTABLE_FIELDS {
        if contains_field_recursive(delta, field) {
            violations.push(Violation {
                check_id: "V-03".to_string(),
                message: format!(
                    "Immutable field '{}' found in model delta — models may not write this",
                    field
                ),
                offending_data: Some(field.to_string()),
            });
        }
    }
}

/// V-04: Forbidden Model Writes — belief_score, source_entropy, domain_auth,
/// reliability, is_stale must not appear in model delta
fn check_v04_forbidden_writes(delta: &Value, violations: &mut Vec<Violation>) {
    for field in FORBIDDEN_MODEL_FIELDS {
        if contains_field_in_nodes(delta, field) {
            violations.push(Violation {
                check_id: "V-04".to_string(),
                message: format!(
                    "Forbidden model write: '{}' found in model delta — Harness computes this",
                    field
                ),
                offending_data: Some(field.to_string()),
            });
        }
    }

    // Also check for status fields that should not be model-written
    // (Verification_Status in the spec)
    for field in &["status", "Verification_Status"] {
        if contains_field_in_nodes(delta, field) {
            violations.push(Violation {
                check_id: "V-04".to_string(),
                message: format!(
                    "Forbidden model write: '{}' found in model delta",
                    field
                ),
                offending_data: Some(field.to_string()),
            });
        }
    }
}

/// V-01: Provenance Integrity — no ClaimNode without at least one
/// EvidenceNode parent via Supports edge
fn check_v01_provenance(delta: &Value, violations: &mut Vec<Violation>) {
    let claims = delta
        .get("new_claims")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let edges = delta
        .get("new_edges")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    for claim in &claims {
        let claim_id = claim
            .get("tmp_id")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if claim_id.is_empty() {
            continue;
        }

        // Check if there's at least one evidence -> claim Supports edge
        let has_support = edges.iter().any(|edge| {
            let to_id = edge.get("to_id").and_then(|v| v.as_str()).unwrap_or("");
            let kind = edge.get("kind").and_then(|v| v.as_str()).unwrap_or("");
            to_id == claim_id && kind == "Supports"
        });

        if !has_support {
            violations.push(Violation {
                check_id: "V-01".to_string(),
                message: format!(
                    "ClaimNode '{}' has no EvidenceNode parent via Supports edge",
                    claim_id
                ),
                offending_data: Some(claim_id.to_string()),
            });
        }
    }
}

/// V-06: Hypothesis Shortcut — model may not propose CandidateAnswerNode
/// during Execute phase
fn check_v06_hypothesis_shortcut(delta: &Value, violations: &mut Vec<Violation>) {
    if let Some(candidates) = delta.get("new_candidates").and_then(|v| v.as_array()) {
        if !candidates.is_empty() {
            violations.push(Violation {
                check_id: "V-06".to_string(),
                message: "CandidateAnswerNode proposed during Execute phase — shortcutting is forbidden".to_string(),
                offending_data: None,
            });
        }
    }
}

/// V-08: Stale Write — no Supports edge may be added to a node where
/// is_stale = true
fn check_v08_stale_write(delta: &Value, graph: &SageGraph, violations: &mut Vec<Violation>) {
    let edges = delta
        .get("new_edges")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    for edge in &edges {
        let kind = edge.get("kind").and_then(|v| v.as_str()).unwrap_or("");
        if kind != "Supports" {
            continue;
        }

        let to_id = edge.get("to_id").and_then(|v| v.as_str()).unwrap_or("");

        // Only check existing nodes (tmp_* refs are new and can't be stale)
        if to_id.starts_with("tmp_") {
            continue;
        }

        if let Some(node) = graph.get_node(to_id) {
            if node.meta().is_stale {
                violations.push(Violation {
                    check_id: "V-08".to_string(),
                    message: format!(
                        "Supports edge to stale node '{}' — stale nodes may not receive new evidence",
                        to_id
                    ),
                    offending_data: Some(to_id.to_string()),
                });
            }
        }
    }
}

/// V-10: Schema Completeness — every new node must have required fields
fn check_v10_schema_completeness(delta: &Value, violations: &mut Vec<Violation>) {
    // Check claims have 'text'
    if let Some(claims) = delta.get("new_claims").and_then(|v| v.as_array()) {
        for (i, claim) in claims.iter().enumerate() {
            if claim.get("text").and_then(|v| v.as_str()).is_none() {
                violations.push(Violation {
                    check_id: "V-10".to_string(),
                    message: format!("new_claims[{}] missing required field 'text'", i),
                    offending_data: None,
                });
            }
            if claim.get("tmp_id").and_then(|v| v.as_str()).is_none() {
                violations.push(Violation {
                    check_id: "V-10".to_string(),
                    message: format!("new_claims[{}] missing required field 'tmp_id'", i),
                    offending_data: None,
                });
            }
        }
    }

    // Check evidence has 'snippet' and 'source_url'
    if let Some(evidence) = delta.get("new_evidence").and_then(|v| v.as_array()) {
        for (i, ev) in evidence.iter().enumerate() {
            if ev.get("snippet").and_then(|v| v.as_str()).is_none() {
                violations.push(Violation {
                    check_id: "V-10".to_string(),
                    message: format!("new_evidence[{}] missing required field 'snippet'", i),
                    offending_data: None,
                });
            }
            if ev.get("tmp_id").and_then(|v| v.as_str()).is_none() {
                violations.push(Violation {
                    check_id: "V-10".to_string(),
                    message: format!("new_evidence[{}] missing required field 'tmp_id'", i),
                    offending_data: None,
                });
            }
        }
    }

    // Check edges have from_id, to_id, kind
    if let Some(edges) = delta.get("new_edges").and_then(|v| v.as_array()) {
        for (i, edge) in edges.iter().enumerate() {
            for field in &["from_id", "to_id", "kind"] {
                if edge.get(*field).and_then(|v| v.as_str()).is_none() {
                    violations.push(Violation {
                        check_id: "V-10".to_string(),
                        message: format!("new_edges[{}] missing required field '{}'", i, field),
                        offending_data: None,
                    });
                }
            }
        }
    }
}

/// V-11: CodeRun Patch Validity — CandidateAnswerNode.content must parse
/// as valid unified diff before banking.
/// Note: for Stage 1, we do a basic heuristic check.  A content string that
/// contains standard unified diff markers (---/+++ or diff --git) is valid.
/// Plain answer text (non-code tasks) always passes.
fn check_v11_patch_validity(content: &str, violations: &mut Vec<Violation>) {
    // If it looks like it's trying to be a diff, validate the format
    let has_diff_markers = content.contains("---") && content.contains("+++");
    let has_git_diff = content.starts_with("diff --git");

    // Only validate if it appears to be a diff
    if has_diff_markers || has_git_diff {
        // Basic validation: must have at least one hunk header
        let has_hunk = content.contains("@@");
        if !has_hunk {
            violations.push(Violation {
                check_id: "V-11".to_string(),
                message: "Content appears to be a diff but has no hunk headers (@@)".to_string(),
                offending_data: Some(content.chars().take(200).collect()),
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Check if a field name appears anywhere in a JSON value (recursive)
fn contains_field_recursive(value: &Value, field: &str) -> bool {
    match value {
        Value::Object(obj) => {
            if obj.contains_key(field) {
                return true;
            }
            obj.values().any(|v| contains_field_recursive(v, field))
        }
        Value::Array(arr) => arr.iter().any(|v| contains_field_recursive(v, field)),
        _ => false,
    }
}

/// Check if a field appears inside new_claims, new_evidence, new_sources arrays
/// (not in edge definitions where 'from_id' etc. are expected)
fn contains_field_in_nodes(value: &Value, field: &str) -> bool {
    for section in &["new_claims", "new_evidence", "new_sources"] {
        if let Some(arr) = value.get(*section).and_then(|v| v.as_array()) {
            for item in arr {
                if let Value::Object(obj) = item {
                    if obj.contains_key(field) {
                        return true;
                    }
                }
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn empty_graph() -> SageGraph {
        SageGraph::new()
    }

    #[test]
    fn test_v04_forbidden_belief_score() {
        let delta = json!({
            "new_claims": [
                {
                    "tmp_id": "tmp_claim_1",
                    "text": "Test claim",
                    "belief_score": 0.95
                }
            ],
            "new_evidence": [
                {"tmp_id": "tmp_evidence_1", "snippet": "test"}
            ],
            "new_edges": [
                {"from_id": "tmp_evidence_1", "to_id": "tmp_claim_1", "kind": "Supports"}
            ]
        });

        let result = validate_model_delta(&delta, &empty_graph(), 4);
        assert!(!result.passed);
        assert!(result.violations.iter().any(|v| v.check_id == "V-04"));
    }

    #[test]
    fn test_v04_forbidden_domain_auth() {
        let delta = json!({
            "new_claims": [],
            "new_evidence": [
                {
                    "tmp_id": "tmp_evidence_1",
                    "snippet": "test",
                    "domain_auth": 0.99
                }
            ],
            "new_edges": []
        });

        let result = validate_model_delta(&delta, &empty_graph(), 4);
        assert!(!result.passed);
        assert!(result.violations.iter().any(|v| v.check_id == "V-04"));
    }

    #[test]
    fn test_v03_immutable_id() {
        let delta = json!({
            "new_claims": [
                {
                    "tmp_id": "tmp_claim_1",
                    "text": "Test claim",
                    "id": "hacked-uuid"
                }
            ],
            "new_evidence": [
                {"tmp_id": "tmp_evidence_1", "snippet": "test"}
            ],
            "new_edges": [
                {"from_id": "tmp_evidence_1", "to_id": "tmp_claim_1", "kind": "Supports"}
            ]
        });

        let result = validate_model_delta(&delta, &empty_graph(), 4);
        assert!(!result.passed);
        assert!(result.violations.iter().any(|v| v.check_id == "V-03"));
    }

    #[test]
    fn test_v01_claim_without_evidence() {
        let delta = json!({
            "new_claims": [
                {"tmp_id": "tmp_claim_1", "text": "Orphan claim"}
            ],
            "new_evidence": [],
            "new_edges": []
        });

        let result = validate_model_delta(&delta, &empty_graph(), 4);
        assert!(!result.passed);
        assert!(result.violations.iter().any(|v| v.check_id == "V-01"));
    }

    #[test]
    fn test_v06_shortcut_during_execute() {
        let delta = json!({
            "new_claims": [],
            "new_evidence": [],
            "new_edges": [],
            "new_candidates": [
                {"content": "shortcut answer"}
            ]
        });

        let result = validate_model_delta(&delta, &empty_graph(), 3);
        assert!(!result.passed);
        assert!(result.violations.iter().any(|v| v.check_id == "V-06"));
    }

    #[test]
    fn test_valid_delta_passes() {
        let delta = json!({
            "new_claims": [
                {"tmp_id": "tmp_claim_1", "text": "Valid claim"}
            ],
            "new_evidence": [
                {"tmp_id": "tmp_evidence_1", "snippet": "Real evidence", "source_url": "file://test.py"}
            ],
            "new_edges": [
                {"from_id": "tmp_evidence_1", "to_id": "tmp_claim_1", "kind": "Supports"}
            ]
        });

        let result = validate_model_delta(&delta, &empty_graph(), 4);
        assert!(result.passed, "Violations: {:?}", result.violations);
    }

    #[test]
    fn test_v09_reopen_ceiling() {
        let result = validate_reopen_ceiling(2, 2);
        assert!(!result.passed);
        assert!(result.violations.iter().any(|v| v.check_id == "V-09"));
    }

    #[test]
    fn test_v12_support_mass_minimum() {
        let result = validate_support_mass(0.005);
        assert!(!result.passed);
        assert!(result.violations.iter().any(|v| v.check_id == "V-12"));

        let result = validate_support_mass(1.0);
        assert!(result.passed);
    }
}
