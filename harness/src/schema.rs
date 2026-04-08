//! GREAT SAGE Data Schemas (§2)
//!
//! Every node in the system strictly adheres to these schemas.
//! The Harness Validator rejects any graph mutation that violates them.
//! All UUIDs are v4. All timestamps are RFC 3339 UTC.

use chrono::{DateTime, Utc};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// §2.1 NodeMeta Header — required on every node type
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct NodeMeta {
    #[pyo3(get)]
    pub id: String, // UUID v4, immutable after creation
    #[pyo3(get)]
    pub created_at: String, // RFC 3339 UTC, immutable after creation
    #[pyo3(get, set)]
    pub last_updated: String, // set by Harness on every mutation
    #[pyo3(get, set)]
    pub round_id: u32, // round in which this node was created
    #[pyo3(get, set)]
    pub belief_score: f32, // [0.0, 1.0]; set by Harness scoring only
    #[pyo3(get, set)]
    pub source_entropy: f32, // local uncertainty = 1.0 - belief_score
    #[pyo3(get, set)]
    pub is_stale: bool, // true if parent hypothesis Refuted
}

impl NodeMeta {
    pub fn new(round_id: u32, belief_score: f32) -> Self {
        let now = Utc::now().to_rfc3339();
        Self {
            id: Uuid::new_v4().to_string(),
            created_at: now.clone(),
            last_updated: now,
            round_id,
            belief_score,
            source_entropy: 1.0 - belief_score,
            is_stale: false,
        }
    }

    pub fn uuid(&self) -> Uuid {
        Uuid::parse_str(&self.id).unwrap()
    }

    pub fn touch(&mut self) {
        self.last_updated = Utc::now().to_rfc3339();
    }
}

// ---------------------------------------------------------------------------
// §2.2 HypothesisNode
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[pyclass(eq)]
pub enum HypothesisStatus {
    Open,
    Active,
    Stalled,
    Resolved,
    Refuted,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct HypothesisNode {
    #[pyo3(get)]
    pub meta: NodeMeta,
    #[pyo3(get, set)]
    pub text: String, // falsifiable statement, 1-3 sentences
    #[pyo3(get, set)]
    pub status: HypothesisStatus,
    #[pyo3(get, set)]
    pub priority: f32, // [0.0, 1.0]; model-proposed, Harness-normalized
    #[pyo3(get, set)]
    pub stall_count: u32, // incremented by Harness; triggers Reframe at >= 2
    #[pyo3(get, set)]
    pub max_reopen: u8, // max times this node may reopen; default 2
    #[pyo3(get, set)]
    pub reopen_count: u8, // incremented by Harness on each Final Verify failure
    #[pyo3(get, set)]
    pub test_id: Option<String>, // Some(id) for test-derived hypotheses (§7.3)
}

// ---------------------------------------------------------------------------
// §2.3 ClaimNode
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[pyclass(eq)]
pub enum ClaimStatus {
    Unverified,
    Supported,
    Verified,
    Disputed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct ClaimNode {
    #[pyo3(get)]
    pub meta: NodeMeta,
    #[pyo3(get, set)]
    pub text: String,
    #[pyo3(get, set)]
    pub evidence_ids: Vec<String>, // must all resolve to EvidenceNode
    #[pyo3(get, set)]
    pub status: ClaimStatus,
}

// ---------------------------------------------------------------------------
// §2.4 EvidenceNode
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[pyclass(eq)]
pub enum QualityTier {
    TestVerified,  // confirmed by passing test(s); highest epistemic weight
    Primary,       // direct source, no corroboration required
    Corroborated,  // supported by >= 2 independent sources
    Inferred,      // derived; must carry inference chain in snippet
    Contradicted,  // active Refutes edge present; belief_score -> 0.0
}

impl QualityTier {
    /// Initial belief_score by tier (§2.4 table)
    pub fn initial_belief_score(&self) -> f32 {
        match self {
            QualityTier::TestVerified => 0.95,
            QualityTier::Primary => 0.70,
            QualityTier::Corroborated => 0.85,
            QualityTier::Inferred => 0.50,
            QualityTier::Contradicted => 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct EvidenceNode {
    #[pyo3(get)]
    pub meta: NodeMeta,
    #[pyo3(get, set)]
    pub snippet: String, // raw extracted text; pruned from prompt on Verified
    #[pyo3(get, set)]
    pub source_id: String, // must resolve to SourceNode
    #[pyo3(get, set)]
    pub quality: QualityTier,
}

// ---------------------------------------------------------------------------
// §2.5 SourceNode
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[pyclass(eq)]
pub enum SourceStatus {
    Unvisited,
    Visited,
    Unreachable,
    Stale,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct SourceNode {
    #[pyo3(get)]
    pub meta: NodeMeta,
    #[pyo3(get, set)]
    pub url: String, // file path or URL
    #[pyo3(get, set)]
    pub status: SourceStatus,
    #[pyo3(get, set)]
    pub domain_auth: f32, // from static YAML lookup (§2.5.1)
    #[pyo3(get, set)]
    pub reliability: f32, // domain_auth * recency_factor; Harness-computed
}

// ---------------------------------------------------------------------------
// §2.6 CandidateAnswerNode
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct CandidateAnswerNode {
    #[pyo3(get)]
    pub meta: NodeMeta,
    #[pyo3(get, set)]
    pub content: String, // proposed code diff, patch, or answer text
    #[pyo3(get, set)]
    pub supporting_claims: Vec<String>, // all must resolve to ClaimNode
    #[pyo3(get, set)]
    pub contradiction_score: f32, // Harness-computed (§5.4)
    #[pyo3(get, set)]
    pub max_reopen_count: u8, // default 2
    #[pyo3(get, set)]
    pub reopen_count: u8, // Harness-incremented
}

// ---------------------------------------------------------------------------
// §2.7 ContradictionNode
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[pyclass(eq)]
pub enum ContradictionSource {
    FinalVerifySkeptic,
    AuditRefutes,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct ContradictionNode {
    #[pyo3(get)]
    pub meta: NodeMeta,
    #[pyo3(get, set)]
    pub critique: String, // Skeptic output.critique; max 300 chars
    #[pyo3(get, set)]
    pub source: ContradictionSource,
    #[pyo3(get, set)]
    pub contested_claims: Vec<String>, // ClaimNode UUIDs the Skeptic contested
    #[pyo3(get, set)]
    pub skeptic_p_fail: f32, // p_fail from Skeptic response
}

// ---------------------------------------------------------------------------
// §2.8 Edge Kinds
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[pyclass(eq)]
pub enum EdgeKind {
    Supports,    // EvidenceNode -> ClaimNode, ClaimNode -> HypothesisNode
    Refutes,     // EvidenceNode -> ClaimNode (contradicting)
    DerivedFrom, // ClaimNode -> ClaimNode (inference chain)
    Cites,       // EvidenceNode -> SourceNode
    Proposes,    // HypothesisNode -> CandidateAnswerNode
    Invalidates, // ContradictionNode -> CandidateAnswerNode
}

// ---------------------------------------------------------------------------
// Unified node wrapper for graph storage
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeKind {
    Hypothesis(HypothesisNode),
    Claim(ClaimNode),
    Evidence(EvidenceNode),
    Source(SourceNode),
    CandidateAnswer(CandidateAnswerNode),
    Contradiction(ContradictionNode),
}

impl NodeKind {
    pub fn meta(&self) -> &NodeMeta {
        match self {
            NodeKind::Hypothesis(n) => &n.meta,
            NodeKind::Claim(n) => &n.meta,
            NodeKind::Evidence(n) => &n.meta,
            NodeKind::Source(n) => &n.meta,
            NodeKind::CandidateAnswer(n) => &n.meta,
            NodeKind::Contradiction(n) => &n.meta,
        }
    }

    pub fn meta_mut(&mut self) -> &mut NodeMeta {
        match self {
            NodeKind::Hypothesis(n) => &mut n.meta,
            NodeKind::Claim(n) => &mut n.meta,
            NodeKind::Evidence(n) => &mut n.meta,
            NodeKind::Source(n) => &mut n.meta,
            NodeKind::CandidateAnswer(n) => &mut n.meta,
            NodeKind::Contradiction(n) => &mut n.meta,
        }
    }

    pub fn id(&self) -> &str {
        &self.meta().id
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            NodeKind::Hypothesis(_) => "Hypothesis",
            NodeKind::Claim(_) => "Claim",
            NodeKind::Evidence(_) => "Evidence",
            NodeKind::Source(_) => "Source",
            NodeKind::CandidateAnswer(_) => "CandidateAnswer",
            NodeKind::Contradiction(_) => "Contradiction",
        }
    }
}
