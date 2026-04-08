//! Phase State Machine (§4)
//!
//! The Harness enforces phase sequencing.  Any model attempt to shortcut
//! phases is rejected by the Validator.  Phase transitions are logged to
//! the audit trail with timestamps and round_id.
//!
//! Routing: 1 → 2 → 3 → 4 → 5 → 6 → 7 → route
//! From 7: Continue → 2, Stall → 8 → 2, Converge → 9 → 10.

use crate::graph::SageGraph;
use crate::schema::*;
use crate::scorer;
use serde::{Deserialize, Serialize};

/// Phase identifiers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Decompose,   // 1
    Score,       // 2
    Execute,     // 3
    Compress,    // 4
    Audit,       // 5
    Skeptic,     // 6
    Monitor,     // 7
    Reframe,     // 8
    Bank,        // 9
    FinalVerify, // 10
}

impl Phase {
    pub fn number(&self) -> u32 {
        match self {
            Phase::Decompose => 1,
            Phase::Score => 2,
            Phase::Execute => 3,
            Phase::Compress => 4,
            Phase::Audit => 5,
            Phase::Skeptic => 6,
            Phase::Monitor => 7,
            Phase::Reframe => 8,
            Phase::Bank => 9,
            Phase::FinalVerify => 10,
        }
    }
}

/// Routing decision from Phase 7 Monitor
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MonitorRouting {
    Continue, // → Phase 2
    Stall,    // → Phase 8
    Converge, // → Phase 9
}

/// Configuration thresholds for phase decisions
#[derive(Debug, Clone)]
pub struct PhaseConfig {
    pub eig_min: f32,
    pub eig_delta_stall: f32,
    pub stall_count_threshold: u32,
    pub convergence_sa: f32,
    pub diversity_min: f32,
    pub support_mass_min: f32,
    pub total_budget: u32,
    pub max_rounds: u32,
    pub redundancy_pct: f32,
}

impl Default for PhaseConfig {
    fn default() -> Self {
        Self {
            eig_min: 0.05,
            eig_delta_stall: 0.01,
            stall_count_threshold: 2,
            convergence_sa: 0.85,
            diversity_min: 0.50,
            support_mass_min: 1.00,
            total_budget: 200_000,
            max_rounds: 12,
            redundancy_pct: 0.60,
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 2: Score (pure Harness, no LLM)
// ---------------------------------------------------------------------------

/// Phase 2: Compute EIG for each Open HypothesisNode, rank, allocate budget
pub fn phase_score(graph: &mut SageGraph, round_budget: u32) -> Vec<(String, f32, u32)> {
    let hypotheses = graph.hypotheses();

    // Collect Open hypotheses with their EIG
    let mut scored: Vec<(String, f32, bool)> = hypotheses
        .iter()
        .filter(|h| h.status == HypothesisStatus::Open || h.status == HypothesisStatus::Active)
        .map(|h| {
            let has_failing_test = h.test_id.is_some(); // Code agent: assume test-derived = potentially failing
            let e = scorer::eig_code(h, has_failing_test);
            (h.meta.id.clone(), e, has_failing_test)
        })
        .collect();

    // Sort descending by EIG
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Discard nodes where EIG < 0.05
    scored.retain(|(_, e, _)| *e >= 0.05);

    if scored.is_empty() {
        return Vec::new();
    }

    // Allocate budget
    let sum_eig: f32 = scored.iter().map(|(_, e, _)| e).sum();
    let allocations: Vec<(String, f32, u32)> = scored
        .iter()
        .map(|(id, e, _)| {
            let budget = if sum_eig > 0.0 {
                (round_budget as f32 * e / sum_eig) as u32
            } else {
                round_budget / scored.len() as u32
            };
            (id.clone(), *e, budget.max(1))
        })
        .collect();

    // Mark selected branches as Active
    for (id, _, _) in &allocations {
        if let Some(NodeKind::Hypothesis(h)) = graph.get_node_mut(id) {
            h.status = HypothesisStatus::Active;
            h.meta.touch();
        }
    }

    allocations
}

// ---------------------------------------------------------------------------
// Phase 5: Audit (pure Harness, no LLM)
// ---------------------------------------------------------------------------

/// Phase 5: Deterministic contradiction detection and structural analysis
pub fn phase_audit(graph: &mut SageGraph) -> AuditSummary {
    let mut summary = AuditSummary::default();

    // Clone claim IDs to avoid borrow issues
    let claim_ids: Vec<String> = graph.claims().iter().map(|c| c.meta.id.clone()).collect();

    // Direct contradiction check: traverse all Refutes edges
    for claim_id in &claim_ids {
        let refutes_count = graph.count_active_refutes(claim_id);
        if refutes_count > 0 {
            if let Some(NodeKind::Claim(c)) = graph.get_node_mut(claim_id) {
                if c.status != ClaimStatus::Disputed {
                    c.status = ClaimStatus::Disputed;
                    c.meta.belief_score = 0.0;
                    c.meta.touch();
                    summary.disputed_count += 1;
                }
            }
        }
    }

    // Stale cascade: for each Refuted HypothesisNode, set is_stale on descendants
    let refuted_ids: Vec<String> = graph
        .hypotheses()
        .iter()
        .filter(|h| h.status == HypothesisStatus::Refuted)
        .map(|h| h.meta.id.clone())
        .collect();

    for hyp_id in &refuted_ids {
        cascade_stale(graph, hyp_id);
        summary.stale_count += 1;
    }

    // Inference chain validation: DerivedFrom(C1 → C2) where C1 is Unverified
    // means C2's derivation is unsupported — mark C2 Disputed.
    let claim_ids_for_inference: Vec<String> =
        graph.claims().iter().map(|c| c.meta.id.clone()).collect();

    for c1_id in &claim_ids_for_inference {
        let is_unverified = match graph.get_node(c1_id) {
            Some(NodeKind::Claim(c)) => c.status == ClaimStatus::Unverified,
            _ => false,
        };
        if !is_unverified {
            continue;
        }
        // Collect DerivedFrom targets (collect IDs first to release the borrow)
        let derived_targets: Vec<String> = graph
            .outgoing_neighbors(c1_id)
            .into_iter()
            .filter_map(|(node, edge)| {
                if edge == EdgeKind::DerivedFrom {
                    Some(node.id().to_string())
                } else {
                    None
                }
            })
            .collect();

        for c2_id in derived_targets {
            if let Some(NodeKind::Claim(c2)) = graph.get_node_mut(&c2_id) {
                if c2.status != ClaimStatus::Disputed {
                    c2.status = ClaimStatus::Disputed;
                    c2.meta.belief_score = 0.0;
                    c2.meta.touch();
                    summary.disputed_count += 1;
                }
            }
        }
    }

    summary.contradiction_count = summary.disputed_count;
    summary
}

/// Cascade is_stale to all descendant nodes of a given node
fn cascade_stale(graph: &mut SageGraph, node_id: &str) {
    // Get all outgoing neighbors (children)
    let child_ids: Vec<String> = graph
        .outgoing_neighbors(node_id)
        .iter()
        .map(|(n, _)| n.id().to_string())
        .collect();

    for child_id in child_ids {
        if let Some(node) = graph.get_node_mut(&child_id) {
            if !node.meta().is_stale {
                node.meta_mut().is_stale = true;
                node.meta_mut().touch();
                // Recurse to descendants
                cascade_stale(graph, &child_id);
            }
        }
    }
}

#[derive(Debug, Default)]
pub struct AuditSummary {
    pub contradiction_count: u32,
    pub stale_count: u32,
    pub disputed_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkepticDecision {
    pub claim_id: String,
    pub result: String,
    pub confidence: f32,
    pub critique: Option<String>,
}

/// Phase 6: apply Skeptic outcomes to Supported claims.
/// confidence is interpreted as p_fail per spec.
pub fn phase_skeptic_apply(
    graph: &mut SageGraph,
    config: &PhaseConfig,
    decisions: &[SkepticDecision],
) -> (u32, u32) {
    let mut promoted = 0u32;
    let mut disputed = 0u32;

    for d in decisions {
        let skeptic_result = match d.result.as_str() {
            "Pass" => scorer::SkepticResult::Pass,
            "Fail" => scorer::SkepticResult::Fail,
            _ => scorer::SkepticResult::Inconclusive,
        };

        if let Some(NodeKind::Claim(claim)) = graph.get_node(&d.claim_id) {
            if claim.status != ClaimStatus::Supported {
                continue;
            }
        } else {
            continue;
        }

        let can_promote = scorer::can_promote(
            &d.claim_id,
            skeptic_result,
            graph,
            config.diversity_min,
            config.support_mass_min,
        );

        if can_promote {
            if let Some(NodeKind::Claim(c)) = graph.get_node_mut(&d.claim_id) {
                c.status = ClaimStatus::Verified;
                let skeptic_confidence_correct = (1.0 - d.confidence).clamp(0.0, 1.0);
                c.meta.belief_score =
                    scorer::bayesian_update(c.meta.belief_score, skeptic_confidence_correct);
                c.meta.source_entropy = 1.0 - c.meta.belief_score;
                c.meta.touch();
                promoted += 1;
            }
        } else if let Some(NodeKind::Claim(c)) = graph.get_node_mut(&d.claim_id) {
            c.status = ClaimStatus::Disputed;
            c.meta.belief_score = 0.0;
            c.meta.source_entropy = 1.0;
            c.meta.touch();
            disputed += 1;
        }
    }

    (promoted, disputed)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkepticDecision {
    pub claim_id: String,
    pub result: String,
    pub confidence: f32,
    pub critique: Option<String>,
}

/// Phase 6: apply Skeptic outcomes to Supported claims.
/// confidence is interpreted as p_fail per spec.
pub fn phase_skeptic_apply(
    graph: &mut SageGraph,
    config: &PhaseConfig,
    decisions: &[SkepticDecision],
) -> (u32, u32) {
    let mut promoted = 0u32;
    let mut disputed = 0u32;

    for d in decisions {
        let skeptic_result = match d.result.as_str() {
            "Pass" => scorer::SkepticResult::Pass,
            "Fail" => scorer::SkepticResult::Fail,
            _ => scorer::SkepticResult::Inconclusive,
        };

        if let Some(NodeKind::Claim(claim)) = graph.get_node(&d.claim_id) {
            if claim.status != ClaimStatus::Supported {
                continue;
            }
        } else {
            continue;
        }

        let can_promote = scorer::can_promote(
            &d.claim_id,
            skeptic_result,
            graph,
            config.diversity_min,
            config.support_mass_min,
        );

        if can_promote {
            if let Some(NodeKind::Claim(c)) = graph.get_node_mut(&d.claim_id) {
                c.status = ClaimStatus::Verified;
                let skeptic_confidence_correct = (1.0 - d.confidence).clamp(0.0, 1.0);
                c.meta.belief_score =
                    scorer::bayesian_update(c.meta.belief_score, skeptic_confidence_correct);
                c.meta.source_entropy = 1.0 - c.meta.belief_score;
                c.meta.touch();
                promoted += 1;
            }
        } else if let Some(NodeKind::Claim(c)) = graph.get_node_mut(&d.claim_id) {
            c.status = ClaimStatus::Disputed;
            c.meta.belief_score = 0.0;
            c.meta.source_entropy = 1.0;
            c.meta.touch();
            disputed += 1;
        }
    }

    (promoted, disputed)
}

// ---------------------------------------------------------------------------
// Phase 7: Monitor
// ---------------------------------------------------------------------------

/// Phase 7: Evaluate meta-state and determine routing.
pub fn phase_monitor(
    graph: &mut SageGraph,
    round_id: u32,
    tokens_consumed: u32,
    all_tests_pass: bool,
    config: &PhaseConfig,
    prev_eigs: &[(String, f32)],
) -> MonitorRouting {
    // Code agent convergence: all tests pass (§7.4)
    if all_tests_pass {
        return MonitorRouting::Converge;
    }

    // Budget check: force convergence if over budget
    if tokens_consumed >= config.total_budget {
        return MonitorRouting::Converge;
    }

    // Round limit
    if round_id >= config.max_rounds {
        return MonitorRouting::Converge;
    }

    // Convergence check: all hypotheses Resolved or Refuted
    //   + Stall check: if EIG delta < threshold for active hypotheses
    // Collect all needed data first (immutable borrow), then mutate.
    let hyp_data: Vec<(String, HypothesisStatus, f32)> = graph
        .hypotheses()
        .iter()
        .map(|h| (h.meta.id.clone(), h.status, scorer::eig(h)))
        .collect();

    let all_terminal = hyp_data
        .iter()
        .all(|(_, s, _)| matches!(s, HypothesisStatus::Resolved | HypothesisStatus::Refuted));
    if all_terminal && !hyp_data.is_empty() {
        return MonitorRouting::Converge;
    }

    // Stall check
    let mut any_stalled = false;
    let active_data: Vec<(String, f32)> = hyp_data
        .iter()
        .filter(|(_, s, _)| *s == HypothesisStatus::Active)
        .map(|(id, _, eig)| (id.clone(), *eig))
        .collect();

    for (hyp_id, current_eig) in &active_data {
        let prev_eig = prev_eigs
            .iter()
            .find(|(id, _)| id == hyp_id)
            .map(|(_, e)| *e)
            .unwrap_or(0.0);

        let eig_delta = (current_eig - prev_eig).abs();

        if *current_eig < config.eig_min || eig_delta < config.eig_delta_stall {
            if let Some(NodeKind::Hypothesis(h)) = graph.get_node_mut(hyp_id) {
                h.stall_count += 1;
                if h.stall_count >= config.stall_count_threshold {
                    h.status = HypothesisStatus::Stalled;
                    h.meta.touch();
                    any_stalled = true;
                }
            }
        }
    }

    if any_stalled {
        return MonitorRouting::Stall;
    }

    MonitorRouting::Continue
}

// ---------------------------------------------------------------------------
// Phase 8: Reframe
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReframeClaimSeed {
    pub claim_id: String,
    pub text: String,
    pub status: ClaimStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReframeSeed {
    pub stalled_hypotheses: Vec<String>,
    pub refuted_hypotheses: Vec<String>,
    pub supported_claims: Vec<ReframeClaimSeed>,
    pub unverified_claims: Vec<ReframeClaimSeed>,
    pub disputed_claims: Vec<ReframeClaimSeed>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReframeHypothesisProposal {
    pub text: String,
    pub priority: f32,
    pub test_id: Option<String>,
}

/// Deterministic Phase 8 preparation:
/// - prune stalled hypotheses (mark Refuted + stale cascade)
/// - extract reframe seed claims for the Worker
pub fn phase_reframe_prepare(graph: &mut SageGraph) -> ReframeSeed {
    let stalled_ids: Vec<String> = graph
        .hypotheses()
        .iter()
        .filter(|h| h.status == HypothesisStatus::Stalled)
        .map(|h| h.meta.id.clone())
        .collect();

    let refuted_hypotheses: Vec<String> = graph
        .hypotheses()
        .iter()
        .filter(|h| h.status == HypothesisStatus::Refuted)
        .map(|h| h.text.clone())
        .collect();

    for hyp_id in &stalled_ids {
        if let Some(NodeKind::Hypothesis(h)) = graph.get_node_mut(&hyp_id) {
            h.status = HypothesisStatus::Refuted;
            h.meta.touch();
        }
        cascade_stale(graph, hyp_id);
    }

    let mut supported_claims = Vec::new();
    let mut unverified_claims = Vec::new();
    let mut disputed_claims = Vec::new();

    for c in graph.claims() {
        let item = ReframeClaimSeed {
            claim_id: c.meta.id.clone(),
            text: c.text.clone(),
            status: c.status,
        };
        match c.status {
            ClaimStatus::Supported => supported_claims.push(item),
            ClaimStatus::Unverified => unverified_claims.push(item),
            ClaimStatus::Disputed => {
                if graph.count_active_refutes(&c.meta.id) > 0 {
                    disputed_claims.push(item);
                }
            }
            ClaimStatus::Verified => {}
        }
    }

    ReframeSeed {
        stalled_hypotheses: stalled_ids,
        refuted_hypotheses,
        supported_claims,
        unverified_claims,
        disputed_claims,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReframeInsertResult {
    pub inserted: u32,
    pub rejected_duplicate_or_empty: u32,
}

/// Deterministically inserts reframe proposals while preventing exact duplicates.
/// Duplicate check is case-insensitive against all existing hypothesis texts.
pub fn phase_reframe_insert(
    graph: &mut SageGraph,
    proposals: &[ReframeHypothesisProposal],
) -> ReframeInsertResult {
    let mut existing: std::collections::HashSet<String> = graph
        .hypotheses()
        .iter()
        .map(|h| h.text.trim().to_lowercase())
        .collect();
    let round_id = graph.current_round;

    let mut inserted = 0u32;
    let mut rejected = 0u32;

    for p in proposals {
        let text_norm = p.text.trim().to_lowercase();
        if text_norm.is_empty() || existing.contains(&text_norm) {
            rejected += 1;
            continue;
        }
        let node = NodeKind::Hypothesis(HypothesisNode {
            meta: NodeMeta::new(round_id, 0.0),
            text: p.text.trim().to_string(),
            status: HypothesisStatus::Open,
            priority: p.priority.clamp(0.0, 1.0),
            stall_count: 0,
            max_reopen: 2,
            reopen_count: 0,
            test_id: p.test_id.clone(),
        });
        graph.add_node(node);
        existing.insert(text_norm);
        inserted += 1;
    }

    ReframeInsertResult {
        inserted,
        rejected_duplicate_or_empty: rejected,
    }
}

/// Final Verify response coherence check.
pub fn validate_final_verify_decision(result: &str, p_fail: f32, delta: f32) -> Result<(), String> {
    if !(0.0..=1.0).contains(&p_fail) {
        return Err(format!("p_fail out of range: {}", p_fail));
    }
    match result {
        "Pass" if p_fail < delta => Ok(()),
        "Fail" if p_fail >= delta => Ok(()),
        "Pass" => Err(format!(
            "Incoherent Final Verify: Pass with p_fail {} >= delta {}",
            p_fail, delta
        )),
        "Fail" => Err(format!(
            "Incoherent Final Verify: Fail with p_fail {} < delta {}",
            p_fail, delta
        )),
        other => Err(format!("Unknown Final Verify result: {}", other)),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinalVerifyDecision {
    pub candidate_id: String,
    pub result: String,
    pub confidence: f32, // p_fail
    pub contested_claim_ids: Vec<String>,
    pub critique: String,
    pub delta: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinalVerifyApplyResult {
    pub outcome: String, // pass | reopen | emit_with_uncertainty
    pub contradiction_id: Option<String>,
    pub reopen_count: u8,
    pub max_reopen_count: u8,
    pub reopened_hypothesis_ids: Vec<String>,
}

/// Apply Final Verify decision deterministically in the harness.
/// - Validates coherence constraints for result/p_fail.
/// - On Fail, creates ContradictionNode + Invalidates edge.
/// - Handles reopen ceiling and candidate reopen_count updates.
pub fn phase_final_verify_apply(
    graph: &mut SageGraph,
    decision: &FinalVerifyDecision,
) -> Result<FinalVerifyApplyResult, String> {
    validate_final_verify_decision(&decision.result, decision.confidence, decision.delta)?;

    let Some(NodeKind::CandidateAnswer(candidate_view)) = graph.get_node(&decision.candidate_id) else {
        return Err(format!(
            "CandidateAnswer not found for candidate_id={}",
            decision.candidate_id
        ));
    };
    let current_reopen = candidate_view.reopen_count;
    let max_reopen = candidate_view.max_reopen_count;

    if decision.result == "Pass" {
        return Ok(FinalVerifyApplyResult {
            outcome: "pass".to_string(),
            contradiction_id: None,
            reopen_count: current_reopen,
            max_reopen_count: max_reopen,
            reopened_hypothesis_ids: vec![],
        });
    }

    if current_reopen >= max_reopen {
        return Ok(FinalVerifyApplyResult {
            outcome: "emit_with_uncertainty".to_string(),
            contradiction_id: None,
            reopen_count: current_reopen,
            max_reopen_count: max_reopen,
            reopened_hypothesis_ids: vec![],
        });
    }

    let round_id = graph.current_round;
    let contradiction = ContradictionNode {
        meta: NodeMeta::new(round_id, 0.0),
        critique: decision.critique.chars().take(300).collect(),
        source: ContradictionSource::FinalVerifySkeptic,
        contested_claims: decision.contested_claim_ids.clone(),
        skeptic_p_fail: decision.confidence.clamp(0.0, 1.0),
    };
    let contradiction_id = contradiction.meta.id.clone();
    graph.add_node(NodeKind::Contradiction(contradiction));
    graph.add_edge(&contradiction_id, &decision.candidate_id, EdgeKind::Invalidates)?;

    if let Some(NodeKind::CandidateAnswer(c)) = graph.get_node_mut(&decision.candidate_id) {
        c.reopen_count = c.reopen_count.saturating_add(1);
        c.meta.belief_score = 0.0;
        c.meta.source_entropy = 1.0;
        c.meta.touch();
    }

    // Re-open parent hypothesis if we can find a Proposes(Hypothesis -> Candidate) edge.
    let proposer_hyp_ids: Vec<String> = graph
        .incoming_neighbors(&decision.candidate_id)
        .into_iter()
        .filter_map(|(node, edge)| match (node, edge) {
            (NodeKind::Hypothesis(_), EdgeKind::Proposes) => Some(node.id().to_string()),
            _ => None,
        })
        .collect();

    for hyp_id in &proposer_hyp_ids {
        if let Some(NodeKind::Hypothesis(h)) = graph.get_node_mut(&hyp_id) {
            h.status = HypothesisStatus::Open;
            h.reopen_count = h.reopen_count.saturating_add(1);
            h.meta.touch();
        }
    }

    let updated_reopen = match graph.get_node(&decision.candidate_id) {
        Some(NodeKind::CandidateAnswer(c)) => c.reopen_count,
        _ => current_reopen,
    };

    Ok(FinalVerifyApplyResult {
        outcome: "reopen".to_string(),
        contradiction_id: Some(contradiction_id),
        reopen_count: updated_reopen,
        max_reopen_count: max_reopen,
        reopened_hypothesis_ids: proposer_hyp_ids,
    })
}

// ---------------------------------------------------------------------------
// Phase 9: Bank
// ---------------------------------------------------------------------------

/// Phase 9: Score all candidates and return them ranked by S_A descending.
/// Returns (candidate_id, s_a, contradiction_score) triples.
pub fn phase_bank(graph: &mut SageGraph) -> Vec<(String, f32, f32)> {
    let candidate_ids: Vec<String> = graph
        .candidates()
        .iter()
        .map(|c| c.meta.id.clone())
        .collect();

    let mut scored: Vec<(String, f32, f32)> = candidate_ids
        .iter()
        .map(|id| {
            let (s_a, cs) = match graph.get_node(id) {
                Some(NodeKind::CandidateAnswer(c)) => scorer::score_candidate(c, graph),
                _ => (0.0, 0.0),
            };
            (id.clone(), s_a, cs)
        })
        .collect();

    // Store contradiction_score on nodes
    for (id, _, cs) in &scored {
        if let Some(NodeKind::CandidateAnswer(c)) = graph.get_node_mut(id) {
            c.contradiction_score = *cs;
            c.meta.touch();
        }
    }

    // Rank by S_A descending
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    scored
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_graph_with_hypotheses() -> SageGraph {
        let mut g = SageGraph::new();

        let h1 = NodeKind::Hypothesis(HypothesisNode {
            meta: NodeMeta::new(0, 0.0),
            text: "High priority hypothesis".to_string(),
            status: HypothesisStatus::Open,
            priority: 1.0,
            stall_count: 0,
            max_reopen: 2,
            reopen_count: 0,
            test_id: None,
        });

        let h2 = NodeKind::Hypothesis(HypothesisNode {
            meta: NodeMeta::new(0, 0.0),
            text: "Low priority hypothesis".to_string(),
            status: HypothesisStatus::Open,
            priority: 0.3,
            stall_count: 0,
            max_reopen: 2,
            reopen_count: 0,
            test_id: None,
        });

        g.add_node(h1);
        g.add_node(h2);
        g
    }

    #[test]
    fn test_phase_score() {
        let mut g = make_graph_with_hypotheses();
        let allocations = phase_score(&mut g, 1000);

        assert_eq!(allocations.len(), 2);
        // First should be the high priority one
        assert!(allocations[0].1 > allocations[1].1);
        assert!(allocations[0].2 > allocations[1].2);
    }

    #[test]
    fn test_monitor_converge_all_tests_pass() {
        let mut g = make_graph_with_hypotheses();
        let config = PhaseConfig::default();
        let routing = phase_monitor(&mut g, 0, 0, true, &config, &[]);
        assert_eq!(routing, MonitorRouting::Converge);
    }

    #[test]
    fn test_monitor_converge_budget_exceeded() {
        let mut g = make_graph_with_hypotheses();
        let config = PhaseConfig::default();
        let routing = phase_monitor(&mut g, 0, 999_999, false, &config, &[]);
        assert_eq!(routing, MonitorRouting::Converge);
    }

    #[test]
    fn test_monitor_continue_normal() {
        let mut g = make_graph_with_hypotheses();
        let config = PhaseConfig::default();
        let routing = phase_monitor(&mut g, 0, 0, false, &config, &[]);
        assert_eq!(routing, MonitorRouting::Continue);
    }
}
