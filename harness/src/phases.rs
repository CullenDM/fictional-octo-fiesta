//! Phase State Machine (§4)
//!
//! The Harness enforces phase sequencing.  Any model attempt to shortcut
//! phases is rejected by the Validator.  Phase transitions are logged to
//! the audit trail with timestamps and round_id.
//!
//! Stage 1 routing: 1 → 2 → 3 → 4 → 5 → [stub 6] → 7 → route
//! From 7: Continue → 2, Stall → 8 (stub), Converge → 9 → [stub 10] → emit

use crate::graph::SageGraph;
use crate::schema::*;
use crate::scorer;

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
    let claim_ids: Vec<String> = graph
        .claims()
        .iter()
        .map(|c| c.meta.id.clone())
        .collect();

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

// ---------------------------------------------------------------------------
// Phase 6: Skeptic STUB (Stage 1)
// ---------------------------------------------------------------------------

/// Stage 1 stub: auto-promote all Supported claims to Verified.
/// In Stage 2 this is replaced with real adversarial evaluation.
pub fn phase_skeptic_stub(graph: &mut SageGraph, config: &PhaseConfig) {
    let supported_ids: Vec<String> = graph
        .claims()
        .iter()
        .filter(|c| c.status == ClaimStatus::Supported)
        .map(|c| c.meta.id.clone())
        .collect();

    for claim_id in supported_ids {
        // Check promotion predicate with stub skeptic (always Pass)
        let can_promote = scorer::can_promote(
            &claim_id,
            scorer::SkepticResult::Pass,
            graph,
            config.diversity_min,
            config.support_mass_min,
        );

        if can_promote {
            if let Some(NodeKind::Claim(c)) = graph.get_node_mut(&claim_id) {
                c.status = ClaimStatus::Verified;
                // Bayesian update with stub confidence (§5.6)
                c.meta.belief_score = scorer::bayesian_update(c.meta.belief_score, 0.9);
                c.meta.source_entropy = 1.0 - c.meta.belief_score;
                c.meta.touch();
            }
        }
        // Claims that don't meet the promotion predicate stay Supported
    }
}

// ---------------------------------------------------------------------------
// Phase 7: Monitor (simplified for Stage 1)
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

    let all_terminal = hyp_data.iter().all(|(_, s, _)| {
        matches!(s, HypothesisStatus::Resolved | HypothesisStatus::Refuted)
    });
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
// Phase 8: Reframe STUB (Stage 1)
// ---------------------------------------------------------------------------

/// Stage 1 stub: just mark stalled hypotheses as Refuted and cascade stale.
/// In Stage 2 this generates new hypotheses via the Worker.
pub fn phase_reframe_stub(graph: &mut SageGraph) {
    let stalled_ids: Vec<String> = graph
        .hypotheses()
        .iter()
        .filter(|h| h.status == HypothesisStatus::Stalled)
        .map(|h| h.meta.id.clone())
        .collect();

    for hyp_id in stalled_ids {
        if let Some(NodeKind::Hypothesis(h)) = graph.get_node_mut(&hyp_id) {
            h.status = HypothesisStatus::Refuted;
            h.meta.touch();
        }
        cascade_stale(graph, &hyp_id);
    }
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

// ---------------------------------------------------------------------------
// Phase 10: Final Verify STUB (Stage 1)
// ---------------------------------------------------------------------------

/// Stage 1 stub: auto-pass.
/// Returns true (pass) always.  In Stage 2 this runs the Skeptic.
pub fn phase_final_verify_stub() -> bool {
    true
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
