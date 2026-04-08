//! Controller Mathematics & Scoring Formulas (§5)
//!
//! All formulas are implemented in Rust.  No formula computation may be
//! delegated to the LLM.  All inputs are graph-resident fields set by
//! the Harness.

use crate::domain_auth::extract_root_domain;
use crate::graph::SageGraph;
use crate::schema::*;
use std::collections::{HashMap, HashSet};

/// Penalty weight for unresolved_gaps in S_A formula
const LAMBDA: f32 = 1.5;
/// Penalty weight for contradiction_score in S_A formula
const MU: f32 = 2.0;
/// Epsilon guard against division by zero
const EPSILON: f32 = 1e-6;

// ---------------------------------------------------------------------------
// §5.1 Expected Information Gain: EIG(h)
// ---------------------------------------------------------------------------

/// EIG for a HypothesisNode
pub fn eig(h: &HypothesisNode) -> f32 {
    (1.0 - h.meta.belief_score) * h.priority
}

/// Code agent variant: boost EIG for hypotheses tied to failing tests
pub fn eig_code(h: &HypothesisNode, has_failing_test: bool) -> f32 {
    let base = (1.0 - h.meta.belief_score) * h.priority;
    if has_failing_test {
        base * 1.5
    } else {
        base
    }
}

/// Budget allocation across all Active branches in a round (§5.1)
pub fn allocate_budget(
    nodes: &[&HypothesisNode],
    total_round_budget: u32,
) -> HashMap<String, u32> {
    let eigs: Vec<f32> = nodes.iter().map(|h| eig(h)).collect();
    let sum_eig: f32 = eigs.iter().sum();

    if sum_eig < EPSILON {
        // All EIG near zero — distribute evenly
        let per_branch = total_round_budget / nodes.len().max(1) as u32;
        return nodes
            .iter()
            .map(|h| (h.meta.id.clone(), per_branch))
            .collect();
    }

    nodes
        .iter()
        .zip(eigs.iter())
        .map(|(h, e)| {
            let budget = (total_round_budget as f32 * e / sum_eig) as u32;
            (h.meta.id.clone(), budget.max(1)) // at least 1 token
        })
        .collect()
}

// ---------------------------------------------------------------------------
// §5.2 Diversity Index: D(S)
// ---------------------------------------------------------------------------

/// D(S): fraction of unique root domains across all SourceNodes
/// supporting a given ClaimNode.  Measures source independence.
pub fn diversity_index(claim_id: &str, graph: &SageGraph) -> f32 {
    let sources = graph.get_evidence_sources(claim_id);
    if sources.is_empty() {
        return 0.0;
    }

    let unique_domains: HashSet<String> = sources
        .iter()
        .map(|s| extract_root_domain(&s.url))
        .collect();

    unique_domains.len() as f32 / sources.len() as f32
}

// ---------------------------------------------------------------------------
// §5.3 Global Candidate Score: S_A
// ---------------------------------------------------------------------------

/// S_A: score for a CandidateAnswerNode.
/// Higher is better.  Penalized by unresolved gaps, contradictions,
/// and contradiction_score.
pub fn score_candidate(candidate: &CandidateAnswerNode, graph: &SageGraph) -> (f32, f32) {
    let support_claims: Vec<&ClaimNode> = candidate
        .supporting_claims
        .iter()
        .filter_map(|id| match graph.get_node(id) {
            Some(NodeKind::Claim(c)) => Some(c),
            _ => None,
        })
        .collect();

    // W_support: sum of belief_scores of all Verified supporting claims
    let w_support: f32 = support_claims
        .iter()
        .filter(|c| c.status == ClaimStatus::Verified)
        .map(|c| c.meta.belief_score)
        .sum();

    // D: diversity index averaged across all supporting claims
    let d: f32 = if support_claims.is_empty() {
        0.0
    } else {
        support_claims
            .iter()
            .map(|c| diversity_index(&c.meta.id, graph))
            .sum::<f32>()
            / support_claims.len() as f32
    };

    // W_refute: sum of belief_scores of all Disputed claims in lineage
    let w_refute: f32 = support_claims
        .iter()
        .filter(|c| c.status == ClaimStatus::Disputed)
        .map(|c| c.meta.belief_score)
        .sum();

    // Unresolved_Gaps: count of Disputed or Unverified claims
    let unresolved_gaps: u32 = support_claims
        .iter()
        .filter(|c| matches!(c.status, ClaimStatus::Disputed | ClaimStatus::Unverified))
        .count() as u32;

    // contradiction_score (§5.4)
    let cs = compute_contradiction_score(&support_claims);

    let s_a = (w_support * d) / (EPSILON + w_refute + LAMBDA * unresolved_gaps as f32 + MU * cs);

    (s_a, cs)
}

// ---------------------------------------------------------------------------
// §5.4 Contradiction Score
// ---------------------------------------------------------------------------

/// contradiction_score for a CandidateAnswerNode.
/// Range: [0.0, 1.0].  0.0 = no disputed claims.  1.0 = all disputed.
pub fn compute_contradiction_score(support_claims: &[&ClaimNode]) -> f32 {
    if support_claims.is_empty() {
        return 0.0;
    }

    let disputed_mass: f32 = support_claims
        .iter()
        .filter(|c| c.status == ClaimStatus::Disputed)
        .map(|c| c.meta.belief_score)
        .sum();

    let total_mass: f32 = support_claims.iter().map(|c| c.meta.belief_score).sum();

    disputed_mass / (total_mass + EPSILON)
}

// ---------------------------------------------------------------------------
// §5.5 / §5.6 Belief Score Updates
// ---------------------------------------------------------------------------

/// Compute support_mass for a ClaimNode (§5.5)
pub fn compute_support_mass(claim_id: &str, graph: &SageGraph) -> f32 {
    graph
        .get_supporting_evidence(claim_id)
        .iter()
        .map(|e| e.meta.belief_score)
        .sum()
}

/// Update belief_score from QualityTier of linked EvidenceNodes (§5.6)
/// Called after Phase 4 (Compress)
pub fn update_belief_from_evidence(claim_id: &str, graph: &mut SageGraph) {
    let best_tier_score = graph
        .get_supporting_evidence(claim_id)
        .iter()
        .map(|e| e.quality.initial_belief_score())
        .fold(0.0f32, f32::max);

    if let Some(NodeKind::Claim(claim)) = graph.get_node_mut(claim_id) {
        claim.meta.belief_score = best_tier_score;
        claim.meta.source_entropy = 1.0 - best_tier_score;
        claim.meta.touch();
    }
}

/// Bayesian update after Skeptic Pass (§5.6)
pub fn bayesian_update(belief_score: f32, skeptic_confidence: f32) -> f32 {
    (belief_score * skeptic_confidence).min(1.0).max(0.0)
}

// ---------------------------------------------------------------------------
// §3.2 Promotion Predicate
// ---------------------------------------------------------------------------

/// A ClaimNode advances to Verified if and only if all four conditions hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkepticResult {
    Pass,
    Fail,
    Inconclusive,
}

pub fn can_promote(
    claim_id: &str,
    skeptic_result: SkepticResult,
    graph: &SageGraph,
    diversity_min: f32,
    support_mass_min: f32,
) -> bool {
    let d = diversity_index(claim_id, graph);
    let support_mass = compute_support_mass(claim_id, graph);
    let active_refutes = graph.count_active_refutes(claim_id);

    skeptic_result == SkepticResult::Pass
        && d >= diversity_min
        && active_refutes == 0
        && support_mass >= support_mass_min
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_hypothesis(text: &str, priority: f32, belief: f32) -> HypothesisNode {
        HypothesisNode {
            meta: NodeMeta::new(0, belief),
            text: text.to_string(),
            status: HypothesisStatus::Open,
            priority,
            stall_count: 0,
            max_reopen: 2,
            reopen_count: 0,
            test_id: None,
        }
    }

    #[test]
    fn test_eig_basic() {
        let h = make_hypothesis("test", 0.8, 0.0);
        assert!((eig(&h) - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_eig_resolved_hypothesis() {
        let h = make_hypothesis("test", 0.8, 1.0);
        assert!((eig(&h) - 0.0).abs() < 0.001); // fully believed -> 0 EIG
    }

    #[test]
    fn test_eig_code_boost() {
        let h = make_hypothesis("test", 0.8, 0.0);
        let base = eig(&h);
        let boosted = eig_code(&h, true);
        assert!((boosted - base * 1.5).abs() < 0.001);
    }

    #[test]
    fn test_budget_allocation() {
        let h1 = make_hypothesis("high priority", 1.0, 0.0);
        let h2 = make_hypothesis("low priority", 0.2, 0.0);
        let nodes: Vec<&HypothesisNode> = vec![&h1, &h2];

        let budgets = allocate_budget(&nodes, 1000);
        let b1 = budgets[&h1.meta.id];
        let b2 = budgets[&h2.meta.id];

        // h1 should get ~833 tokens, h2 ~167 tokens
        assert!(b1 > b2);
        assert!(b1 > 700);
        assert!(b2 > 100);
    }

    #[test]
    fn test_contradiction_score() {
        let verified = ClaimNode {
            meta: NodeMeta::new(0, 0.9),
            text: "verified".to_string(),
            evidence_ids: vec![],
            status: ClaimStatus::Verified,
        };
        let disputed = ClaimNode {
            meta: NodeMeta::new(0, 0.7),
            text: "disputed".to_string(),
            evidence_ids: vec![],
            status: ClaimStatus::Disputed,
        };

        let claims: Vec<&ClaimNode> = vec![&verified, &disputed];
        let cs = compute_contradiction_score(&claims);

        // 0.7 / (0.9 + 0.7 + epsilon) ~ 0.437
        assert!(cs > 0.4 && cs < 0.5, "contradiction_score = {}", cs);
    }

    #[test]
    fn test_bayesian_update() {
        let result = bayesian_update(0.7, 0.9);
        assert!((result - 0.63).abs() < 0.001);
    }
}
