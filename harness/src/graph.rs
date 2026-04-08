//! GREAT SAGE Knowledge Graph (§1.2, §1.3)
//!
//! Wraps petgraph::Graph<NodeKind, EdgeKind> behind Arc<RwLock<>> with
//! strict lock ordering as specified in §1.2.

use crate::schema::*;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// The knowledge graph.  The orchestrator holds this behind Arc<RwLock<>>
/// on the Python side; Rust functions receive &mut SageGraph when they
/// need write access.
#[derive(Debug, Serialize, Deserialize)]
pub struct SageGraph {
    pub graph: DiGraph<NodeKind, EdgeKind>,
    /// UUID string -> petgraph NodeIndex lookup for O(1) node access
    #[serde(skip)]
    id_index: HashMap<String, NodeIndex>,
    pub current_round: u32,
}

impl SageGraph {
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            id_index: HashMap::new(),
            current_round: 0,
        }
    }

    // -----------------------------------------------------------------------
    // Node operations
    // -----------------------------------------------------------------------

    pub fn add_node(&mut self, node: NodeKind) -> NodeIndex {
        let id = node.id().to_string();
        let idx = self.graph.add_node(node);
        self.id_index.insert(id, idx);
        idx
    }

    pub fn get_node(&self, id: &str) -> Option<&NodeKind> {
        self.id_index.get(id).map(|idx| &self.graph[*idx])
    }

    pub fn get_node_mut(&mut self, id: &str) -> Option<&mut NodeKind> {
        self.id_index.get(id).copied().map(|idx| &mut self.graph[idx])
    }

    pub fn get_index(&self, id: &str) -> Option<NodeIndex> {
        self.id_index.get(id).copied()
    }

    pub fn node_exists(&self, id: &str) -> bool {
        self.id_index.contains_key(id)
    }

    /// Rebuild the id_index from the graph (needed after deserialization)
    pub fn rebuild_index(&mut self) {
        self.id_index.clear();
        for idx in self.graph.node_indices() {
            let id = self.graph[idx].id().to_string();
            self.id_index.insert(id, idx);
        }
    }

    // -----------------------------------------------------------------------
    // Edge operations
    // -----------------------------------------------------------------------

    pub fn add_edge(&mut self, from_id: &str, to_id: &str, kind: EdgeKind) -> Result<(), String> {
        let from_idx = self
            .id_index
            .get(from_id)
            .ok_or_else(|| format!("from_id not found: {}", from_id))?;
        let to_idx = self
            .id_index
            .get(to_id)
            .ok_or_else(|| format!("to_id not found: {}", to_id))?;
        self.graph.add_edge(*from_idx, *to_idx, kind);
        Ok(())
    }

    /// Get all neighbors of a node in the given direction, with edge kinds
    pub fn neighbors_directed(
        &self,
        id: &str,
        direction: Direction,
    ) -> Vec<(&NodeKind, EdgeKind)> {
        let Some(idx) = self.id_index.get(id) else {
            return Vec::new();
        };
        let edges: Vec<_> = self
            .graph
            .edges_directed(*idx, direction)
            .map(|e| {
                let neighbor_idx = match direction {
                    Direction::Outgoing => e.target(),
                    Direction::Incoming => e.source(),
                };
                (&self.graph[neighbor_idx], *e.weight())
            })
            .collect();
        edges
    }

    /// Get all incoming neighbors (nodes pointing TO this node)
    pub fn incoming_neighbors(&self, id: &str) -> Vec<(&NodeKind, EdgeKind)> {
        self.neighbors_directed(id, Direction::Incoming)
    }

    /// Get all outgoing neighbors (nodes this node points TO)
    pub fn outgoing_neighbors(&self, id: &str) -> Vec<(&NodeKind, EdgeKind)> {
        self.neighbors_directed(id, Direction::Outgoing)
    }

    // -----------------------------------------------------------------------
    // Type-filtered queries
    // -----------------------------------------------------------------------

    pub fn hypotheses(&self) -> Vec<&HypothesisNode> {
        self.graph
            .node_weights()
            .filter_map(|n| match n {
                NodeKind::Hypothesis(h) => Some(h),
                _ => None,
            })
            .collect()
    }

    pub fn claims(&self) -> Vec<&ClaimNode> {
        self.graph
            .node_weights()
            .filter_map(|n| match n {
                NodeKind::Claim(c) => Some(c),
                _ => None,
            })
            .collect()
    }

    pub fn evidence_nodes(&self) -> Vec<&EvidenceNode> {
        self.graph
            .node_weights()
            .filter_map(|n| match n {
                NodeKind::Evidence(e) => Some(e),
                _ => None,
            })
            .collect()
    }

    pub fn sources(&self) -> Vec<&SourceNode> {
        self.graph
            .node_weights()
            .filter_map(|n| match n {
                NodeKind::Source(s) => Some(s),
                _ => None,
            })
            .collect()
    }

    pub fn candidates(&self) -> Vec<&CandidateAnswerNode> {
        self.graph
            .node_weights()
            .filter_map(|n| match n {
                NodeKind::CandidateAnswer(c) => Some(c),
                _ => None,
            })
            .collect()
    }

    /// Get evidence nodes linked to a claim via Supports edges
    pub fn get_supporting_evidence(&self, claim_id: &str) -> Vec<&EvidenceNode> {
        self.incoming_neighbors(claim_id)
            .into_iter()
            .filter_map(|(node, edge)| match (node, edge) {
                (NodeKind::Evidence(e), EdgeKind::Supports) => Some(e),
                _ => None,
            })
            .collect()
    }

    /// Get source nodes for evidence of a claim (traverse Evidence -> Source)
    pub fn get_evidence_sources(&self, claim_id: &str) -> Vec<&SourceNode> {
        let evidence = self.get_supporting_evidence(claim_id);
        evidence
            .iter()
            .filter_map(|e| match self.get_node(&e.source_id) {
                Some(NodeKind::Source(s)) => Some(s),
                _ => None,
            })
            .collect()
    }

    /// Count active Refutes edges pointing at a claim
    pub fn count_active_refutes(&self, claim_id: &str) -> u32 {
        self.incoming_neighbors(claim_id)
            .into_iter()
            .filter(|(node, edge)| {
                matches!(edge, EdgeKind::Refutes)
                    && match node {
                        NodeKind::Evidence(e) => e.meta.belief_score >= 0.7,
                        _ => false,
                    }
            })
            .count() as u32
    }

    /// Get all ClaimNodes supporting a hypothesis (via Supports edge)
    pub fn get_hypothesis_claims(&self, hypothesis_id: &str) -> Vec<&ClaimNode> {
        self.incoming_neighbors(hypothesis_id)
            .into_iter()
            .filter_map(|(node, edge)| match (node, edge) {
                (NodeKind::Claim(c), EdgeKind::Supports) => Some(c),
                _ => None,
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // Graph statistics for monitoring
    // -----------------------------------------------------------------------

    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Summary counts by node type
    pub fn type_counts(&self) -> HashMap<&'static str, usize> {
        let mut counts = HashMap::new();
        for node in self.graph.node_weights() {
            *counts.entry(node.type_name()).or_insert(0) += 1;
        }
        counts
    }
}

impl Default for SageGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_hypothesis(text: &str) -> NodeKind {
        NodeKind::Hypothesis(HypothesisNode {
            meta: NodeMeta::new(0, 0.0),
            text: text.to_string(),
            status: HypothesisStatus::Open,
            priority: 0.8,
            stall_count: 0,
            max_reopen: 2,
            reopen_count: 0,
            test_id: None,
        })
    }

    fn make_claim(text: &str) -> NodeKind {
        NodeKind::Claim(ClaimNode {
            meta: NodeMeta::new(0, 0.0),
            text: text.to_string(),
            evidence_ids: vec![],
            status: ClaimStatus::Unverified,
        })
    }

    fn make_evidence(snippet: &str, source_id: &str, quality: QualityTier) -> NodeKind {
        NodeKind::Evidence(EvidenceNode {
            meta: NodeMeta::new(0, quality.initial_belief_score()),
            snippet: snippet.to_string(),
            source_id: source_id.to_string(),
            quality,
        })
    }

    fn make_source(url: &str) -> NodeKind {
        NodeKind::Source(SourceNode {
            meta: NodeMeta::new(0, 0.5),
            url: url.to_string(),
            status: SourceStatus::Visited,
            domain_auth: 0.75,
            reliability: 0.75,
        })
    }

    #[test]
    fn test_add_and_retrieve_nodes() {
        let mut g = SageGraph::new();
        let h = make_hypothesis("test hyp");
        let id = h.id().to_string();
        g.add_node(h);

        assert!(g.node_exists(&id));
        assert!(g.get_node(&id).is_some());
        assert_eq!(g.node_count(), 1);
    }

    #[test]
    fn test_add_edge_and_neighbors() {
        let mut g = SageGraph::new();
        let source = make_source("https://example.com");
        let source_id = source.id().to_string();
        g.add_node(source);

        let evidence = make_evidence("test snippet", &source_id, QualityTier::Primary);
        let evidence_id = evidence.id().to_string();
        g.add_node(evidence);

        let claim = make_claim("test claim");
        let claim_id = claim.id().to_string();
        g.add_node(claim);

        // Evidence -> Claim (Supports)
        g.add_edge(&evidence_id, &claim_id, EdgeKind::Supports)
            .unwrap();

        let supporting = g.get_supporting_evidence(&claim_id);
        assert_eq!(supporting.len(), 1);
        assert_eq!(supporting[0].snippet, "test snippet");
    }

    #[test]
    fn test_edge_nonexistent_node() {
        let mut g = SageGraph::new();
        let result = g.add_edge("fake-id-1", "fake-id-2", EdgeKind::Supports);
        assert!(result.is_err());
    }

    #[test]
    fn test_type_counts() {
        let mut g = SageGraph::new();
        g.add_node(make_hypothesis("h1"));
        g.add_node(make_hypothesis("h2"));
        g.add_node(make_claim("c1"));

        let counts = g.type_counts();
        assert_eq!(counts["Hypothesis"], 2);
        assert_eq!(counts["Claim"], 1);
    }
}
