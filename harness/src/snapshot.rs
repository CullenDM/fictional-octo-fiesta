//! JSONL Snapshot Serialization (§1.3)
//!
//! Snapshot is written after Phase 6 (Skeptic) completes — before any
//! model unload.  On restart, the harness deserializes and reconstructs
//! the petgraph in-memory structure before resuming.

use crate::graph::SageGraph;
use std::fs;
use std::io::Write;
use std::path::Path;

/// Serialize the full graph state to JSONL.
/// Includes: full node table, full edge table, current round_id,
/// and all CandidateAnswerNode entries with their lineage UUIDs.
pub fn save_snapshot(graph: &SageGraph, path: &Path) -> Result<(), String> {
    let json = serde_json::to_string_pretty(graph)
        .map_err(|e| format!("Snapshot serialization failed: {}", e))?;

    let mut file = fs::File::create(path)
        .map_err(|e| format!("Cannot create snapshot file: {}", e))?;

    file.write_all(json.as_bytes())
        .map_err(|e| format!("Snapshot write failed: {}", e))?;

    // fsync enforcement (§1.3, §1.4)
    file.sync_all()
        .map_err(|e| format!("fsync failed: {}", e))?;

    Ok(())
}

/// Load a snapshot from disk and reconstruct the SageGraph.
pub fn load_snapshot(path: &Path) -> Result<SageGraph, String> {
    let content =
        fs::read_to_string(path).map_err(|e| format!("Cannot read snapshot: {}", e))?;

    let mut graph: SageGraph =
        serde_json::from_str(&content).map_err(|e| format!("Snapshot parse failed: {}", e))?;

    // Rebuild the id_index from the deserialized graph
    graph.rebuild_index();

    Ok(graph)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::*;

    #[test]
    fn test_snapshot_round_trip() {
        let mut graph = SageGraph::new();
        graph.current_round = 3;

        let h = NodeKind::Hypothesis(HypothesisNode {
            meta: NodeMeta::new(0, 0.0),
            text: "Test hypothesis".to_string(),
            status: HypothesisStatus::Open,
            priority: 0.8,
            stall_count: 0,
            max_reopen: 2,
            reopen_count: 0,
            test_id: None,
        });
        let h_id = h.id().to_string();
        graph.add_node(h);

        let c = NodeKind::Claim(ClaimNode {
            meta: NodeMeta::new(0, 0.5),
            text: "Test claim".to_string(),
            evidence_ids: vec![],
            status: ClaimStatus::Supported,
        });
        let c_id = c.id().to_string();
        graph.add_node(c);

        graph
            .add_edge(&c_id, &h_id, EdgeKind::Supports)
            .unwrap();

        // Save
        let path = std::env::temp_dir().join("great_sage_test_snapshot.json");
        save_snapshot(&graph, &path).unwrap();

        // Load
        let loaded = load_snapshot(&path).unwrap();
        assert_eq!(loaded.current_round, 3);
        assert_eq!(loaded.node_count(), 2);
        assert_eq!(loaded.edge_count(), 1);
        assert!(loaded.node_exists(&h_id));
        assert!(loaded.node_exists(&c_id));

        // Clean up
        fs::remove_file(&path).ok();
    }
}
