//! Temporary Reference Resolution (Phase 4 Compress)
//!
//! Models cannot know the UUIDs of nodes created in the same batch.
//! They use tmp_claim_N, tmp_evidence_N, tmp_source_N as local references.
//! The Harness resolves all tmp_* refs to real UUIDs during atomic commit.

use std::collections::HashMap;
use uuid::Uuid;

/// Parse a model delta JSON and resolve all tmp_* references to real UUIDs.
/// Returns a mapping of tmp_id -> real UUID, and the rewritten JSON value.
pub fn resolve_tmp_refs(
    delta: &serde_json::Value,
    existing_ids: &[String],
) -> Result<(serde_json::Value, HashMap<String, String>), String> {
    let mut tmp_map: HashMap<String, String> = HashMap::new();

    // First pass: collect all tmp_* IDs and assign real UUIDs
    collect_tmp_ids(delta, &mut tmp_map);

    // Assign UUIDs
    for val in tmp_map.values_mut() {
        *val = Uuid::new_v4().to_string();
    }

    // Second pass: replace all tmp_* references with real UUIDs
    let resolved = replace_tmp_refs(delta, &tmp_map);

    // Third pass: validate that all referenced IDs exist
    // (either in the existing graph or in the newly assigned batch)
    let new_ids: Vec<String> = tmp_map.values().cloned().collect();
    validate_refs(&resolved, existing_ids, &new_ids)?;

    Ok((resolved, tmp_map))
}

/// Recursively collect all tmp_* identifiers from the delta
fn collect_tmp_ids(value: &serde_json::Value, map: &mut HashMap<String, String>) {
    match value {
        serde_json::Value::String(s) if is_tmp_ref(s) => {
            map.entry(s.clone()).or_insert_with(String::new);
        }
        serde_json::Value::Object(obj) => {
            for v in obj.values() {
                collect_tmp_ids(v, map);
            }
        }
        serde_json::Value::Array(arr) => {
            for v in arr {
                collect_tmp_ids(v, map);
            }
        }
        _ => {}
    }
}

/// Recursively replace all tmp_* references with assigned UUIDs
fn replace_tmp_refs(
    value: &serde_json::Value,
    map: &HashMap<String, String>,
) -> serde_json::Value {
    match value {
        serde_json::Value::String(s) if is_tmp_ref(s) => {
            serde_json::Value::String(map.get(s).cloned().unwrap_or_else(|| s.clone()))
        }
        serde_json::Value::Object(obj) => {
            let mut new_obj = serde_json::Map::new();
            for (k, v) in obj {
                new_obj.insert(k.clone(), replace_tmp_refs(v, map));
            }
            serde_json::Value::Object(new_obj)
        }
        serde_json::Value::Array(arr) => {
            serde_json::Value::Array(arr.iter().map(|v| replace_tmp_refs(v, map)).collect())
        }
        other => other.clone(),
    }
}

/// Validate that all UUID-like references in edge from_id/to_id resolve
/// to either an existing graph node or a node created in the current batch.
fn validate_refs(
    value: &serde_json::Value,
    existing_ids: &[String],
    new_ids: &[String],
) -> Result<(), String> {
    if let Some(edges) = value.get("new_edges").and_then(|v| v.as_array()) {
        for edge in edges {
            if let Some(from_id) = edge.get("from_id").and_then(|v| v.as_str()) {
                if !existing_ids.contains(&from_id.to_string())
                    && !new_ids.contains(&from_id.to_string())
                {
                    return Err(format!(
                        "V-02: Dangling from_id reference: {}",
                        from_id
                    ));
                }
            }
            if let Some(to_id) = edge.get("to_id").and_then(|v| v.as_str()) {
                if !existing_ids.contains(&to_id.to_string())
                    && !new_ids.contains(&to_id.to_string())
                {
                    return Err(format!("V-02: Dangling to_id reference: {}", to_id));
                }
            }
        }
    }
    Ok(())
}

/// Check if a string is a tmp_* reference
fn is_tmp_ref(s: &str) -> bool {
    s.starts_with("tmp_claim_")
        || s.starts_with("tmp_evidence_")
        || s.starts_with("tmp_source_")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_resolve_basic_delta() {
        let delta = json!({
            "new_claims": [
                {"tmp_id": "tmp_claim_1", "text": "Test claim"}
            ],
            "new_evidence": [
                {"tmp_id": "tmp_evidence_1", "snippet": "test", "source_url": "file://test"}
            ],
            "new_edges": [
                {"from_id": "tmp_evidence_1", "to_id": "tmp_claim_1", "kind": "Supports"}
            ]
        });

        let (resolved, tmp_map) = resolve_tmp_refs(&delta, &[]).unwrap();
        assert_eq!(tmp_map.len(), 2);

        // Verify tmp_* refs were replaced with UUIDs
        let claim_id = &tmp_map["tmp_claim_1"];
        let evidence_id = &tmp_map["tmp_evidence_1"];
        assert!(!claim_id.starts_with("tmp_"));
        assert!(!evidence_id.starts_with("tmp_"));

        // Verify the edge references were resolved
        let edge = &resolved["new_edges"][0];
        assert_eq!(edge["from_id"].as_str().unwrap(), evidence_id);
        assert_eq!(edge["to_id"].as_str().unwrap(), claim_id);
    }

    #[test]
    fn test_dangling_reference_rejected() {
        let delta = json!({
            "new_claims": [],
            "new_evidence": [],
            "new_edges": [
                {"from_id": "nonexistent-uuid", "to_id": "also-nonexistent", "kind": "Supports"}
            ]
        });

        let result = resolve_tmp_refs(&delta, &[]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("V-02"));
    }

    #[test]
    fn test_mixed_existing_and_new_refs() {
        let existing_id = "existing-uuid-12345";
        let delta = json!({
            "new_claims": [
                {"tmp_id": "tmp_claim_1", "text": "New claim"}
            ],
            "new_evidence": [],
            "new_edges": [
                {"from_id": existing_id, "to_id": "tmp_claim_1", "kind": "Supports"}
            ]
        });

        let result = resolve_tmp_refs(&delta, &[existing_id.to_string()]);
        assert!(result.is_ok());
    }
}
