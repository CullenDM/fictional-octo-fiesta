//! GREAT SAGE Harness — PyO3 Module Entry Point
//!
//! This is the Python-importable module.  Python calls into these functions
//! for all deterministic operations (graph mutations, validation, scoring,
//! phase execution, snapshots).

pub mod audit;
pub mod domain_auth;
pub mod graph;
pub mod phases;
pub mod schema;
pub mod scorer;
pub mod snapshot;
pub mod tmpref;
pub mod tools;
pub mod validator;

use audit::AuditTrail;
use domain_auth::DomainAuthorityConfig;
use graph::SageGraph;
use phases::{MonitorRouting, PhaseConfig};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use schema::*;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

/// Thread-safe shared graph handle exposed to Python
#[pyclass]
struct HarnessState {
    graph: Arc<RwLock<SageGraph>>,
    audit: Arc<AuditTrail>,
    domain_auth: DomainAuthorityConfig,
    config: PhaseConfig,
    snapshot_dir: PathBuf,
    run_id: String,
}

#[pymethods]
impl HarnessState {
    #[new]
    #[pyo3(signature = (snapshot_dir, config_dict=None, domain_auth_path=None))]
    #[allow(unused_variables)]
    fn new(
        snapshot_dir: String,
        config_dict: Option<&Bound<'_, PyDict>>,
        domain_auth_path: Option<String>,
    ) -> PyResult<Self> {
        let snapshot_path = PathBuf::from(&snapshot_dir);
        let run_id = uuid::Uuid::new_v4().to_string()[..8].to_string();

        let config = if let Some(dict) = config_dict {
            parse_config(dict)?
        } else {
            PhaseConfig::default()
        };

        let domain_auth = if let Some(path) = domain_auth_path {
            DomainAuthorityConfig::load(&PathBuf::from(path))
        } else {
            DomainAuthorityConfig::default()
        };

        let audit = AuditTrail::new(&snapshot_path, &run_id);

        Ok(Self {
            graph: Arc::new(RwLock::new(SageGraph::new())),
            audit: Arc::new(audit),
            domain_auth,
            config,
            snapshot_dir: snapshot_path,
            run_id,
        })
    }

    // -----------------------------------------------------------------------
    // Graph operations
    // -----------------------------------------------------------------------

    /// Insert a hypothesis node from Python dict
    #[pyo3(signature = (text, priority, test_id=None))]
    fn add_hypothesis(
        &self,
        text: String,
        priority: f32,
        test_id: Option<String>,
    ) -> PyResult<String> {
        let mut g = self.graph.write().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Lock poisoned: {}", e))
        })?;

        let round_id = g.current_round;
        let node = NodeKind::Hypothesis(HypothesisNode {
            meta: NodeMeta::new(round_id, 0.0),
            text,
            status: HypothesisStatus::Open,
            priority: priority.clamp(0.0, 1.0),
            stall_count: 0,
            max_reopen: 2,
            reopen_count: 0,
            test_id,
        });

        let id = node.id().to_string();
        g.add_node(node);

        self.audit.log_mutation(round_id, 1, vec![id.clone()]);
        Ok(id)
    }

    /// Get graph statistics as a Python dict
    fn graph_stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let g = self.graph.read().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Lock poisoned: {}", e))
        })?;

        let dict = PyDict::new(py);
        dict.set_item("node_count", g.node_count())?;
        dict.set_item("edge_count", g.edge_count())?;
        dict.set_item("current_round", g.current_round)?;

        let type_counts = g.type_counts();
        let types_dict = PyDict::new(py);
        for (k, v) in type_counts {
            types_dict.set_item(k, v)?;
        }
        dict.set_item("type_counts", types_dict)?;

        Ok(dict.into())
    }

    /// Get all hypothesis statuses as JSON string
    fn get_hypotheses_json(&self) -> PyResult<String> {
        let g = self.graph.read().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Lock poisoned: {}", e))
        })?;
        let hypotheses = g.hypotheses();
        serde_json::to_string(&hypotheses)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Serialize: {}", e)))
    }

    /// Get all Verified claims as JSON string (for prompt context pruning)
    fn get_verified_claims_json(&self) -> PyResult<String> {
        let g = self.graph.read().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Lock poisoned: {}", e))
        })?;
        let verified: Vec<&ClaimNode> = g
            .claims()
            .into_iter()
            .filter(|c| c.status == ClaimStatus::Verified)
            .collect();
        serde_json::to_string(&verified)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Serialize: {}", e)))
    }

    /// Get all node IDs currently in the graph (for tmp-ref validation)
    fn get_all_node_ids(&self) -> PyResult<Vec<String>> {
        let g = self.graph.read().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Lock poisoned: {}", e))
        })?;

        Ok(g.graph
            .node_weights()
            .map(|n| n.id().to_string())
            .collect())
    }

    // -----------------------------------------------------------------------
    // Validation & Delta Application
    // -----------------------------------------------------------------------

    /// Validate a model delta JSON string against the Forbidden Model Writes
    /// wall and all structural validators.
    /// Returns (passed: bool, violations: list[str])
    fn validate_delta(&self, delta_json: &str, phase: u32) -> PyResult<(bool, Vec<String>)> {
        let delta: serde_json::Value = serde_json::from_str(delta_json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid JSON: {}", e)))?;

        let g = self.graph.read().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Lock poisoned: {}", e))
        })?;

        let result = validator::validate_model_delta(&delta, &g, phase);

        if !result.passed {
            let round_id = g.current_round;
            for v in &result.violations {
                self.audit
                    .log_rejection(round_id, phase, &v.check_id, &v.message);
            }
        }

        let violation_strs: Vec<String> = result.violations.iter().map(|v| v.to_string()).collect();
        Ok((result.passed, violation_strs))
    }

    /// Resolve tmp_* references in a model delta and apply to graph.
    /// This is the atomic commit path: validate → resolve → apply.
    /// Returns (success: bool, message: str, new_node_ids: list[str])
    fn apply_model_delta(
        &self,
        delta_json: &str,
        phase: u32,
    ) -> PyResult<(bool, String, Vec<String>)> {
        let delta: serde_json::Value = serde_json::from_str(delta_json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid JSON: {}", e)))?;

        let mut g = self.graph.write().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Lock poisoned: {}", e))
        })?;
        let round_id = g.current_round;

        // Step 1: Validate
        let validation = validator::validate_model_delta(&delta, &g, phase);
        if !validation.passed {
            for v in &validation.violations {
                self.audit
                    .log_rejection(round_id, phase, &v.check_id, &v.message);
            }
            let msg = validation
                .violations
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join("; ");
            return Ok((false, msg, vec![]));
        }

        // Step 2: Resolve tmp_* references
        let existing_ids = g
            .graph
            .node_weights()
            .map(|n| n.id().to_string())
            .collect::<Vec<_>>();

        let (resolved, tmp_map) = match tmpref::resolve_tmp_refs(&delta, &existing_ids) {
            Ok(r) => r,
            Err(e) => {
                self.audit.log_rejection(round_id, phase, "V-02", &e);
                return Ok((false, e, vec![]));
            }
        };

        // Step 3: Apply nodes and edges atomically
        let mut new_ids = Vec::new();

        // Add sources first (evidence nodes reference them)
        if let Some(sources) = resolved.get("new_sources").and_then(|v| v.as_array()) {
            for source_val in sources {
                let tmp_id = source_val
                    .get("tmp_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let real_id = tmp_map
                    .get(tmp_id)
                    .cloned()
                    .unwrap_or_else(|| tmp_id.to_string());
                let url = source_val
                    .get("source_url")
                    .or_else(|| source_val.get("url"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                let domain_auth = self.domain_auth.get_domain_auth(&url);

                let node = NodeKind::Source(SourceNode {
                    meta: NodeMeta {
                        id: real_id.clone(),
                        ..NodeMeta::new(round_id, 0.5)
                    },
                    url,
                    status: SourceStatus::Unvisited,
                    domain_auth,
                    reliability: domain_auth, // recency_factor applied later for local files
                });
                g.add_node(node);
                new_ids.push(real_id);
            }
        }

        // Add evidence
        if let Some(evidence) = resolved.get("new_evidence").and_then(|v| v.as_array()) {
            for ev_val in evidence {
                let tmp_id = ev_val
                    .get("tmp_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let real_id = tmp_map
                    .get(tmp_id)
                    .cloned()
                    .unwrap_or_else(|| tmp_id.to_string());

                let snippet = ev_val
                    .get("snippet")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                // Resolve source: either an existing source or find/create from source_url
                let source_url = ev_val
                    .get("source_url")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                let source_id = ev_val
                    .get("source_id")
                    .and_then(|v| v.as_str())
                    .map(|s| {
                        tmp_map
                            .get(s)
                            .cloned()
                            .unwrap_or_else(|| s.to_string())
                    })
                    .unwrap_or_else(|| {
                        // Auto-create source if needed
                        let sid = uuid::Uuid::new_v4().to_string();
                        let domain_auth = self.domain_auth.get_domain_auth(&source_url);
                        let src = NodeKind::Source(SourceNode {
                            meta: NodeMeta {
                                id: sid.clone(),
                                ..NodeMeta::new(round_id, 0.5)
                            },
                            url: source_url.clone(),
                            status: SourceStatus::Unvisited,
                            domain_auth,
                            reliability: domain_auth,
                        });
                        g.add_node(src);
                        new_ids.push(sid.clone());
                        sid
                    });

                let quality_hint = ev_val
                    .get("quality_hint")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Primary");

                let quality = match quality_hint {
                    "TestVerified" => QualityTier::TestVerified,
                    "Corroborated" => QualityTier::Corroborated,
                    "Inferred" => QualityTier::Inferred,
                    "Contradicted" => QualityTier::Contradicted,
                    _ => QualityTier::Primary,
                };

                let node = NodeKind::Evidence(EvidenceNode {
                    meta: NodeMeta {
                        id: real_id.clone(),
                        belief_score: quality.initial_belief_score(),
                        source_entropy: 1.0 - quality.initial_belief_score(),
                        ..NodeMeta::new(round_id, quality.initial_belief_score())
                    },
                    snippet,
                    source_id: source_id.clone(),
                    quality,
                });
                g.add_node(node);
                new_ids.push(real_id.clone());

                // Auto-add Cites edge: Evidence -> Source
                let _ = g.add_edge(&real_id, &source_id, EdgeKind::Cites);
            }
        }

        // Add claims
        if let Some(claims) = resolved.get("new_claims").and_then(|v| v.as_array()) {
            for claim_val in claims {
                let tmp_id = claim_val
                    .get("tmp_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let real_id = tmp_map
                    .get(tmp_id)
                    .cloned()
                    .unwrap_or_else(|| tmp_id.to_string());

                let text = claim_val
                    .get("text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                let node = NodeKind::Claim(ClaimNode {
                    meta: NodeMeta {
                        id: real_id.clone(),
                        ..NodeMeta::new(round_id, 0.0)
                    },
                    text,
                    evidence_ids: vec![],
                    status: ClaimStatus::Unverified,
                });
                g.add_node(node);
                new_ids.push(real_id);
            }
        }

        // Add edges
        if let Some(edges) = resolved.get("new_edges").and_then(|v| v.as_array()) {
            for edge_val in edges {
                let from_id = edge_val
                    .get("from_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let to_id = edge_val
                    .get("to_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let kind_str = edge_val
                    .get("kind")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");

                let kind = match kind_str {
                    "Supports" => EdgeKind::Supports,
                    "Refutes" => EdgeKind::Refutes,
                    "DerivedFrom" => EdgeKind::DerivedFrom,
                    "Cites" => EdgeKind::Cites,
                    "Proposes" => EdgeKind::Proposes,
                    "Invalidates" => EdgeKind::Invalidates,
                    _ => continue,
                };

                if let Err(e) = g.add_edge(from_id, to_id, kind) {
                    self.audit
                        .log_rejection(round_id, phase, "V-02", &e);
                    // Non-fatal: log and continue with remaining edges
                }
            }
        }

        // Update claim statuses based on support
        // Claims with at least one Supports edge -> Supported
        for id in &new_ids {
            if let Some(NodeKind::Claim(_)) = g.get_node(id) {
                let support_mass = scorer::compute_support_mass(id, &g);
                if support_mass > 0.0 {
                    if let Some(NodeKind::Claim(c)) = g.get_node_mut(id) {
                        c.status = ClaimStatus::Supported;
                    }
                    // Update belief from evidence quality
                    scorer::update_belief_from_evidence(id, &mut g);
                }
            }
        }

        self.audit
            .log_mutation(round_id, phase, new_ids.clone());

        Ok((true, "Applied successfully".to_string(), new_ids))
    }

    // -----------------------------------------------------------------------
    // Phase execution
    // -----------------------------------------------------------------------

    /// Phase 2: Score and allocate budget.
    /// Returns JSON array of [{id, eig, budget}]
    fn run_phase_score(&self, round_budget: u32) -> PyResult<String> {
        let mut g = self.graph.write().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Lock poisoned: {}", e))
        })?;

        let allocations = phases::phase_score(&mut g, round_budget);

        self.audit.log_eig_allocation(
            g.current_round,
            &allocations
                .iter()
                .map(|(id, eig, budget)| (id.clone(), *eig, *budget))
                .collect::<Vec<_>>(),
        );

        let result: Vec<serde_json::Value> = allocations
            .iter()
            .map(|(id, eig, budget)| {
                serde_json::json!({
                    "id": id,
                    "eig": eig,
                    "budget": budget
                })
            })
            .collect();

        serde_json::to_string(&result)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Serialize: {}", e)))
    }

    /// Phase 5: Audit — deterministic contradiction detection
    fn run_phase_audit(&self) -> PyResult<String> {
        let mut g = self.graph.write().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Lock poisoned: {}", e))
        })?;
        let round_id = g.current_round;

        let summary = phases::phase_audit(&mut g);

        self.audit.log(
            &audit::AuditEntry::new(round_id, 5, "audit_complete")
                .with_detail(&format!(
                    "contradictions={}, stale={}, disputed={}",
                    summary.contradiction_count, summary.stale_count, summary.disputed_count
                )),
        );

        Ok(serde_json::to_string(&serde_json::json!({
            "contradiction_count": summary.contradiction_count,
            "stale_count": summary.stale_count,
            "disputed_count": summary.disputed_count,
        }))
        .unwrap())
    }

    /// Phase 6: Skeptic STUB — auto-promote Supported → Verified
    fn run_phase_skeptic_stub(&self) -> PyResult<()> {
        let mut g = self.graph.write().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Lock poisoned: {}", e))
        })?;
        phases::phase_skeptic_stub(&mut g, &self.config);
        Ok(())
    }

    /// Phase 7: Monitor — evaluate routing decision
    /// Returns "continue", "stall", or "converge"
    fn run_phase_monitor(
        &self,
        tokens_consumed: u32,
        all_tests_pass: bool,
        prev_eigs_json: &str,
    ) -> PyResult<String> {
        let prev_eigs: Vec<(String, f32)> = serde_json::from_str(prev_eigs_json)
            .unwrap_or_default();

        let mut g = self.graph.write().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Lock poisoned: {}", e))
        })?;
        let round_id = g.current_round;

        let routing = phases::phase_monitor(
            &mut g,
            round_id,
            tokens_consumed,
            all_tests_pass,
            &self.config,
            &prev_eigs,
        );

        let (decision, reason) = match routing {
            MonitorRouting::Continue => ("continue", "active hypotheses remain"),
            MonitorRouting::Stall => ("stall", "hypotheses stalled"),
            MonitorRouting::Converge => ("converge", "convergence criteria met"),
        };

        self.audit.log_routing(round_id, decision, reason);

        Ok(decision.to_string())
    }

    /// Phase 8: Reframe STUB — mark stalled as Refuted
    fn run_phase_reframe_stub(&self) -> PyResult<()> {
        let mut g = self.graph.write().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Lock poisoned: {}", e))
        })?;
        phases::phase_reframe_stub(&mut g);
        Ok(())
    }

    /// Phase 9: Bank — score candidates, return ranked list
    fn run_phase_bank(&self) -> PyResult<String> {
        let mut g = self.graph.write().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Lock poisoned: {}", e))
        })?;

        let scored = phases::phase_bank(&mut g);

        let result: Vec<serde_json::Value> = scored
            .iter()
            .map(|(id, s_a, cs)| {
                serde_json::json!({
                    "id": id,
                    "s_a": s_a,
                    "contradiction_score": cs
                })
            })
            .collect();

        serde_json::to_string(&result)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Serialize: {}", e)))
    }

    /// Increment the round counter
    fn advance_round(&self) -> PyResult<u32> {
        let mut g = self.graph.write().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Lock poisoned: {}", e))
        })?;
        g.current_round += 1;
        Ok(g.current_round)
    }

    // -----------------------------------------------------------------------
    // Snapshot operations
    // -----------------------------------------------------------------------

    /// Save graph snapshot to disk
    fn save_snapshot(&self) -> PyResult<String> {
        let g = self.graph.read().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Lock poisoned: {}", e))
        })?;

        let filename = format!("snapshot_round_{}.json", g.current_round);
        let path = self.snapshot_dir.join(&filename);

        snapshot::save_snapshot(&g, &path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e))?;

        Ok(path.to_string_lossy().to_string())
    }

    /// Load graph from snapshot file
    fn load_snapshot(&self, path: &str) -> PyResult<()> {
        let loaded = snapshot::load_snapshot(&PathBuf::from(path))
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e))?;

        let mut g = self.graph.write().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Lock poisoned: {}", e))
        })?;
        *g = loaded;

        Ok(())
    }

    /// Get the run ID
    fn get_run_id(&self) -> String {
        self.run_id.clone()
    }
}

/// Parse Python config dict into PhaseConfig
fn parse_config(dict: &Bound<'_, PyDict>) -> PyResult<PhaseConfig> {
    let mut config = PhaseConfig::default();

    if let Some(v) = dict.get_item("eig_min")? {
        config.eig_min = v.extract()?;
    }
    if let Some(v) = dict.get_item("eig_delta_stall")? {
        config.eig_delta_stall = v.extract()?;
    }
    if let Some(v) = dict.get_item("stall_count")? {
        config.stall_count_threshold = v.extract()?;
    }
    if let Some(v) = dict.get_item("convergence_sa")? {
        config.convergence_sa = v.extract()?;
    }
    if let Some(v) = dict.get_item("diversity_min")? {
        config.diversity_min = v.extract()?;
    }
    if let Some(v) = dict.get_item("support_mass_min")? {
        config.support_mass_min = v.extract()?;
    }
    if let Some(v) = dict.get_item("total_tokens")? {
        config.total_budget = v.extract()?;
    }
    if let Some(v) = dict.get_item("max_rounds")? {
        config.max_rounds = v.extract()?;
    }
    if let Some(v) = dict.get_item("redundancy_pct")? {
        config.redundancy_pct = v.extract()?;
    }

    Ok(config)
}

/// Python module definition
#[pymodule]
fn great_sage_harness(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<HarnessState>()?;
    m.add_class::<HypothesisStatus>()?;
    m.add_class::<ClaimStatus>()?;
    m.add_class::<QualityTier>()?;
    m.add_class::<EdgeKind>()?;
    m.add_class::<SourceStatus>()?;
    m.add_class::<ContradictionSource>()?;
    Ok(())
}
