"""Integration test for GREAT SAGE Harness (runs without Ollama)."""
import json
import sys

import great_sage_harness as h


def main():
    print("=== Test 1: Harness Lifecycle ===")
    state = h.HarnessState("./snapshots/")
    h1 = state.add_hypothesis(
        "The function should return the sum of two numbers", 0.9, "test_add"
    )
    h2 = state.add_hypothesis("Edge cases should handle negative numbers", 0.5)
    print(f"H1: {h1[:8]}  H2: {h2[:8]}")
    print(f"Stats: {state.graph_stats()}")

    allocs = json.loads(state.run_phase_score(10000))
    print(
        f"Score allocations: "
        f"{[(a['id'][:8], round(a['eig'], 3), a['budget']) for a in allocs]}"
    )

    print()
    print("=== Test 2: Validator Wall ===")

    # FORBIDDEN: belief_score in model delta → REJECT
    bad_delta = json.dumps(
        {
            "new_claims": [
                {"tmp_id": "tmp_claim_1", "text": "test", "belief_score": 0.95}
            ],
            "new_evidence": [
                {
                    "tmp_id": "tmp_evidence_1",
                    "snippet": "proof",
                    "source_url": "file://test.py",
                }
            ],
            "new_edges": [
                {
                    "from_id": "tmp_evidence_1",
                    "to_id": "tmp_claim_1",
                    "kind": "Supports",
                }
            ],
        }
    )
    ok, msg, ids = state.apply_model_delta(bad_delta, 4)
    print(f"Forbidden field: passed={ok}  msg={msg[:100]}")
    assert not ok, "V-04 should reject belief_score!"

    # FORBIDDEN: status in model delta → REJECT
    bad_delta2 = json.dumps(
        {
            "new_claims": [
                {"tmp_id": "tmp_claim_1", "text": "test", "status": "Verified"}
            ],
            "new_evidence": [
                {"tmp_id": "tmp_evidence_1", "snippet": "p", "source_url": "file://x"}
            ],
            "new_edges": [
                {
                    "from_id": "tmp_evidence_1",
                    "to_id": "tmp_claim_1",
                    "kind": "Supports",
                }
            ],
        }
    )
    ok2, msg2, _ = state.apply_model_delta(bad_delta2, 4)
    print(f"Forbidden status: passed={ok2}  msg={msg2[:100]}")
    assert not ok2, "V-04 should reject status!"

    # FORBIDDEN: id in model delta → REJECT (V-03)
    bad_delta3 = json.dumps(
        {
            "new_claims": [
                {"tmp_id": "tmp_claim_1", "text": "test", "id": "hacked-uuid"}
            ],
            "new_evidence": [
                {"tmp_id": "tmp_evidence_1", "snippet": "p", "source_url": "file://x"}
            ],
            "new_edges": [
                {
                    "from_id": "tmp_evidence_1",
                    "to_id": "tmp_claim_1",
                    "kind": "Supports",
                }
            ],
        }
    )
    ok3, msg3, _ = state.apply_model_delta(bad_delta3, 4)
    print(f"Immutable id: passed={ok3}  msg={msg3[:100]}")
    assert not ok3, "V-03 should reject id!"

    # CLEAN delta → ACCEPT
    good_delta = json.dumps(
        {
            "new_claims": [{"tmp_id": "tmp_claim_1", "text": "sum(1,2) returns 3"}],
            "new_evidence": [
                {
                    "tmp_id": "tmp_evidence_1",
                    "snippet": "assert sum(1,2)==3 PASSED",
                    "source_url": "file://test.py",
                    "quality_hint": "TestVerified",
                }
            ],
            "new_edges": [
                {
                    "from_id": "tmp_evidence_1",
                    "to_id": "tmp_claim_1",
                    "kind": "Supports",
                },
                {"from_id": "tmp_claim_1", "to_id": h1, "kind": "Supports"},
            ],
        }
    )
    ok, msg, ids = state.apply_model_delta(good_delta, 4)
    print(f"Clean delta: passed={ok}  new_ids={len(ids)}  msg={msg}")
    assert ok, f"Should accept: {msg}"

    print()
    print("=== Test 3: Phase Chain ===")
    audit = json.loads(state.run_phase_audit())
    print(f"Audit: {audit}")

    state.run_phase_skeptic_stub()
    verified = json.loads(state.get_verified_claims_json())
    print(f"Verified claims after stub Skeptic: {len(verified)}")

    routing = state.run_phase_monitor(100, False, "[]")
    print(f"Monitor routing: {routing}")

    # Test convergence on all tests pass
    routing2 = state.run_phase_monitor(100, True, "[]")
    print(f"Monitor routing (tests pass): {routing2}")
    assert routing2 == "converge", "Should converge when tests pass!"

    print()
    print("=== Test 4: Snapshot Round-trip ===")
    snap_path = state.save_snapshot()
    print(f"Saved: {snap_path}")
    state.load_snapshot(snap_path)
    stats = state.graph_stats()
    print(f"Loaded: {stats}")
    assert stats["node_count"] > 0, "Should have nodes after load!"

    print()
    print("=== Test 5: Tool Executor ===")
    from orchestrator.tool_executor import execute_code_run, parse_pytest_output

    result = execute_code_run("echo 'hello world'", ".", 5)
    print(f"CodeRun: exit={result.exit_code} stdout={result.stdout.strip()}")
    assert result.exit_code == 0
    assert "hello world" in result.stdout

    # Test pytest parser
    sample_output = """
test_math.py::test_add PASSED
test_math.py::test_subtract PASSED
test_math.py::test_divide FAILED
============== 2 passed, 1 failed ==============
"""
    passed, failed = parse_pytest_output(sample_output)
    print(f"Pytest parser: {len(passed)} passed, {len(failed)} failed")
    assert len(passed) == 2
    assert len(failed) == 1

    print()
    print("=" * 50)
    print("  ALL INTEGRATION TESTS PASSED ✓")
    print("=" * 50)


if __name__ == "__main__":
    main()
