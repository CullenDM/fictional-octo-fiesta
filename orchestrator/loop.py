"""
GREAT SAGE Main Orchestrator Loop

Routing: 1 → 2 → 3 → 4 → 5 → 6 → 7 → route
From 7: Continue → 2, Stall → 8 (stub), Converge → 9 → 10.

Python owns model call sites and role swaps.
Rust harness owns deterministic graph logic, scoring, and validation.
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml

# Rust harness (via PyO3)
import great_sage_harness as harness

from .ollama_client import OllamaClient, OllamaConfig
from . import brain
from .tool_executor import execute_tool

logger = logging.getLogger("great_sage")


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    """Load and validate configuration."""
    if os.path.exists(config_path):
        with open(config_path) as f:
            return yaml.safe_load(f)
    else:
        # Defaults from §9.2
        return {
            "model": {
                "worker": "hf.co/TeichAI/Qwen3-4B-Thinking-2507-Claude-4.5-Opus-High-Reasoning-Distill-GGUF:Q3_K_S",
                "skeptic": "nemotron-mini:4b",
                "worker_ctx": 8192,
                "skeptic_ctx": 4096,
            },
            "budget": {
                "total_tokens": 200000,
                "round_tokens": 25000,
                "max_rounds": 12,
                "tool_steps": 5,
                "tool_timeout": 30,
            },
            "thresholds": {
                "eig_min": 0.05,
                "eig_delta_stall": 0.01,
                "stall_count": 2,
                "convergence_sa": 0.85,
                "diversity_min": 0.50,
                "support_mass_min": 1.00,
                "reopen_max": 2,
            },
            "snapshot": {
                "format": "jsonl",
                "path": "./snapshots/",
            },
        }


async def run_great_sage(
    task_prompt: str,
    config_path: str = "config.yaml",
    working_dir: str = ".",
    is_code_task: bool = True,
    language: str = "python",
    existing_code: str = "",
    existing_tests: str = "",
) -> dict[str, Any]:
    """
    Main entry point for running a GREAT SAGE investigation.

    Args:
        task_prompt: The research question or coding task
        config_path: Path to config.yaml
        working_dir: Working directory for CodeRun tools
        is_code_task: If True, runs SpecPass in Phase 1
        language: Programming language (for code tasks)
        existing_code: Existing code context
        existing_tests: Existing test code
    """
    config = load_config(config_path)
    budget_cfg = config.get("budget", {})
    model_cfg = config.get("model", {})
    threshold_cfg = config.get("thresholds", {})
    snapshot_dir = config.get("snapshot", {}).get("path", "./snapshots/")

    # --- Initialize Harness ---
    state = harness.HarnessState(
        snapshot_dir=snapshot_dir,
        config_dict=threshold_cfg,
        domain_auth_path="domain_authority.yaml",
    )
    run_id = state.get_run_id()

    logger.info(f"=== GREAT SAGE Run {run_id} ===")
    logger.info(f"Task: {task_prompt[:200]}")
    logger.info(f"Worker model: {model_cfg.get('worker', 'default')}")

    # --- Initialize Ollama Client ---
    ollama = OllamaClient(OllamaConfig(
        worker_model=model_cfg.get("worker", "hf.co/TeichAI/Qwen3-4B-Thinking-2507-Claude-4.5-Opus-High-Reasoning-Distill-GGUF:Q3_K_S"),
        skeptic_model=model_cfg.get("skeptic", "nemotron-mini:4b"),
        worker_ctx=model_cfg.get("worker_ctx", 8192),
        skeptic_ctx=model_cfg.get("skeptic_ctx", 4096),
    ))

    # Health check
    if not await ollama.check_health():
        logger.error("Ollama is not running!")
        return {"error": "Ollama server not available", "run_id": run_id}

    total_tokens = 0
    max_rounds = budget_cfg.get("max_rounds", 12)
    round_budget = budget_cfg.get("round_tokens", 25000)
    total_budget = budget_cfg.get("total_tokens", 200000)
    tool_steps = budget_cfg.get("tool_steps", 5)
    tool_timeout = budget_cfg.get("tool_timeout", 30)
    test_code = ""
    prev_eigs: list[tuple[str, float]] = []

    try:
        # ===================================================================
        # Phase 1: Decompose
        # ===================================================================
        logger.info("── Phase 1: Decompose ──")

        if is_code_task:
            spec = await brain.spec_pass(
                ollama, task_prompt, language,
                existing_code, existing_tests,
            )
            hypotheses = spec["hypotheses"]
            test_code = spec.get("test_code", "")
            total_tokens += spec.get("tokens", 0)

            # Write test code to working directory if generated
            if test_code:
                test_path = os.path.join(working_dir, "test_sage_spec.py")
                with open(test_path, "w") as f:
                    f.write(test_code)
                logger.info(f"Test spec written to {test_path}")
        else:
            hypotheses = await brain.decompose(ollama, task_prompt)

        # Insert hypotheses into graph
        hypothesis_ids = {}
        for h in hypotheses:
            hid = state.add_hypothesis(
                text=h["text"],
                priority=h["priority"],
                test_id=h.get("test_id"),
            )
            hypothesis_ids[hid] = h
            logger.info(f"  H[{hid[:8]}] p={h['priority']:.2f}: {h['text'][:80]}")

        logger.info(f"Phase 1: {len(hypotheses)} hypotheses, {total_tokens} tokens")

        # ===================================================================
        # Main Loop: Phases 2-7
        # ===================================================================
        all_tests_pass = False

        for round_id in range(max_rounds):
            round_start = time.monotonic()
            logger.info(f"\n{'='*60}")
            logger.info(f"  ROUND {round_id + 1}/{max_rounds}  (tokens: {total_tokens}/{total_budget})")
            logger.info(f"{'='*60}")

            # ---------------------------------------------------------------
            # Phase 2: Score (pure Harness)
            # ---------------------------------------------------------------
            logger.info("── Phase 2: Score ──")
            allocations_json = state.run_phase_score(round_budget)
            allocations = json.loads(allocations_json)

            if not allocations:
                logger.info("No active branches — forcing convergence")
                break

            for a in allocations:
                logger.info(f"  Branch[{a['id'][:8]}] EIG={a['eig']:.3f} budget={a['budget']}")

            # ---------------------------------------------------------------
            # Phase 3: Execute (Worker LLM + tools)
            # ---------------------------------------------------------------
            logger.info("── Phase 3: Execute ──")

            round_tool_results = {}

            for branch in allocations:
                branch_id = branch["id"]

                # Get hypothesis text
                hyp_json = state.get_hypotheses_json()
                hyp_list = json.loads(hyp_json)
                hyp_text = next(
                    (h["text"] for h in hyp_list if h["meta"]["id"] == branch_id),
                    "Unknown hypothesis",
                )

                # Get verified claims for context pruning (V-05)
                verified_json = state.get_verified_claims_json()

                # Get failing test info
                failing_info = ""
                if test_code:
                    failing_info = f"Test file: test_sage_spec.py\nRun: pytest test_sage_spec.py -v"

                # Plan tool calls
                tool_calls = await brain.plan_execution(
                    ollama, hyp_text, verified_json,
                    failing_info, tool_steps, branch["budget"],
                )
                total_tokens += tool_calls.__len__()  # approximate

                # Execute tools
                branch_results = []
                for tc in tool_calls[:tool_steps]:
                    tc["working_dir"] = tc.get("working_dir", working_dir)
                    tc["timeout_secs"] = min(tc.get("timeout_secs", tool_timeout), tool_timeout)

                    logger.info(f"    Tool: {tc['type']} — {tc.get('command', tc.get('query', tc.get('url', '')))[:60]}")
                    result = await execute_tool(tc)
                    branch_results.append(result)

                    # Check if all tests pass (code convergence)
                    if result.get("type") == "CodeRun":
                        passed = result.get("tests_passed", [])
                        failed = result.get("tests_failed", [])
                        if passed and not failed and result.get("exit_code", -1) == 0:
                            all_tests_pass = True
                            logger.info("    ✓ ALL TESTS PASS — convergence candidate")

                round_tool_results[branch_id] = branch_results

            # ---------------------------------------------------------------
            # Phase 4: Compress (Worker LLM → Harness validates)
            # ---------------------------------------------------------------
            logger.info("── Phase 4: Compress ──")

            for branch_id, results in round_tool_results.items():
                hyp_list = json.loads(state.get_hypotheses_json())
                hyp_text = next(
                    (h["text"] for h in hyp_list if h["meta"]["id"] == branch_id),
                    "",
                )

                graph_summary = json.dumps(state.graph_stats(), indent=2)
                results_json = json.dumps(results, indent=2)

                compress_result = await brain.compress(
                    ollama, branch_id, hyp_text,
                    results_json, graph_summary,
                )
                total_tokens += compress_result.get("tokens", 0)

                delta = compress_result["delta"]
                delta_json = json.dumps(delta)

                # Apply through the Harness — validate + resolve + commit
                success, message, new_ids = state.apply_model_delta(delta_json, 4)

                if success:
                    logger.info(f"  ✓ Applied: {len(new_ids)} new nodes")
                else:
                    logger.warning(f"  ✗ REJECTED: {message}")

            # ---------------------------------------------------------------
            # Phase 5: Audit (pure Harness)
            # ---------------------------------------------------------------
            logger.info("── Phase 5: Audit ──")
            audit_json = state.run_phase_audit()
            audit = json.loads(audit_json)
            logger.info(
                f"  Contradictions: {audit['contradiction_count']}, "
                f"Stale: {audit['stale_count']}, "
                f"Disputed: {audit['disputed_count']}"
            )

            # ---------------------------------------------------------------
            # Phase 6: Skeptic
            # ---------------------------------------------------------------
            logger.info("── Phase 6: Skeptic ──")

            # Snapshot before role swap (§1.3 / §4.6)
            state.save_snapshot()
            await ollama.unload_model(model_cfg.get("worker", "hf.co/TeichAI/Qwen3-4B-Thinking-2507-Claude-4.5-Opus-High-Reasoning-Distill-GGUF:Q3_K_S"))
            await ollama.load_model(model_cfg.get("skeptic", "nemotron-mini:4b"))

            supported_claims = json.loads(state.get_supported_claims_json())
            skeptic_decisions = []
            for claim in supported_claims:
                evidence_text = "\n".join(
                    f"- {e.get('snippet', '')}" for e in claim.get("evidence", [])
                ) or "(none)"
                skeptic_eval = await brain.skeptic_evaluate(
                    ollama,
                    claim_text=claim.get("claim_text", ""),
                    evidence_snippets=evidence_text,
                    hypothesis_text="",
                    use_stub=False,
                )
                total_tokens += skeptic_eval.get("tokens", 0)
                skeptic_decisions.append(
                    {
                        "claim_id": claim.get("claim_id"),
                        "result": skeptic_eval.get("result", "Inconclusive"),
                        "confidence": float(skeptic_eval.get("confidence", 0.5)),
                        "critique": skeptic_eval.get("critique", ""),
                    }
                )

            skeptic_summary = json.loads(state.run_phase_skeptic(json.dumps(skeptic_decisions)))
            logger.info(
                "  Skeptic outcomes: promoted=%s disputed=%s",
                skeptic_summary.get("promoted", 0),
                skeptic_summary.get("disputed", 0),
            )
            await ollama.unload_model(model_cfg.get("skeptic", "nemotron-mini:4b"))
            await ollama.load_model(model_cfg.get("worker", "hf.co/TeichAI/Qwen3-4B-Thinking-2507-Claude-4.5-Opus-High-Reasoning-Distill-GGUF:Q3_K_S"))

            # ---------------------------------------------------------------
            # Phase 7: Monitor
            # ---------------------------------------------------------------
            logger.info("── Phase 7: Monitor ──")

            prev_eigs_json = json.dumps(prev_eigs)
            routing = state.run_phase_monitor(
                total_tokens, all_tests_pass, prev_eigs_json,
            )

            # Update prev_eigs for next round
            prev_eigs = [
                (a["id"], a["eig"]) for a in allocations
            ]

            round_time = time.monotonic() - round_start
            logger.info(f"  Routing: {routing.upper()} ({round_time:.1f}s)")

            # Save snapshot
            state.save_snapshot()

            if routing == "converge":
                logger.info("  → Converging to Phase 9")
                break
            elif routing == "stall":
                logger.info("  → Stalled — running Phase 8 (stub)")
                state.run_phase_reframe_stub()
            else:
                logger.info("  → Continuing to Phase 2")

            state.advance_round()

        # ===================================================================
        # Phase 9: Bank
        # ===================================================================
        logger.info("\n── Phase 9: Bank ──")
        candidates_json = state.run_phase_bank()
        candidates = json.loads(candidates_json)

        stats = state.graph_stats()
        logger.info(f"  Final graph: {stats}")

        if candidates:
            winner = candidates[0]
            logger.info(f"  Winner: {winner['id'][:8]} S_A={winner['s_a']:.3f}")
        else:
            logger.info("  No candidates banked")

        # ===================================================================
        # Phase 10: Final Verify
        # ===================================================================
        logger.info("── Phase 10: Final Verify ──")

        final_verify = {"result": "Pass", "confidence": 0.0, "critique": ""}
        if candidates:
            # Snapshot before role swap (§1.3 / §4.10)
            state.save_snapshot()
            await ollama.unload_model(model_cfg.get("worker", "hf.co/TeichAI/Qwen3-4B-Thinking-2507-Claude-4.5-Opus-High-Reasoning-Distill-GGUF:Q3_K_S"))
            await ollama.load_model(model_cfg.get("skeptic", "nemotron-mini:4b"))

            test_output = ""
            if is_code_task and test_code:
                run_result = await execute_tool(
                    {
                        "type": "CodeRun",
                        "command": "pytest -q test_sage_spec.py",
                        "working_dir": working_dir,
                        "timeout_secs": tool_timeout,
                    }
                )
                test_output = json.dumps(run_result, indent=2)
                if run_result.get("tests_failed") or run_result.get("exit_code", 0) != 0:
                    final_verify = {
                        "result": "Fail",
                        "confidence": 1.0,
                        "critique": "Automatic fail: full test suite did not pass in Final Verify",
                    }

            if final_verify["result"] != "Fail":
                final_verify = await brain.final_verify(
                    client=ollama,
                    candidate_content=json.dumps(candidates[0], indent=2),
                    supporting_claims=state.get_verified_claims_json(),
                    test_output=test_output,
                    task_prompt=task_prompt,
                    use_stub=False,
                )
                total_tokens += final_verify.get("tokens", 0)

            logger.info(
                "  Final Verify result=%s p_fail=%.3f critique=%s",
                final_verify.get("result", "Fail"),
                float(final_verify.get("confidence", 1.0)),
                final_verify.get("critique", "")[:120],
            )
            await ollama.unload_model(model_cfg.get("skeptic", "nemotron-mini:4b"))
            await ollama.load_model(model_cfg.get("worker", "hf.co/TeichAI/Qwen3-4B-Thinking-2507-Claude-4.5-Opus-High-Reasoning-Distill-GGUF:Q3_K_S"))

        # Final snapshot
        final_snapshot = state.save_snapshot()
        logger.info(f"  Snapshot: {final_snapshot}")

        return {
            "run_id": run_id,
            "candidates": candidates,
            "stats": stats,
            "total_tokens": total_tokens,
            "rounds_completed": round_id + 1 if 'round_id' in dir() else 0,
            "all_tests_pass": all_tests_pass,
            "final_verify": final_verify,
            "snapshot_path": final_snapshot,
        }

    finally:
        await ollama.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """CLI entry point for running GREAT SAGE."""
    import argparse

    parser = argparse.ArgumentParser(
        description="GREAT SAGE — Graph-Recursive Evidence-Adaptive Tree Search"
    )
    parser.add_argument("task", help="The research question or coding task")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--workdir", default=".", help="Working directory for code execution")
    parser.add_argument("--code", action="store_true", help="Treat as a code task (use SpecPass)")
    parser.add_argument("--language", default="python", help="Programming language")
    parser.add_argument("--code-file", default="", help="Path to existing code file")
    parser.add_argument("--test-file", default="", help="Path to existing test file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load existing code/tests if paths provided
    existing_code = ""
    existing_tests = ""
    if args.code_file and os.path.exists(args.code_file):
        with open(args.code_file) as f:
            existing_code = f.read()
    if args.test_file and os.path.exists(args.test_file):
        with open(args.test_file) as f:
            existing_tests = f.read()

    result = asyncio.run(run_great_sage(
        task_prompt=args.task,
        config_path=args.config,
        working_dir=args.workdir,
        is_code_task=args.code,
        language=args.language,
        existing_code=existing_code,
        existing_tests=existing_tests,
    ))

    # Print summary
    print(f"\n{'='*60}")
    print(f"  GREAT SAGE Run Complete: {result.get('run_id', 'unknown')}")
    print(f"{'='*60}")
    print(f"  Rounds: {result.get('rounds_completed', 0)}")
    print(f"  Tokens: {result.get('total_tokens', 0)}")
    print(f"  Tests pass: {result.get('all_tests_pass', False)}")
    print(f"  Candidates: {len(result.get('candidates', []))}")

    if result.get("candidates"):
        winner = result["candidates"][0]
        print(f"  Best S_A: {winner.get('s_a', 0):.3f}")

    print(f"  Snapshot: {result.get('snapshot_path', 'none')}")
    print(f"  Graph: {result.get('stats', {})}")


if __name__ == "__main__":
    main()
