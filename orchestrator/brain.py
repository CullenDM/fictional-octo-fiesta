"""
LLM call sites for GREAT SAGE phases.

Each function here is a call site that:
1. Renders a prompt template
2. Calls Ollama via the async client
3. Validates the JSON response schema
4. Returns a typed result

These are the ONLY places the orchestrator talks to the model.
All numeric computation happens in the Rust harness.
"""

import json
import logging
from typing import Any

from .ollama_client import OllamaClient
from . import prompts

logger = logging.getLogger("great_sage.brain")


# ---------------------------------------------------------------------------
# Phase 1: Decompose
# ---------------------------------------------------------------------------

async def decompose(
    client: OllamaClient,
    task_prompt: str,
    max_hypotheses: int = 5,
) -> list[dict[str, Any]]:
    """
    Phase 1: Break task into falsifiable hypotheses.
    Returns list of {text, priority} dicts.
    """
    prompt = prompts.render_decompose_prompt(task_prompt, max_hypotheses)

    result = await client.generate_json(
        prompt=prompt,
        role="worker",
        system_prompt="You are GREAT SAGE's research decomposition engine.",
        temperature=0.4,
    )

    parsed = result.get("parsed", {})
    hypotheses = parsed.get("hypotheses", [])

    # Validate and normalize
    validated = []
    for h in hypotheses:
        if isinstance(h, dict) and "text" in h:
            validated.append({
                "text": str(h["text"]),
                "priority": float(h.get("priority", 0.5)),
                "test_id": h.get("test_id"),
            })

    logger.info(f"Decompose: {len(validated)} hypotheses, {result.get('tokens', 0)} tokens")
    return validated


# ---------------------------------------------------------------------------
# Phase 1: SpecPass (Code Agent)
# ---------------------------------------------------------------------------

async def spec_pass(
    client: OllamaClient,
    task_prompt: str,
    language: str = "python",
    existing_code: str = "",
    existing_tests: str = "",
) -> dict[str, Any]:
    """
    Phase 1 (code tasks): Generate test spec, derive hypotheses from tests.
    Returns {test_code, hypotheses: [{text, priority, test_id}]}.
    """
    prompt = prompts.render_spec_pass_prompt(
        task_prompt, language, existing_code, existing_tests
    )

    result = await client.generate_json(
        prompt=prompt,
        role="worker",
        system_prompt="You are GREAT SAGE's test-driven development engine.",
        temperature=0.3,
    )

    parsed = result.get("parsed", {})

    return {
        "test_code": parsed.get("test_code", ""),
        "hypotheses": [
            {
                "text": str(h.get("text", "")),
                "priority": float(h.get("priority", 0.5)),
                "test_id": h.get("test_id"),
            }
            for h in parsed.get("hypotheses", [])
            if isinstance(h, dict)
        ],
        "tokens": result.get("tokens", 0),
    }


# ---------------------------------------------------------------------------
# Phase 3: Execute
# ---------------------------------------------------------------------------

async def plan_execution(
    client: OllamaClient,
    hypothesis_text: str,
    verified_claims_json: str,
    failing_tests: str,
    tool_budget: int,
    token_budget: int,
) -> list[dict[str, Any]]:
    """
    Phase 3: Plan tool calls for a hypothesis branch.
    Returns list of tool call dicts.
    """
    prompt = prompts.render_execute_prompt(
        hypothesis_text, verified_claims_json, failing_tests,
        tool_budget, token_budget
    )

    result = await client.generate_json(
        prompt=prompt,
        role="worker",
        system_prompt="You are GREAT SAGE's research execution engine.",
        temperature=0.3,
    )

    parsed = result.get("parsed", {})
    tool_calls = parsed.get("tool_calls", [])

    # Validate tool calls
    validated = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        tc_type = tc.get("type", "")
        if tc_type == "CodeRun":
            validated.append({
                "type": "CodeRun",
                "command": str(tc.get("command", "")),
                "working_dir": str(tc.get("working_dir", ".")),
                "timeout_secs": int(tc.get("timeout_secs", 30)),
            })
        elif tc_type == "Search":
            validated.append({
                "type": "Search",
                "query": str(tc.get("query", "")),
            })
        elif tc_type == "Visit":
            validated.append({
                "type": "Visit",
                "url": str(tc.get("url", "")),
            })

    logger.info(f"Execute plan: {len(validated)} tool calls, {result.get('tokens', 0)} tokens")
    return validated


# ---------------------------------------------------------------------------
# Phase 4: Compress
# ---------------------------------------------------------------------------

async def compress(
    client: OllamaClient,
    hypothesis_id: str,
    hypothesis_text: str,
    tool_results_json: str,
    graph_summary: str,
) -> dict[str, Any]:
    """
    Phase 4: Compress tool results into a model delta.
    Returns the raw delta dict (to be validated by the Harness).
    """
    prompt = prompts.render_compress_prompt(
        hypothesis_text, tool_results_json, graph_summary
    )

    # Inject the real hypothesis UUID for edge targets
    prompt = prompt.replace("HYPOTHESIS_UUID_HERE", hypothesis_id)

    result = await client.generate_json(
        prompt=prompt,
        role="worker",
        system_prompt=(
            "You are GREAT SAGE's evidence compression engine. "
            "FORBIDDEN: Do NOT include id, created_at, belief_score, "
            "source_entropy, domain_auth, reliability, is_stale, or status "
            "in your output. The Harness computes these."
        ),
        temperature=0.2,
    )

    parsed = result.get("parsed", {})

    # Ensure required structure exists
    delta = {
        "new_claims": parsed.get("new_claims", []),
        "new_evidence": parsed.get("new_evidence", []),
        "new_sources": parsed.get("new_sources", []),
        "new_edges": parsed.get("new_edges", []),
    }

    logger.info(
        f"Compress: {len(delta['new_claims'])} claims, "
        f"{len(delta['new_evidence'])} evidence, "
        f"{len(delta['new_edges'])} edges, "
        f"{result.get('tokens', 0)} tokens"
    )

    return {
        "delta": delta,
        "tokens": result.get("tokens", 0),
    }


# ---------------------------------------------------------------------------
# Phase 6: Skeptic (Stage 1 = stub, Stage 2 = real)
# ---------------------------------------------------------------------------

async def skeptic_evaluate(
    client: OllamaClient,
    claim_text: str,
    evidence_snippets: str,
    hypothesis_text: str,
    use_stub: bool = True,
) -> dict[str, Any]:
    """
    Phase 6: Adversarial evaluation of a claim.
    Stage 1 stub: always returns Pass with confidence 0.1
    """
    if use_stub:
        return {
            "verdict": "Pass",
            "confidence": 0.1,
            "critique": "Stage 1 stub — Skeptic auto-pass",
            "p_fail": 0.0,
            "tokens": 0,
        }

    prompt = prompts.render_skeptic_prompt(
        claim_text, evidence_snippets, hypothesis_text
    )

    result = await client.generate_json(
        prompt=prompt,
        role="skeptic",
        system_prompt="You are an adversarial skeptic. Find flaws.",
        temperature=0.5,
    )

    parsed = result.get("parsed", {})

    return {
        "verdict": parsed.get("verdict", "Inconclusive"),
        "confidence": float(parsed.get("confidence", 0.5)),
        "critique": str(parsed.get("critique", ""))[:300],
        "p_fail": float(parsed.get("p_fail", 0.5)),
        "tokens": result.get("tokens", 0),
    }


# ---------------------------------------------------------------------------
# Phase 10: Final Verify (stub)
# ---------------------------------------------------------------------------

async def final_verify(
    client: OllamaClient,
    candidate_content: str,
    supporting_claims: str,
    test_output: str,
    task_prompt: str,
    use_stub: bool = True,
) -> dict[str, Any]:
    """
    Phase 10: Final adversarial verification.
    Stage 1 stub: always passes.
    """
    if use_stub:
        return {
            "verdict": "Pass",
            "confidence": 1.0,
            "critique": "",
            "p_fail": 0.0,
            "tokens": 0,
        }

    prompt = prompts.render_final_verify_prompt(
        candidate_content, supporting_claims, test_output, task_prompt
    )

    result = await client.generate_json(
        prompt=prompt,
        role="skeptic",
        system_prompt="You are performing final adversarial verification.",
        temperature=0.5,
    )

    parsed = result.get("parsed", {})
    return {
        "verdict": parsed.get("verdict", "Fail"),
        "confidence": float(parsed.get("confidence", 0.5)),
        "critique": str(parsed.get("critique", ""))[:300],
        "p_fail": float(parsed.get("p_fail", 0.5)),
        "contested_claims": parsed.get("contested_claims", []),
        "tokens": result.get("tokens", 0),
    }
