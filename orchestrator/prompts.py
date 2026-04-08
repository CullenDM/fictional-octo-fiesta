"""
Prompt templates for GREAT SAGE phases (§8).

Each function renders a structured prompt with Harness-injected values.
The model receives these prompts and returns structured JSON.
Variables in {CURLY_BRACES} are substituted by the Harness/Orchestrator.
"""

# ---------------------------------------------------------------------------
# §8.0 Decompose (Phase 1)
# ---------------------------------------------------------------------------

def render_decompose_prompt(
    task_prompt: str,
    max_hypotheses: int = 5,
) -> str:
    return f"""You are a research decomposition engine.  You are given a task/query.
Break it into {max_hypotheses} or fewer falsifiable hypotheses.
Each hypothesis must be a specific, testable statement.

TASK:
{task_prompt}

Respond in JSON:
{{
  "hypotheses": [
    {{
      "text": "A falsifiable statement (1-3 sentences)",
      "priority": 0.0 to 1.0 (how likely to yield the answer)
    }}
  ]
}}

Rules:
- Each hypothesis must be independently testable
- Order by expected information gain (most promising first)
- Hypotheses should cover different approaches, not restate the same idea
- Be specific — "The API supports X" not "There might be something useful"
"""


# ---------------------------------------------------------------------------
# §8.1 SpecPass (Phase 1, Code Agent) — §7.3
# ---------------------------------------------------------------------------

def render_spec_pass_prompt(
    task_prompt: str,
    language: str = "python",
    existing_code: str = "",
    existing_tests: str = "",
) -> str:
    code_section = f"\nEXISTING CODE:\n```{language}\n{existing_code}\n```" if existing_code else ""
    test_section = f"\nEXISTING TESTS:\n```{language}\n{existing_tests}\n```" if existing_tests else ""

    return f"""You are a test-driven development engine.  Given a coding task, generate
a test specification FIRST, then derive hypotheses from failing tests.

TASK:
{task_prompt}

LANGUAGE: {language}
{code_section}
{test_section}

Respond in JSON:
{{
  "test_code": "Complete test file content (pytest for Python)",
  "hypotheses": [
    {{
      "text": "What must be true for test_X to pass",
      "priority": 0.0 to 1.0,
      "test_id": "test_function_name"
    }}
  ]
}}

Rules:
- Write real, runnable tests (not pseudocode)
- Each test should verify ONE specific behavior
- Derive one hypothesis per test that captures what must be implemented
- Tests should be independent of each other
- Priority reflects how foundational the test is (core logic > edge cases)
"""


# ---------------------------------------------------------------------------
# §8.2 Execute (Phase 3)
# ---------------------------------------------------------------------------

def render_execute_prompt(
    hypothesis_text: str,
    verified_claims: str,
    failing_tests: str,
    tool_budget: int,
    token_budget: int,
) -> str:
    return f"""You are a research execution engine.  Given a hypothesis and known facts,
use tools to gather evidence.

HYPOTHESIS: {hypothesis_text}

VERIFIED FACTS (do not re-investigate):
{verified_claims if verified_claims else "(none yet)"}

FAILING TESTS:
{failing_tests if failing_tests else "(none)"}

BUDGET: {tool_budget} tool calls, {token_budget} tokens

Respond in JSON:
{{
  "tool_calls": [
    {{
      "type": "CodeRun",
      "command": "shell command to run",
      "working_dir": ".",
      "timeout_secs": 30
    }}
  ]
}}

Available tool types:
- CodeRun: Execute a shell command (run tests, inspect files, compile code)
- Search: Web search for information
- Visit: Fetch and parse a URL

Rules:
- Focus tool calls on the hypothesis — don't investigate unrelated topics
- When tests are failing, prioritize running those tests first
- Prefer specific, targeted commands over broad exploration
- Each tool call should have a clear purpose toward confirming or refuting the hypothesis
"""


# ---------------------------------------------------------------------------
# §8.3 Compress (Phase 4)
# ---------------------------------------------------------------------------

def render_compress_prompt(
    hypothesis_text: str,
    tool_results_json: str,
    graph_summary: str,
) -> str:
    return f"""You are an evidence compression engine.  Given tool execution results,
extract new claims and evidence for the knowledge graph.

HYPOTHESIS: {hypothesis_text}

TOOL RESULTS:
{tool_results_json}

CURRENT GRAPH STATE:
{graph_summary}

Respond in JSON with a model delta:
{{
  "new_claims": [
    {{
      "tmp_id": "tmp_claim_1",
      "text": "Specific factual statement supported by evidence"
    }}
  ],
  "new_evidence": [
    {{
      "tmp_id": "tmp_evidence_1",
      "snippet": "Exact text or output that supports the claim",
      "source_url": "file:///path or https://url",
      "quality_hint": "TestVerified|Primary|Corroborated|Inferred"
    }}
  ],
  "new_edges": [
    {{
      "from_id": "tmp_evidence_1",
      "to_id": "tmp_claim_1",
      "kind": "Supports"
    }},
    {{
      "from_id": "tmp_claim_1",
      "to_id": "HYPOTHESIS_UUID_HERE",
      "kind": "Supports"
    }}
  ]
}}

Rules:
- Each claim MUST be supported by at least one evidence node via a Supports edge
- Use "TestVerified" quality only if a test explicitly passed
- Use tmp_claim_N, tmp_evidence_N, tmp_source_N for new nodes
- Reference existing hypothesis UUIDs directly (not tmp_*)
- Extract precise snippets — quotes from output, not summaries
- Do NOT include any of these fields: id, created_at, belief_score,
  source_entropy, domain_auth, reliability, is_stale, status
  These are computed by the Harness — writing them will REJECT the entire batch
"""


# ---------------------------------------------------------------------------
# §8.4 Skeptic (Phase 6) — Stage 2 only
# ---------------------------------------------------------------------------

def render_skeptic_prompt(
    claim_text: str,
    evidence_snippets: str,
    hypothesis_text: str,
) -> str:
    return f"""You are an adversarial skeptic evaluating a factual claim.
Your job is to find flaws, gaps, and unsupported inferences.

CLAIM: {claim_text}

SUPPORTING EVIDENCE:
{evidence_snippets}

HYPOTHESIS CONTEXT: {hypothesis_text}

Respond in JSON:
{{
  "result": "Pass" or "Fail" or "Inconclusive",
  "confidence": 0.0 to 1.0 (p_fail: probability the claim is wrong),
  "critique": "One paragraph explaining your reasoning (max 300 chars)",
}}

Rules:
- Pass: Evidence clearly supports the claim
- Fail: Evidence contradicts or is insufficient
- Inconclusive: Cannot determine from available evidence
- confidence MUST be p_fail (probability claim is wrong)
- Keep critique concise and specific
"""


# ---------------------------------------------------------------------------
# §8.4.5 Synthesize (pre-Phase 9) — generate implementation from Verified claims
# ---------------------------------------------------------------------------

def render_synthesize_prompt(
    task_prompt: str,
    verified_claims_json: str,
    language: str = "python",
    test_code: str = "",
) -> str:
    test_section = f"\nTESTS TO PASS:\n```{language}\n{test_code}\n```" if test_code else ""
    return f"""You are a code synthesis engine.  Given verified factual claims about what
an implementation must do, write the implementation that satisfies all claims.

TASK:
{task_prompt}

VERIFIED CLAIMS (all must hold in your implementation):
{verified_claims_json}
{test_section}

Respond in JSON:
{{
  "implementation": "complete file content as a string",
  "filename": "relative path to write (e.g. solution.py)"
}}

Rules:
- Write a complete, runnable file — not pseudocode, not snippets
- Every Verified claim must be satisfied by the implementation
- If tests are provided, the implementation must pass all of them
- Output ONLY the JSON object — no markdown fences, no explanation outside the JSON
- Do NOT include id, created_at, belief_score, or any harness fields
"""


# ---------------------------------------------------------------------------
# §8.5 Reframe (Phase 8) — Stage 2 only
# ---------------------------------------------------------------------------

def render_reframe_prompt(
    task_prompt: str,
    refuted_hypotheses: str,
    stalled_hypotheses: str,
    graph_summary: str,
) -> str:
    return f"""You are a research strategy reframing engine.  Previous hypotheses have
stalled or been refuted.  Propose new approaches.

ORIGINAL TASK:
{task_prompt}

REFUTED/STALLED HYPOTHESES:
{refuted_hypotheses}
{stalled_hypotheses}

CURRENT KNOWLEDGE:
{graph_summary}

Respond in JSON:
{{
  "hypotheses": [
    {{
      "text": "A new falsifiable hypothesis (must be different from previous)",
      "priority": 0.0 to 1.0,
      "rationale": "Why this approach differs from what was tried"
    }}
  ]
}}

Rules:
- New hypotheses must be genuinely different (cosine distance > 0.3 from previous)
- Learn from failures — don't repeat approaches that didn't work
- Consider orthogonal decompositions of the problem
"""


# ---------------------------------------------------------------------------
# §8.6 Final Verify (Phase 10) — Stage 2 only
# ---------------------------------------------------------------------------

def render_final_verify_prompt(
    candidate_content: str,
    supporting_claims: str,
    test_output: str,
    task_prompt: str,
) -> str:
    return f"""You are performing a final adversarial verification of a proposed answer.

ORIGINAL TASK:
{task_prompt}

CANDIDATE ANSWER:
{candidate_content}

SUPPORTING CLAIMS:
{supporting_claims}

TEST OUTPUT:
{test_output if test_output else "(no tests run)"}

Respond in JSON:
{{
  "result": "Pass" or "Fail",
  "confidence": 0.0 to 1.0 (p_fail: probability candidate is wrong),
  "critique": "Explanation if Fail (max 300 chars)",
  "contested_claims": ["claim_uuid_1", "claim_uuid_2"]
}}

Rules:
- Pass only if ALL supporting claims are consistent with the answer
- Check for logical gaps, unsupported leaps, and contradictions
- If tests exist and pass, weigh that heavily (but not conclusively)
- confidence MUST be p_fail (probability candidate is wrong)
"""
