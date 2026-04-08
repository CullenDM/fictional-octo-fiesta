# GREAT SAGE Harness (fictional-octo-fiesta)

This repository is an in-progress implementation of the **GREAT SAGE** graph-recursive harness described in:
- `GREAT SAGE Graph Recursive.txt`
- `GREAT_SAGE_v1.1_Spec.docx (1).pdf`

It combines:
- **Rust harness (`harness/`)** for deterministic graph logic, validation, scoring, and snapshots.
- **Python orchestrator (`orchestrator/`)** for model calls, tool execution, and phase control.

---

## Current model defaults

Configured in `config.yaml`:
- **Worker**: `hf.co/TeichAI/Qwen3-4B-Thinking-2507-Claude-4.5-Opus-High-Reasoning-Distill-GGUF:Q3_K_S`
- **Skeptic**: `nemotron-mini:4b`

These can be changed in `config.yaml` at runtime.

---

## What is implemented

- 10-phase orchestration skeleton with active scoring/audit/bank logic.
- Rust-side deterministic validation wall for model deltas (forbidden model writes, provenance checks, etc.).
- Snapshot save/load for graph state.
- Phase 6 skeptic application path (supported claims evaluated externally, applied deterministically).
- Phase 10 final-verify orchestration with code-task test gate.

---

## Security hardening currently in place

The tool layer now applies guardrails:
- **CodeRun command filtering** for dangerous patterns (e.g. destructive shell payloads).
- **Working directory confinement** to repository root.
- **Sanitized subprocess environment** for CodeRun.
- **Visit URL restrictions**:
  - only `http`/`https`
  - blocks loopback/private/local network hosts (SSRF mitigation)

> Note: This remains a development harness. You should still run it in a sandboxed environment.

---

## Quickstart

### 1) Build harness crate

```bash
cd harness
cargo test
```

### 2) Run orchestrator CLI

```bash
python -m orchestrator.loop "your task prompt here" --code --workdir .
```

Useful flags:
- `--config config.yaml`
- `--language python`
- `--code-file path/to/file.py`
- `--test-file path/to/test_file.py`
- `-v` for verbose logs

---

## Repository layout

- `harness/src/` — Rust graph/schema/validator/scorer/phases/snapshot modules
- `orchestrator/` — Python loop, prompts, model client, tool executor
- `config.yaml` — runtime budgets/models/thresholds
- `domain_authority.yaml` — source authority configuration

---

## Status

This project is still under active construction toward full fidelity with the GREAT SAGE document.
Some advanced sections (e.g., full phase completeness and complete candidate reopen micro-round behavior)
are not yet fully end-to-end.
