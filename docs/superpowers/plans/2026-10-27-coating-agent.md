# Coating Supply Agent Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans. Steps use checkboxes for tracking.

**Goal:** Deliver a reproducible, simulated coating-supply maintenance agent with evidence-based diagnosis, controlled actions, forecasting, optimization, and evaluation.

**Architecture:** FastAPI serves isolated in-memory simulator sessions and a local HTML interface; LangGraph orchestrates bounded read-only diagnostic tools and structured evidence. A simulator, knowledge index, risk training pipeline and safety gate are separate modules; SQLite stores audit and reports.

**Tech Stack:** Python 3.12, FastAPI, Pydantic, LangGraph, OpenAI-compatible provider optional, scikit-learn/XGBoost, NumPy, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-10-27-coating-supply-agent-design.md`

## Global Constraints

- Only synthetic equipment and data; no PLC, real device connectors, real-world operating advice or claims.
- Fix seeds, isolate truth from runtime observations, enforce simulation clock, train/test split by asset.
- Any simulator state-changing action needs explicit one-time confirmation; validate version, safety limits and audit.
- Local deterministic diagnosis fallback is visibly labeled; an optional actual LLM invokes read-only tools only.
- Do not commit or initialize Git without a need: current project is not a Git repository.

## Review Focus

- Future timestamps must be rejected; no latent state or labels in public responses.
- Confirmation reuse, refusal, expiration, invalid state and version conflict must not change simulator state.
- Actuation must branch from the current state; verification uses subsequent non-overlapping windows.
- Missing observations and unrecognized causes must yield uncertainty, never invented citations.
- Risk training must exclude event-after windows, future leakage, and censored horizon windows.

## Files and tasks

### Task 1: Synthetic simulator and observation API

**Files:** `pyproject.toml`, `src/coating_agent/simulator.py`, `src/coating_agent/detection.py`, `tests/test_simulator.py`.

**Interfaces:** `Simulator(seed, scenario).advance(n)`, `.window(end, size)`, `.snapshot()`, `.clean(success)`, `detect(window)`.

- [x] Write failing tests proving seed reproducibility, future rejection, public truth exclusion, anomaly detection and different simulator branches after cleaning.
- [x] Implement smooth normalized pressure and flow, scenarios, private hidden state, replay clock and rules; run targeted tests.

### Task 2: Evidence, retrieval and diagnosis workflow

**Files:** `knowledge/cards.json`, `src/coating_agent/knowledge.py`, `src/coating_agent/diagnosis.py`, `tests/test_diagnosis.py`.

**Interfaces:** `search(query, k)`, `diagnose(window, mode)` returning structured evidence and limitations; LangGraph governs bounded tools.

- [x] Write failing tests for retrieval, evidence IDs, missing data and bias vs blockage uncertainty.
- [x] Implement source-tagged knowledge cards, BM25, deterministic diagnosis and optional LLM tool calling with bounded calls and validated references; run targeted tests.

### Task 3: Confirmation, audit and end-to-end HTTP interface

**Files:** `src/coating_agent/service.py`, `src/coating_agent/api.py`, `src/coating_agent/static/index.html`, `tests/test_workflow.py`.

**Interfaces:** `Service.create`, `.advance`, `.diagnose`, `.request_action`, `.confirm_action`, `.verify`, `.report`; FastAPI endpoints expose these.

- [x] Write failing tests for state, refusal/reuse/expiry/version conflict, branch continuation, verification and audit.
- [x] Implement single-use authorization, whitelist and preconditions, SQLite event audit, dashboard, API and downloadable report; run workflow tests.

### Task 4: Predictive maintenance experiment

**Files:** `src/coating_agent/predict.py`, `scripts/train.py`, `tests/test_predict.py`.

**Interfaces:** `label(t, event_time, last_time, horizon)`, `features(window)`, grouped train/validation/test; fitted prediction returned with training domain and version.

- [x] Write failing tests for censoring, event exclusion, temporal feature isolation and asset split.
- [x] Generate trajectories, fit rule/logistic/XGBoost, select threshold on validation, report held-out metrics and persist fitted artifact; wire prediction into API; run targeted tests.

### Task 5: Constrained simulated experiment

**Files:** `src/coating_agent/optimize.py`, `tests/test_optimize.py`.

**Interfaces:** `optimize(snapshot, target, seed)` compares a constrained pump-speed grid using identical noise per candidate without mutating the live simulator.

- [x] Write failing tests for constraints, no feasible candidate, same-noise comparison and state isolation.
- [x] Implement paired seed experiments, baseline and per-candidate objective/violations; expose experiment in API; run targeted tests.

### Task 6: Evaluation and documentation

**Files:** `scripts/evaluate.py`, `tests/test_evaluation.py`, `README.md`, `docs/evaluation.md`, `docs/knowledge-sources.md`.

- [x] Write tests that evaluation reports retrieval, diagnosis, detection and end-to-end failure cases without using hidden labels as input.
- [x] Implement evaluation report and scripts, run full suite, train/evaluate, launch API and execute successful and failure scenarios; document actual commands, limits and observed results.
- [x] Review design conformance and security, repair any material findings and rerun affected tests.
