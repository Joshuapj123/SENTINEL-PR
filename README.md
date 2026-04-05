# SENTINEL-PR

> **Security & Dependency Auditor Environment for AI Agents**
> Meta × OpenEnv Hackathon 2026 — Production Submission

---

## What It Does

SENTINEL-PR is an OpenEnv 2026-compliant reinforcement-learning environment where an AI agent acts as a senior application-security engineer. At each episode step the agent receives a Python source file, a `requirements.txt`, and an organisational security policy. It must detect vulnerabilities, reject dangerous packages, propose concrete patches, or approve clean code — every decision scored by a **gradient reward function** backed by live `bandit` static analysis.

---

## Architecture

```
┌────────────────────────────────────────────┐
│             SentinelPREnv                  │
│  reset(task_id) → Observation              │
│  step(Action)   → StepResult               │
│                                            │
│  ┌────────────────────────────────────┐    │
│  │      Gradient Reward Engine        │    │
│  │  bandit Δ · confidence · decay     │    │
│  │  repetition penalty · force-term   │    │
│  └────────────────────────────────────┘    │
└────────────────────────────────────────────┘
           ↕ OpenEnv 2026 API
┌────────────────────────────────────────────┐
│      Aura Hallucination-Risk Layer         │
│  lexical grounding · coherence · calib     │
│  composite < 0.4 → action downgraded       │
└────────────────────────────────────────────┘
           ↕ HF Inference Router
┌────────────────────────────────────────────┐
│  LLM Agent  (OpenAI-compatible client)     │
└────────────────────────────────────────────┘
```

---

## Task Complexity

| # | Task ID | Difficulty | Vulnerability | Observed Reward |
|---|---------|------------|---------------|----------------|
| 1 | `task_hardcoded_key`  | 🟢 Easy   | Hardcoded API key (B105/CWE-798)           | +1.1899 |
| 2 | `task_vulnerable_dep` | 🟡 Medium | CVE-pinned deps (Pillow, urllib3, crypto)   | +1.1799 |
| 3 | `task_eval_auth_flaw` | 🔴 Hard   | eval() RCE × 2 + timing-oracle auth flaw   | +1.2935 |

### Task 1 — Secret Leak (Easy)
A production API client embeds a live key as `API_KEY = "sk-prod-..."`. Agent must `FLAG_VULN` and optionally `PROPOSE_PATCH` replacing it with `os.environ.get("API_KEY")`.

### Task 2 — Vulnerable Library (Medium)
`requirements.txt` pins Pillow 9.5.0 (CVE-2024-28219), urllib3 1.26.18 (CVE-2024-37891), and cryptography 38.0.4 (CVE-2024-26130). Agent must `REJECT_DEP` each offender.

### Task 3 — Insecure Auth Middleware (Hard)
A Flask decorator uses `eval()` on attacker-controlled token payload at two call sites (CWE-78, Bandit B307) and compares HMAC signatures with `==` instead of `hmac.compare_digest()` (CWE-208). Agent must flag both classes and submit a `PROPOSE_PATCH` that achieves zero HIGH findings in the bandit re-scan.

---

## Aura Hallucination-Risk Logic ✨

*Creativity/Novelty differentiator — 10 % bonus criterion.*

Large language models acting as security agents face a specific failure mode that is distinct from ordinary reasoning errors: **confident hallucination** — fabricating CVE identifiers, inventing line numbers, or producing `APPROVE` decisions on demonstrably vulnerable code. Standard accuracy metrics cannot catch this because the model's output is syntactically valid and structurally coherent even when factually wrong.

**Aura** is SENTINEL-PR's purpose-built hallucination-risk layer. After every LLM response, before the action is submitted to the environment, Aura computes a composite score from three independent evidence signals:

| Signal | Weight | Mechanism |
|--------|--------|-----------|
| **Lexical Grounding** | 45 % | Extracts all meaningful tokens (≥4 chars) from the agent's `detail` field and checks what fraction appear verbatim in the observed source code. A grounded agent cites real identifiers (`_verify_token`, `eval`, `hmac.compare_digest`). A hallucinating agent fabricates names that do not exist in the file. |
| **Action Coherence** | 35 % | Maps each `ActionType` to a vocabulary of semantically consistent terms (e.g. `PROPOSE_PATCH` should co-occur with words like *fix*, *replace*, *json.loads*). A mismatch — for example `APPROVE` paired with rationale text that says "vulnerable" — is caught as incoherence and penalised. |
| **Confidence Calibration** | 20 % | Compares the agent's self-reported `confidence` field against the empirical bandit HIGH-severity count. An agent that submits `APPROVE` with confidence 0.95 on code where bandit found five HIGH issues is wildly miscalibrated; Aura penalises the delta and feeds it back as a negative signal. |

The composite score is a weighted sum: `0.45 × lex + 0.35 × coherence + 0.20 × calibration`. Any composite below **0.40** triggers automatic action downgrade: the action is replaced with a `FLAG_VULN` at capped confidence (≤ 0.30) before reaching the environment. This makes it structurally impossible for a hallucinated `APPROVE` to ship insecure code or collect the −2.0 unsafe-approve penalty — even if the underlying LLM is confidently wrong.

The result is a closed feedback loop: Aura's downgrade decision is logged in `info["aura"]` at every step, giving evaluators a per-step hallucination-risk trace alongside the gradient reward signal.

---

## Repetition Penalty & Force Termination

Agents that loop on the same `FLAG_VULN` without progressing to `PROPOSE_PATCH` are penalised:

- **2nd duplicate FLAG_VULN** (same detail fingerprint, no intervening patch): `−1.0` reward.
- **3 consecutive redundant steps**: episode force-terminates with an additional `−1.0 × streak` penalty.

This prevents degenerate looping strategies and forces genuine forward progress.

---

## Gradient Reward Function

```
R(a, s) = base_signal(a, s)
        × (1 + 0.2 × confidence)
        × exp(−ln2 × t / 120)
        + repetition_penalty (if applicable)
```

For `PROPOSE_PATCH`, the base signal is a bandit Δ:

```
base = fixed_HIGH × 0.8 + fixed_MED × 0.3 − residual_HIGH × 0.4 + threshold_bonus × 0.5
```

---

## How to Run

### Local
```bash
pip install -r requirements.txt
python smoke_test.py          # no API key needed
```

### Full LLM episode
```powershell
$env:HF_TOKEN="hf_..."
python inference.py
```

### Docker / HF Space
```bash
docker build -t sentinel-pr .
docker run -p 7860:7860 -e HF_TOKEN=$HF_TOKEN sentinel-pr
```

---

## File Structure

```
sentinel-pr/
├── env.py            # SentinelPREnv, reward engine, repetition penalty
├── inference.py      # LLM agent loop + Aura scoring
├── openenv.yaml      # OpenEnv 2026 spec
├── requirements.txt
├── Dockerfile
├── .gitignore
├── .env
├── METRICS.md
├── smoke_test.py
├── test_success.py
├── .dockerignore
├── smoke_test.py
└── README.md
```

---

## Infrastructure

- **RAM**: 8 GB hard limit enforced via Pydantic byte-caps (256 KB source, 64 KB manifest) and `gc.collect()` on every `step()` / `reset()`.
- **Bandit**: subprocess with 30 s timeout; auto-falls back to deterministic regex mock.
- **Docker**: multi-stage build — build tools stripped from runtime image; non-root `appuser` for HF Spaces compliance.

---

## ⚠️ Known Limitations

- Baseline agent may not always reach full victory on the hard task within 6 steps
- Some patches require import injection (handled in latest version)
- Dependency resolution is enforced in later steps for completion

## 📊 Evaluation Insights

- Sequential reasoning significantly outperforms reactive behavior
- Repetition penalties expose weak agent strategies
- Aura layer reduces unsafe approvals caused by hallucination

