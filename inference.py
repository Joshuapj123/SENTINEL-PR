"""
inference.py  |  SENTINEL-PR v4.0.0  |  Strict deterministic runner

Logging format: [START], [STEP N], [END]
- temperature=0 always
- Retry once on parse failure (explicit error, no silent downgrade)
- JSON-only output enforced via system prompt
- Same run → same score guaranteed
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import textwrap
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from pydantic import ValidationError

# ── Load .env ─────────────────────────────────────────────────────────────────
def _load_dotenv(path: str = ".env") -> None:
    import pathlib
    p = pathlib.Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip(); v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v

_load_dotenv()

from env import (
    Action, ActionType, FlowPhase, SentinelPREnv,
    Observation, StepResult, R_PARSE_FAIL,
)

# ── Config ────────────────────────────────────────────────────────────────────
PROVIDER     : str = os.environ.get("SENTINEL_PROVIDER", "hf").lower()
HF_TOKEN     : str = os.environ.get("HF_TOKEN", "")
HF_MODEL     : str = os.environ.get("HF_MODEL", "meta-llama/Llama-3.2-3B-Instruct:hyperbolic")
HF_BASE_URL  : str = "https://router.huggingface.co/v1"
GROQ_KEY     : str = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL   : str = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
MAX_STEPS    : int = int(os.environ.get("SENTINEL_MAX_STEPS", "12"))
TEMPERATURE  : float = 0.0   # ALWAYS zero – determinism guaranteed

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("sentinel_pr.inference")


# ── Client ────────────────────────────────────────────────────────────────────
def _build_client() -> OpenAI:
    if PROVIDER == "groq":
        if not GROQ_KEY:
            raise RuntimeError("GROQ_API_KEY not set. Get free key at https://console.groq.com")
        return OpenAI(api_key=GROQ_KEY, base_url=GROQ_BASE_URL)
    if PROVIDER == "openai":
        key = os.environ.get("OPENAI_API_KEY","")
        if not key: raise RuntimeError("OPENAI_API_KEY not set.")
        return OpenAI(api_key=key)
    # default: hf
    if not HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN not set. Add to .env:\n"
            "  HF_TOKEN=hf_your_token_here\n"
            "  HF_MODEL=meta-llama/Llama-3.2-3B-Instruct:cerebras\n"
            "Get token: https://huggingface.co/settings/tokens\n"
            "Enable: Make calls to Inference Providers"
        )
    return OpenAI(api_key=HF_TOKEN, base_url=HF_BASE_URL)


# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM = textwrap.dedent("""\
    You are SENTINEL, a deterministic AI security auditor.
    Output EXACTLY ONE raw JSON object. No markdown. No text before or after.
    First char = {  Last char = }

    ══════════════════════════════════════════════════
    OUTPUT SCHEMA (all fields required):
    ══════════════════════════════════════════════════
    {
      "action_type": "FLAG_VULN|PROPOSE_PATCH|REJECT_DEP|REJECT|APPROVE",
      "confidence": 0.9,
      "evidence": ["line 12: payload = eval(payload_str)"],
      "detail": "eval() on attacker input at line 12 – RCE (B307)",
      "package": null,
      "patched_source": null,
      "patched_manifest": null,
      "fix_rationale": null,
      "reasoning": null,
      "resolved_issues": {},
      "remaining_issues": {},
      "validation_results": {"syntax": true, "security": false, "runtime": true},
      "final_decision": null
    }

    ══════════════════════════════════════════════════
    MANDATORY DECISION FLOW — FOLLOW EXACTLY:
    ══════════════════════════════════════════════════
    Read `current_phase` and `allowed_actions` in the observation.
    ONLY use actions listed in `allowed_actions`.

    PHASE CRITICAL → allowed: FLAG_VULN, PROPOSE_PATCH
      Step A: FLAG_VULN to identify eval()/pickle issue with exact line number
      Step B: PROPOSE_PATCH with full corrected source + fix_rationale

    PHASE HIGH → allowed: FLAG_VULN, PROPOSE_PATCH
      Same pattern as CRITICAL

    PHASE DEP → allowed: REJECT_DEP, PROPOSE_PATCH
      Option A: REJECT_DEP with package name for each CVE
      Option B: PROPOSE_PATCH with patched_manifest fixing all CVEs

    PHASE TERMINAL → allowed: APPROVE, REJECT
      If open_issues is empty → APPROVE with full reasoning
      If issues remain → REJECT with full reasoning

    ══════════════════════════════════════════════════
    FIELD RULES:
    ══════════════════════════════════════════════════
    FLAG_VULN:
      - detail: required, non-empty
      - evidence: required, list with line numbers

    PROPOSE_PATCH:
      - patched_source: FULL corrected Python file (not a diff)
      - fix_rationale: WHY this fix (json.loads vs ast.literal_eval etc.)
      - evidence: cite original vulnerable lines

    REJECT_DEP:
      - package: exact package name (lowercase, no version)

    APPROVE / REJECT:
      - reasoning: "CRITICAL: [ids] FIXED. HIGH: [ids] FIXED/NOT. DEP: [ids] FIXED/NOT. Therefore APPROVE/REJECT."
      - resolved_issues: {"CRITICAL": [...], "HIGH": [...], "DEP": [...]}
      - remaining_issues: {} for APPROVE, non-empty for REJECT
      - final_decision: one sentence

    ══════════════════════════════════════════════════
    FORBIDDEN:
    ══════════════════════════════════════════════════
    - APPROVE with non-empty remaining_issues
    - Using action not in allowed_actions
    - Omitting patched_source in PROPOSE_PATCH
    - Omitting package in REJECT_DEP
    - Any text outside the JSON object
""")


# ── JSON parser  –  3 strategies, NO silent downgrade ────────────────────────
def _parse_raw(raw: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Returns (data_dict, error_string).
    error_string is None on success.
    NO fallback action is returned – caller handles failure explicitly.
    """
    raw = re.sub(r"```(?:json)?|```", "", raw).strip()

    # Strategy 1: direct
    try:
        return json.loads(raw), None
    except json.JSONDecodeError:
        pass

    # Strategy 2: first {...} block
    m = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group()), None
        except json.JSONDecodeError:
            pass

    # Strategy 3: outermost braces
    s, e = raw.find("{"), raw.rfind("}")
    if s != -1 and e > s:
        try:
            return json.loads(raw[s:e+1]), None
        except json.JSONDecodeError:
            pass

    return None, f"JSON parse failure. Raw (first 200 chars): {raw[:200]!r}"


def _normalise(data: Dict[str, Any]) -> Dict[str, Any]:
    """Fill in optional defaults so Pydantic validation is cleanest."""
    if not isinstance(data.get("evidence"), list):
        data["evidence"] = [data.get("detail", "no evidence")] if data.get("detail") else ["no evidence"]
    if not isinstance(data.get("resolved_issues"), dict):
        data["resolved_issues"] = {}
    if not isinstance(data.get("remaining_issues"), dict):
        data["remaining_issues"] = {}
    # For APPROVE: ensure remaining_issues is empty dict (not None)
    if data.get("action_type") == "APPROVE" and not data.get("remaining_issues"):
        data["remaining_issues"] = {}
    # For REJECT: ensure remaining_issues has at least a placeholder if missing
    if data.get("action_type") == "REJECT" and not any(data.get("remaining_issues", {}).values()):
        data["remaining_issues"] = {"UNKNOWN": ["see_reasoning"]}
    # Ensure reasoning for terminal actions
    for at in ("APPROVE", "REJECT"):
        if data.get("action_type") == at and not data.get("reasoning"):
            data["reasoning"] = data.get("detail") or data.get("final_decision") or "[reasoning not provided]"
    return data


# ── LLM call ──────────────────────────────────────────────────────────────────
def _call_llm(
    client: OpenAI,
    obs: Observation,
    history: List[Dict[str, str]],
) -> str:
    user_msg = json.dumps({
        "source_code":         obs.source_code,
        "dependency_manifest": obs.dependency_manifest,
        "open_issues":         obs.open_issues,          # structured by category
        "allowed_actions":     obs.allowed_actions,      # ONLY these are legal
        "current_phase":       obs.current_phase,        # what phase we are in
        "status":              obs.status,
        "hint":                obs.hint,                 # loop hint if present
        "step":                obs.step_index,
        "priority_summary":    obs.priority_summary,
        "MANDATORY":           (
            f"current_phase={obs.current_phase!r}. "
            f"You MUST choose from allowed_actions={obs.allowed_actions}. "
            "Read open_issues carefully before acting."
        ),
    }, indent=2)

    history.append({"role": "user", "content": user_msg})
    model = GROQ_MODEL if PROVIDER == "groq" else HF_MODEL

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": _SYSTEM}] + history,
        temperature=TEMPERATURE,
        max_tokens=2048,
        timeout=60,
    )
    raw = response.choices[0].message.content or ""
    history.append({"role": "assistant", "content": raw})
    return raw


# ── Agent step  –  retry once on parse failure, NO silent downgrade ───────────
def _agent_step(
    client: OpenAI,
    obs: Observation,
    history: List[Dict[str, str]],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Returns (action_dict_or_None, parse_error_or_None).
    If action_dict is None → caller submits error to env for penalty.
    Never silently returns a downgraded action.
    """
    for attempt in range(2):
        raw = _call_llm(client, obs, history)
        data, err = _parse_raw(raw)
        if err is None and data is not None:
            data = _normalise(data)
            return data, None
        if attempt == 0:
            logger.warning("[PARSE FAIL attempt 1] %s", err)
            # Inject retry hint into history (not into observation)
            history.append({
                "role": "user",
                "content": (
                    f"[PARSE ERROR] Your last response could not be parsed as JSON. "
                    f"Error: {err}. "
                    "Output ONLY a raw JSON object. No markdown, no text, no fences. "
                    f"You are in phase={obs.current_phase!r}. "
                    f"Allowed actions: {obs.allowed_actions}."
                )
            })
    # Both attempts failed
    logger.error("[PARSE FAIL] Both attempts failed. Returning error to env.")
    return None, err or "Unknown parse error"


# ── Episode runner ─────────────────────────────────────────────────────────────
def run_episode() -> None:
    SEP  = "=" * 72
    SEP2 = "-" * 72
    model = GROQ_MODEL if PROVIDER == "groq" else HF_MODEL
    token = HF_TOKEN or GROQ_KEY or ""
    preview = f"{token[:8]}...{token[-4:]}" if len(token) > 12 else ("NOT SET" if not token else token)

    # ── [START] ───────────────────────────────────────────────────────────────
    print(SEP)
    print("[START] SENTINEL-PR v4.0.0  |  task_eval_auth_flaw (Hard)")
    print(f"  Provider    : {PROVIDER}")
    print(f"  Model       : {model}")
    print(f"  Temperature : {TEMPERATURE} (deterministic)")
    print(f"  Max steps   : {MAX_STEPS}")
    print(f"  Token       : {preview}")
    print(f"  .env loaded : {__import__('pathlib').Path('.env').exists()}")
    print(SEP)

    if not token:
        print("\n[ERROR] No API token. Create .env with HF_TOKEN or GROQ_API_KEY.\n")
        raise SystemExit(1)

    env     = SentinelPREnv(max_steps=MAX_STEPS)
    client  = _build_client()
    history: List[Dict[str, str]] = []

    obs     = env.reset("task_eval_auth_flaw")
    total   = 0.0
    step    = 0
    result: Optional[StepResult] = None

    print(f"\nInitial phase: {obs.current_phase}")
    print(f"Open issues:   {obs.priority_summary}")
    print(f"Allowed:       {obs.allowed_actions}")
    print(f"Status: {obs.status[:200]}\n")

    while True:
        step += 1
        print(f"\n{SEP}")
        print(f"[STEP {step}]  phase={obs.current_phase}  allowed={obs.allowed_actions}")
        print(SEP2)

        if obs.hint:
            print(f"  HINT: {obs.hint}")

        # Agent decision
        action_dict, parse_error = _agent_step(client, obs, history)

        if parse_error is not None:
            # Explicit parse failure – submit error dict to env for penalty
            print(f"  [PARSE ERROR] {parse_error[:200]}")
            # Submit a deliberately invalid dict to trigger R_PARSE_FAIL
            result = env.step({"_parse_error": parse_error})
        else:
            print(f"  action_type : {action_dict.get('action_type')}")
            print(f"  confidence  : {action_dict.get('confidence')}")
            if action_dict.get("evidence"):
                print(f"  evidence    : {action_dict['evidence'][:2]}")
            if action_dict.get("detail"):
                print(f"  detail      : {str(action_dict['detail'])[:150]}")
            if action_dict.get("fix_rationale"):
                print(f"  fix_why     : {str(action_dict['fix_rationale'])[:150]}")
            if action_dict.get("reasoning"):
                print(f"  reasoning   : {str(action_dict['reasoning'])[:200]}")
            result = env.step(action_dict)

        total += result.reward

        # ── [STEP] output ─────────────────────────────────────────────────────
        val = result.info.get("validation_results") or result.info.get("validation") or {}
        print(f"\n  Reward breakdown:")
        for k, v in result.info["reward_breakdown"].items():
            star = " [*]" if k in ("victory_bonus","correct_reject") else "    "
            print(f"{star}  {k:<42} {v:+.4f}")
        print(f"\n  Step reward    : {result.reward:+.4f}")
        print(f"  Episode total  : {total:+.4f}")
        print(f"  Open issues    : {result.info['priority_summary']}")
        print(f"  Phase after    : {result.info['flow_phase']}")
        print(f"  Allowed next   : {result.info['allowed_actions']}")
        if result.info.get("schema_error"):
            print(f"  [SCHEMA ERROR] : {result.info['schema_error'][:120]}")
        if result.info.get("flow_violation"):
            print(f"  [FLOW VIOLATION]: {result.info['flow_violation'][:120]}")
        if val:
            print(f"  Validation     : syntax={val.get('syntax')} security={val.get('security')} runtime={val.get('runtime')}")
        print(f"  done={result.done}  victory={result.info['victory']}")

        if result.done:
            break
        obs = result.observation

    # ── [END] ─────────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("[END] EPISODE COMPLETE")
    print(SEP2)
    print(f"  Steps taken    : {step}")
    print(f"  Episode reward : {total:+.4f}")
    print(f"  Victory        : {result.info['victory']}")
    if result.info["victory"]:
        print("  [VERDICT] SUCCESS – all issues resolved, APPROVE confirmed")
    else:
        print(f"  [VERDICT] NO VICTORY – reward capped at {env._episode_reward:+.4f}")
    print(SEP)


if __name__ == "__main__":
    try:
        run_episode()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
