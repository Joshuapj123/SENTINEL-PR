"""
test_success.py – Perfect Agent simulation for SENTINEL-PR v2.1.0
==================================================================
Strict victory condition: ALL bandit issues (High + Medium + Low)
AND all safety CVEs must be zero. Unpinned deps must be pinned.

Episode structure:
  Step 1 – FLAG_VULN      (eval RCE + manifest CVEs)
  Step 2 – PROPOSE_PATCH  (fix source + patch manifest)
  Step 3 – APPROVE        (zero-issue full scan -> Victory +2.0)

Run: python test_success.py
No API key required.
"""
from __future__ import annotations

import sys
from env import Action, ActionType, SentinelPREnv

SEP  = "=" * 70
SEP2 = "-" * 70

# ---------------------------------------------------------------------------
# Perfectly patched source  (zero bandit issues at all severities)
# ---------------------------------------------------------------------------
CLEAN_SOURCE = (
    "import hmac, hashlib, time, json\n"
    "from functools import wraps\n"
    "from flask import request, jsonify, g\n"
    "\n"
    'SECRET = "super-secret-signing-key"\n'
    "\n"
    "def _verify_token(token: str) -> dict:\n"
    "    try:\n"
    "        import base64\n"
    '        parts = token.split(".")\n'
    "        payload_b64, sig = parts[0], parts[1]\n"
    '        payload_str = base64.b64decode(payload_b64 + "==").decode()\n'
    "        payload = json.loads(payload_str)\n"
    "        expected_sig = hmac.new(\n"
    "            SECRET.encode(), payload_b64.encode(), hashlib.sha256\n"
    "        ).hexdigest()\n"
    "        if hmac.compare_digest(sig, expected_sig):\n"
    "            return payload\n"
    "        return {}\n"
    "    except Exception:\n"
    "        return {}\n"
    "\n"
    "def require_auth(f):\n"
    "    @wraps(f)\n"
    "    def decorated(*args, **kwargs):\n"
    '        token = request.headers.get("X-Auth-Token", "")\n'
    "        identity = _verify_token(token)\n"
    "        if not identity:\n"
    '            return jsonify({"error": "Unauthorized"}), 401\n'
    "        g.user = identity\n"
    "        return f(*args, **kwargs)\n"
    "    return decorated\n"
    "\n"
    "def get_expiry(token: str) -> int:\n"
    "    import base64\n"
    '    parts = token.split(".")\n'
    '    payload_str = base64.b64decode(parts[0] + "==").decode()\n'
    "    data = json.loads(payload_str)\n"
    "    return data.get(\"exp\", 0)\n"
)

# Patched manifest: CVEs fixed, all packages pinned to safe versions
CLEAN_MANIFEST = (
    "flask==3.0.3\n"
    "pyyaml==6.0.1\n"
    "requests==2.32.3\n"
)

AGENT_SCRIPT = [
    Action(
        action_type=ActionType.FLAG_VULN,
        detail=(
            "eval() called on attacker-controlled token payload_str at two sites "
            "in _verify_token and get_expiry (B307, CWE-78). "
            "Manifest has CVE-2022-1471 (pyyaml 5.4.1) and CVE-2023-30861 (flask 2.3.3)."
        ),
        confidence=0.96,
    ),
    Action(
        action_type=ActionType.PROPOSE_PATCH,
        detail=(
            "Replace eval() with json.loads() in _verify_token and get_expiry. "
            "Replace == HMAC comparison with hmac.compare_digest(). "
            "Update manifest: flask==3.0.3, pyyaml==6.0.1, requests==2.32.3."
        ),
        patched_source=CLEAN_SOURCE,
        patched_manifest=CLEAN_MANIFEST,
        confidence=0.98,
    ),
    Action(
        action_type=ActionType.APPROVE,
        detail=(
            "All bandit scans clean: zero High, Medium, Low issues. "
            "All manifest CVEs resolved. All packages pinned. Safe to approve."
        ),
        confidence=0.99,
    ),
]


def run() -> None:
    print(SEP)
    print("  SENTINEL-PR v2.1.0  |  Perfect Agent  |  test_success.py")
    print("  Task   : task_eval_auth_flaw (Hard)")
    print("  Victory: bandit total + safety CVEs + unpinned == 0")
    print(SEP)

    env   = SentinelPREnv(max_steps=10)
    obs   = env.reset("task_eval_auth_flaw")
    total = 0.0
    result = None

    print(f"\nInitial status: {obs.status}\n")

    for step_num, action in enumerate(AGENT_SCRIPT, start=1):
        print(SEP2)
        print(f"STEP {step_num}  |  {action.action_type.value}  (confidence={action.confidence:.2f})")
        print(f"Detail: {(action.detail or '')[:120]}...")
        print(SEP2)

        result = env.step(action)
        total += result.reward

        # Reward breakdown
        print("\n  -- Reward Breakdown --")
        for k, v in result.info["reward_breakdown"].items():
            star = " [*]" if k in ("victory_bonus", "supply_chain_bonus") else "    "
            print(f"{star}  {k:<36} {v:+.4f}")
        print(f"\n  Step reward   : {result.reward:+.4f}")
        print(f"  Episode total : {total:+.4f}")

        # Scan results
        if result.info.get("bandit_after"):
            ba = result.info["bandit_after"]
            sa = result.info.get("safety_after", {})
            print(f"\n  Bandit after  : {ba['high_count']}H {ba['medium_count']}M {ba['low_count']}L "
                  f"(total={ba['total']})  ids={ba['ids']}")
            print(f"  Safety after  : {sa.get('vuln_count', '?')} CVEs  "
                  f"unpinned={sa.get('unpinned', [])}")
            clean = ba["total"] == 0 and sa.get("vuln_count", 1) == 0 and not sa.get("unpinned")
            print(f"  Full scan     : {'CLEAN [*]' if clean else 'issues remain'}")

        # Scan next
        sn = result.info.get("scan_next", {})
        print(f"\n  scan_next.total_issues : {sn.get('total_issues', '?')}")
        print(f"  scan_next.is_clean     : {sn.get('is_clean', '?')}")
        print(f"  scan_next.has_unpinned : {sn.get('has_unpinned', '?')}")

        # Aura
        aura = result.info["aura"]
        print(f"\n  Aura composite={aura['composite']:.3f}  "
              f"lex={aura['lexical']:.3f}  "
              f"coh={aura['coherence']:.3f}  "
              f"cal={aura['calibration']:.3f}  "
              f"risk={aura['risk']}")
        if result.info["system_doubt"]:
            print("  [SYSTEM_DOUBT] triggered in status")

        # Status
        print(f"\n  Status: {result.observation.status[:200]}")

        # Supply chain
        if result.info.get("supply_chain_bonus"):
            print("\n  [SUPPLY CHAIN BONUS TRIGGERED]")

        # Victory / termination
        print(f"\n  done={result.done}  terminated={result.terminated}  "
              f"victory={result.info['victory']}")
        print()

        if result.done:
            break

    print(SEP)
    print("  EPISODE SUMMARY")
    print(SEP2)
    print(f"  Steps taken    : {env._step_count}")
    print(f"  Total reward   : {total:+.4f}")
    print(f"  Victory        : {result.info['victory']}")
    print(f"  Terminated     : {result.terminated}")
    print(SEP2)

    # Assertions
    assert result.info["victory"],         "FAIL: victory flag not set"
    assert result.reward >= 3.5,           f"FAIL: reward={result.reward:.4f} expected >=3.5"
    assert result.terminated,             "FAIL: terminated should be True"
    assert result.info["scan_next"]["is_clean"], "FAIL: scan not fully clean"
    assert not result.info["scan_next"]["has_unpinned"], "FAIL: unpinned deps remain"

    print("\n  [OK] Victory flag set")
    print("  [OK] Final reward >= +3.5")
    print("  [OK] terminated = True")
    print("  [OK] scan_next.is_clean = True")
    print("  [OK] No unpinned deps")
    print("  [OK] PERFECT SCAN achieved - Meaningful Progress criterion MET")
    print(SEP)


if __name__ == "__main__":
    try:
        run()
    except AssertionError as e:
        print(f"\n  [FAIL] {e}")
        sys.exit(1)
