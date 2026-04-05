"""
smoke_test.py – sanity-check all three tasks without a real agent.
Run: python smoke_test.py
"""
import json
from env import (
    SentinelPREnv,
    Action,
    ActionType,
)

env = SentinelPREnv(max_steps=10)

SCENARIOS = [
    (
        "task_hardcoded_key",
        Action(
            action_type=ActionType.FLAG_VULN,
            detail="Hardcoded API key found: API_KEY = 'sk-prod-...'",
            confidence=0.95,
        ),
    ),
    (
        "task_vulnerable_dep",
        Action(
            action_type=ActionType.REJECT_DEP,
            package="pillow",
            detail="Pillow 9.5.0 is affected by CVE-2024-28219; upgrade to >=10.3.0",
            confidence=0.9,
        ),
    ),
    (
        "task_eval_auth_flaw",
        Action(
            action_type=ActionType.PROPOSE_PATCH,
            detail="Replace eval() with json.loads(); use hmac.compare_digest().",
            patched_source="""\
import hmac, hashlib, time, json
from functools import wraps
from flask import request, jsonify, g

SECRET = "super-secret-signing-key"

def _verify_token(token: str) -> dict:
    try:
        import base64
        parts = token.split(".")
        payload_b64, sig = parts[0], parts[1]
        payload_str = base64.b64decode(payload_b64 + "==").decode()
        payload = json.loads(payload_str)          # FIXED: no eval()
        expected_sig = hmac.new(
            SECRET.encode(), payload_b64.encode(), hashlib.sha256
        ).hexdigest()
        if hmac.compare_digest(sig, expected_sig):  # FIXED: timing-safe
            return payload
        return {}
    except Exception:
        return {}

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("X-Auth-Token", "")
        identity = _verify_token(token)
        if not identity:
            return jsonify({"error": "Unauthorized"}), 401
        g.user = identity
        return f(*args, **kwargs)
    return decorated

def get_expiry(token: str) -> int:
    import base64
    parts = token.split(".")
    payload_str = base64.b64decode(parts[0] + "==").decode()
    data = json.loads(payload_str)               # FIXED: no eval()
    return data.get("exp", int(time.time()))
""",
            confidence=0.88,
        ),
    ),
]

for task_id, action in SCENARIOS:
    obs = env.reset(task_id)
    result = env.step(action)
    print(f"\n{'='*60}")
    print(f"Task   : {task_id}")
    print(f"Action : {action.action_type.value}")
    print(f"Reward : {result.reward:+.4f}")
    print(f"Done   : {result.done}")
    print(f"Breakdown:")
    for k, v in result.info["reward_breakdown"].items():
        print(f"  {k}: {v}")
    if result.info.get("bandit_after"):
        ba = result.info["bandit_after"]
        print(f"Bandit after: {ba['high_count']}H {ba['medium_count']}M {ba['low_count']}L (mock={ba['mock']})")

print("\n✓ Smoke test complete.")
