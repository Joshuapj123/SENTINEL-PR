"""
grader.py  |  SENTINEL-PR  |  OpenEnv 2026 Grader
Returns scores strictly in (0.06, 0.93) — never 0.0 or 1.0.

Formula: score = 0.2 + (fixed_issues / total_issues) * 0.6 + 0.2 * validation_pass
Clamped:  score = min(0.93, max(0.06, score))
"""
from __future__ import annotations
from typing import Any, Dict

# ── Per-task config ───────────────────────────────────────────────────────────
_TASK_CONFIG: Dict[str, Dict[str, Any]] = {
    "task_hardcoded_key": {
        "total_issues": 1,
        "reward_min": -3.0,
        "reward_max": 6.0,
    },
    "task_vulnerable_dep": {
        "total_issues": 3,
        "reward_min": -3.0,
        "reward_max": 6.0,
    },
    "task_eval_auth_flaw": {
        "total_issues": 4,
        "reward_min": -18.0,
        "reward_max": 13.0,
    },
    "task_insecure_deserialization": {
        "total_issues": 2,
        "reward_min": -3.0,
        "reward_max": 6.0,
    },
    "task_command_injection": {
        "total_issues": 3,
        "reward_min": -3.0,
        "reward_max": 6.0,
    },
}

_SCORE_MIN = 0.06
_SCORE_MAX = 0.93


def grade(task_id: str, episode_result: Dict[str, Any]) -> float:
    """
    Return score strictly in (0.06, 0.93).
    Clamped: min(0.93, max(0.06, score))

    episode_result keys (all optional):
      fixed_issues  (int)   — issues resolved; derived from total_reward if absent
      victory       (bool)  — True if all objectives met
      total_reward  (float) — raw episode reward; used when fixed_issues missing
    """
    cfg          = _TASK_CONFIG.get(task_id, {"total_issues": 2, "reward_min": -18.0, "reward_max": 13.0})
    total_issues = cfg["total_issues"]
    victory      = bool(episode_result.get("victory", False))

    # Derive fixed_issues robustly — never raises, never returns None
    raw_fi = episode_result.get("fixed_issues")
    if raw_fi is not None:
        try:
            fixed_issues = int(raw_fi)
        except (TypeError, ValueError):
            fixed_issues = None

    if raw_fi is None or fixed_issues is None:
        # Fall back: map total_reward onto [0, total_issues]
        try:
            reward = float(episode_result.get("total_reward", 0.0))
        except (TypeError, ValueError):
            reward = 0.0
        r_min = cfg["reward_min"]
        r_max = cfg["reward_max"]
        span  = r_max - r_min if r_max != r_min else 1.0
        ratio_r      = max(0.0, min(1.0, (reward - r_min) / span))
        fixed_issues = round(ratio_r * total_issues)

    # Ratio of issues fixed
    if total_issues <= 0:
        ratio = 0.0
    else:
        ratio = min(int(fixed_issues), total_issues) / total_issues

    raw   = 0.2 + ratio * 0.6 + 0.2 * int(victory)
    score = min(_SCORE_MAX, max(_SCORE_MIN, raw))
    return round(score, 4)

def make_grader(**kwargs: Any) -> "Grader":
    """OpenEnv entry point."""
    return Grader(**kwargs)


class Grader:
    """OpenEnv-compatible grader class. grader_class: Grader"""

    def __init__(self, **kwargs: Any) -> None:
        self.task_id: str = kwargs.get("task_id", "")

    def score(self, episode_result: Dict[str, Any]) -> float:
        """Return score strictly in (0.06, 0.93)."""
        return grade(self.task_id, episode_result)


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        # (task_id, fixed, victory, expected_score)
        ("task_eval_auth_flaw",          0, False, 0.20),   # nothing fixed  → clamped to 0.20
        ("task_eval_auth_flaw",          2, False, 0.50),   # half fixed
        ("task_eval_auth_flaw",          4, True,  0.93),   # all fixed + victory → clamped to 0.93
        ("task_insecure_deserialization",0, False, 0.20),
        ("task_insecure_deserialization",1, False, 0.50),
        ("task_insecure_deserialization",2, True,  0.93),   # clamped to 0.93
        ("task_command_injection",       0, False, 0.20),
        ("task_command_injection",       3, True,  0.93),   # clamped to 0.93
        ("task_hardcoded_key",           1, True,  0.93),   # clamped to 0.93
        ("task_vulnerable_dep",          0, False, 0.20),
        ("task_vulnerable_dep",          3, True,  0.93),   # clamped to 0.93
    ]
    print("Grader self-test:")
    all_ok = True
    for task_id, fixed, victory, expected in tests:
        s = grade(task_id, {"fixed_issues": fixed, "victory": victory})
        in_range = 0.0 < s < 1.0
        correct  = abs(s - expected) < 0.01
        ok = in_range and correct
        if not ok:
            all_ok = False
        print(f"  {'OK' if ok else 'FAIL'}  {task_id:<32} "
              f"fixed={fixed} victory={str(victory):<5} "
              f"score={s:.4f}  expected={expected:.4f}")
    print()
    print("All tests passed!" if all_ok else "SOME TESTS FAILED!")
