"""
grader.py  |  SENTINEL-PR  |  OpenEnv 2026 Grader

Returns {"score": float} for every task, every episode.
Score is ALWAYS strictly between 0 and 1: max(0.01, min(0.99, score))
Default score = 0.5 if anything fails.

Formula: score = 0.2 + (fixed_ratio * 0.6) + (0.2 * victory)
"""
from __future__ import annotations

from typing import Any, Dict

# ── Per-task config ───────────────────────────────────────────────────────────
_TASK_CONFIG: Dict[str, Dict[str, Any]] = {
    "task_hardcoded_key": {
        "total_issues": 1,
        "reward_min":   -3.0,
        "reward_max":    6.0,
    },
    "task_vulnerable_dep": {
        "total_issues": 3,
        "reward_min":   -3.0,
        "reward_max":    6.0,
    },
    "task_eval_auth_flaw": {
        "total_issues": 4,
        "reward_min":  -18.0,
        "reward_max":   13.0,
    },
    "task_insecure_deserialization": {
        "total_issues": 2,
        "reward_min":   -3.0,
        "reward_max":    6.0,
    },
    "task_command_injection": {
        "total_issues": 3,
        "reward_min":   -3.0,
        "reward_max":    6.0,
    },
}

# Fallback config for unknown task_ids
_DEFAULT_CONFIG: Dict[str, Any] = {
    "total_issues": 2,
    "reward_min":  -18.0,
    "reward_max":   13.0,
}

_SCORE_MIN = 0.01
_SCORE_MAX = 0.99
_SCORE_DEFAULT = 0.5


def _clamp(x: float) -> float:
    """Clamp to strictly open (0, 1). Never returns 0.0 or 1.0."""
    return max(_SCORE_MIN, min(_SCORE_MAX, float(x)))


def _derive_fixed_issues(
    episode_result: Dict[str, Any],
    cfg: Dict[str, Any],
) -> int:
    """
    Get fixed_issues as int. Falls back to total_reward if missing/invalid.
    Never raises.
    """
    total_issues = cfg["total_issues"]

    # Primary: fixed_issues field
    raw = episode_result.get("fixed_issues")
    if raw is not None:
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            pass

    # Fallback: derive from total_reward
    try:
        reward = float(episode_result.get("total_reward", 0.0) or 0.0)
        r_min  = cfg["reward_min"]
        r_max  = cfg["reward_max"]
        span   = r_max - r_min if r_max != r_min else 1.0
        ratio  = max(0.0, min(1.0, (reward - r_min) / span))
        return round(ratio * total_issues)
    except (TypeError, ValueError, ZeroDivisionError):
        pass

    return 0


def grade(task_id: str, episode_result: Dict[str, Any]) -> float:
    """
    Compute score for one episode. Always returns float in (0.01, 0.99).
    Returns _SCORE_DEFAULT (0.5) if anything fails.

    Handles all 5 task IDs. Falls back gracefully for unknown tasks.
    """
    try:
        cfg          = _TASK_CONFIG.get(task_id, _DEFAULT_CONFIG)
        total_issues = max(1, int(cfg["total_issues"]))
        victory      = bool(episode_result.get("victory", False))
        fixed        = _derive_fixed_issues(episode_result, cfg)
        ratio        = min(fixed, total_issues) / total_issues
        raw          = 0.2 + ratio * 0.6 + 0.2 * int(victory)
        return _clamp(raw)
    except Exception:
        return _SCORE_DEFAULT


def make_grader(**kwargs: Any) -> "Grader":
    """OpenEnv entry point."""
    return Grader(**kwargs)


class Grader:
    """
    OpenEnv-compatible grader. Instantiated once per task via grader_class: Grader.
    Calls grade(task_id, episode_result) and wraps result in {"score": float}.
    Never crashes — returns {"score": 0.5} on any error.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.task_id: str = str(kwargs.get("task_id", ""))

    def score(self, episode_result: Dict[str, Any]) -> Dict[str, float]:
        """
        Required by OpenEnv. Returns {"score": float} strictly in (0, 1).
        """
        try:
            s = grade(self.task_id, episode_result or {})
        except Exception:
            s = _SCORE_DEFAULT
        return {"score": _clamp(s)}


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ALL_TASKS = [
        "task_hardcoded_key",
        "task_vulnerable_dep",
        "task_eval_auth_flaw",
        "task_insecure_deserialization",
        "task_command_injection",
    ]

    CASES = [
        # (task_id, episode_result, description)
        ("task_hardcoded_key",           {"fixed_issues": 1,  "victory": True},              "perfect"),
        ("task_hardcoded_key",           {"fixed_issues": 0,  "victory": False},             "zero"),
        ("task_vulnerable_dep",          {"fixed_issues": 3,  "victory": True},              "perfect"),
        ("task_vulnerable_dep",          {"total_reward": 4.5,"victory": True},              "fallback reward"),
        ("task_eval_auth_flaw",          {"fixed_issues": 4,  "victory": True},              "perfect"),
        ("task_eval_auth_flaw",          {"fixed_issues": 2,  "victory": False},             "half"),
        ("task_eval_auth_flaw",          {},                                                  "empty dict"),
        ("task_eval_auth_flaw",          {"fixed_issues": None, "total_reward": 12.5},       "fi=None, reward"),
        ("task_eval_auth_flaw",          {"fixed_issues": "bad"},                            "fi=bad string"),
        ("task_insecure_deserialization",{"fixed_issues": 2,  "victory": True},              "perfect"),
        ("task_insecure_deserialization",{"total_reward": -3.0},                             "worst reward"),
        ("task_command_injection",       {"fixed_issues": 3,  "victory": True},              "perfect"),
        ("task_command_injection",       {"fixed_issues": 0,  "victory": False},             "zero"),
        ("unknown_task",                 {"fixed_issues": 1,  "victory": False},             "unknown task"),
    ]

    print("Grader self-test")
    print("=" * 70)
    all_ok = True
    for task_id, ep, desc in CASES:
        g = Grader(task_id=task_id)
        result = g.score(ep)
        s = result.get("score", None)
        in_range = isinstance(s, float) and 0.0 < s < 1.0
        ok = in_range
        if not ok:
            all_ok = False
        print(f"  {'OK  ' if ok else 'FAIL'} {task_id:<35} [{desc:<20}] score={s}")

    print("=" * 70)
    print("All tests passed!" if all_ok else "SOME TESTS FAILED!")
