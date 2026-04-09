"""
grader.py  |  SENTINEL-PR  |  OpenEnv 2026 Grader
Scores strictly in (0.01, 0.99) — never 0.0 or 1.0.

Called by validator: grade(task_id, episode_result)
Partial credit:  1 of 2 vulns fixed → reward ~2.5 → score ~0.60
Full victory:    all fixed + APPROVE → score 0.99
No progress:     repeated penalties  → score 0.01
"""
from __future__ import annotations
from typing import Any, Dict

# Reward min/max per task (matches env.py reward constants)
_TASK_RANGES: Dict[str, Dict[str, float]] = {
    "task_hardcoded_key":  {"min": -3.0,  "max": 6.0},
    "task_vulnerable_dep": {"min": -3.0,  "max": 6.0},
    "task_eval_auth_flaw": {"min": -18.0, "max": 13.0},
}

# Clip bounds — strictly inside (0, 1), never 0.0 or 1.0
_CLIP_MIN = 0.01
_CLIP_MAX = 0.99


def grade(task_id: str, episode_result: Dict[str, Any]) -> float:
    """
    Return score strictly in (0.01, 0.99).

    Args:
        task_id:        Task identifier string.
        episode_result: Dict with keys:
                          total_reward (float) - sum of all step rewards
                          victory      (bool)  - True if agent completed all objectives

    Returns:
        float in open interval (0.01, 0.99) — never exactly 0.0 or 1.0
    """
    total_reward = float(episode_result.get("total_reward", 0.0))
    victory      = bool(episode_result.get("victory", False))

    ranges = _TASK_RANGES.get(task_id, {"min": -18.0, "max": 13.0})
    r_min  = ranges["min"]
    r_max  = ranges["max"]

    # ── Division-by-zero guard ────────────────────────────────────────────────
    if r_max == r_min:
        normalised = 0.5
    else:
        normalised = (total_reward - r_min) / (r_max - r_min)

    # Victory: push toward 0.95 but never reach clip max
    if victory:
        normalised = max(normalised, 0.90)

    # Strict clip — score is ALWAYS in open interval (0.01, 0.99)
    score = max(_CLIP_MIN, min(_CLIP_MAX, normalised))
    return round(score, 4)


def make_grader(**kwargs: Any) -> "SentinelGrader":
    """OpenEnv entry point."""
    return SentinelGrader(**kwargs)


class SentinelGrader:
    """OpenEnv-compatible grader class."""

    def __init__(self, **kwargs: Any) -> None:
        self.task_id: str = kwargs.get("task_id", "")

    def score(self, episode_result: Dict[str, Any]) -> float:
        """Return score strictly in (0.01, 0.99)."""
        return grade(self.task_id, episode_result)


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        ("task_hardcoded_key",   6.0,   True,  0.99),   # max reward
        ("task_hardcoded_key",  -3.0,   False, 0.01),   # min reward
        ("task_hardcoded_key",   1.5,   False, 0.50),   # partial: ~mid
        ("task_vulnerable_dep",  3.0,   False, 0.6667), # partial credit
        ("task_eval_auth_flaw", 13.0,   True,  0.99),   # max reward
        ("task_eval_auth_flaw",-18.0,   False, 0.01),   # min reward
        ("task_eval_auth_flaw",  0.0,   False, 0.5806), # mid: partial
        # Division-by-zero guard
        ("unknown_task",         0.0,   False, 0.5806), # fallback range
    ]
    print("Grader self-test:")
    all_ok = True
    for task_id, reward, victory, expected in tests:
        s = grade(task_id, {"total_reward": reward, "victory": victory})
        in_range = 0.0 < s < 1.0
        correct  = abs(s - expected) < 0.01
        ok = in_range and correct
        if not ok:
            all_ok = False
        print(f"  {'OK' if ok else 'FAIL'}  {task_id:<24} "
              f"reward={reward:+7.1f}  victory={str(victory):<5}  "
              f"score={s:.4f}  expected~{expected:.4f}")
    print()
    print("All tests passed!" if all_ok else "SOME TESTS FAILED!")
