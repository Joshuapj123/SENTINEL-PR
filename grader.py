"""
grader.py  |  SENTINEL-PR  |  OpenEnv 2026 Grader
Computes a score strictly in (0, 1) for each task based on episode reward.
Called by the OpenEnv validator after each episode.
"""
from __future__ import annotations
from typing import Any, Dict


# Reward ranges per task (from openenv.yaml reward section)
_TASK_RANGES: Dict[str, Dict[str, float]] = {
    "task_hardcoded_key":  {"min": -3.0,  "max": 6.0},
    "task_vulnerable_dep": {"min": -3.0,  "max": 6.0},
    "task_eval_auth_flaw": {"min": -18.0, "max": 13.0},
}

# Scores are clipped to (0.01, 0.99) to stay strictly between 0 and 1
_CLIP_MIN = 0.01
_CLIP_MAX = 0.99


def grade(task_id: str, episode_result: Dict[str, Any]) -> float:
    """
    Compute a score in (0, 1) for the given task and episode result.

    Args:
        task_id:        One of the task IDs defined in openenv.yaml.
        episode_result: Dict containing at minimum:
                          - "total_reward" (float)  total episode reward
                          - "victory"      (bool)   whether agent achieved victory

    Returns:
        float strictly in (0.01, 0.99)
    """
    total_reward: float = float(episode_result.get("total_reward", 0.0))
    victory: bool       = bool(episode_result.get("victory", False))

    ranges = _TASK_RANGES.get(task_id, {"min": -18.0, "max": 13.0})
    r_min  = ranges["min"]
    r_max  = ranges["max"]

    # Normalise reward to [0, 1]
    if r_max == r_min:
        normalised = 0.5
    else:
        normalised = (total_reward - r_min) / (r_max - r_min)

    # Victory bonus: push score toward 0.95
    if victory:
        normalised = max(normalised, 0.90)

    # Clip strictly to (0.01, 0.99) — never 0.0 or 1.0
    score = max(_CLIP_MIN, min(_CLIP_MAX, normalised))
    return round(score, 4)


def grade_all(results: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    """
    Grade all tasks at once.

    Args:
        results: {task_id: episode_result_dict}

    Returns:
        {task_id: score} — all scores strictly in (0.01, 0.99)
    """
    return {task_id: grade(task_id, result) for task_id, result in results.items()}


# ── OpenEnv entry point ───────────────────────────────────────────────────────
def make_grader(**kwargs: Any) -> "SentinelGrader":
    return SentinelGrader(**kwargs)


class SentinelGrader:
    """OpenEnv-compatible grader class."""

    def __init__(self, **kwargs: Any) -> None:
        self.task_id: str = kwargs.get("task_id", "")

    def score(self, episode_result: Dict[str, Any]) -> float:
        """Return score strictly in (0, 1)."""
        return grade(self.task_id, episode_result)


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        ("task_hardcoded_key",  {"total_reward": 6.0,  "victory": True},  0.99),
        ("task_hardcoded_key",  {"total_reward": -3.0, "victory": False}, 0.01),
        ("task_hardcoded_key",  {"total_reward": 1.5,  "victory": False}, None),
        ("task_eval_auth_flaw", {"total_reward": 13.0, "victory": True},  0.99),
        ("task_eval_auth_flaw", {"total_reward": -18.0,"victory": False}, 0.01),
    ]
    print("Grader self-test:")
    all_ok = True
    for task_id, result, expected in tests:
        s = grade(task_id, result)
        ok = (0.0 < s < 1.0)
        if expected is not None:
            ok = ok and (s == expected)
        status = "OK" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"  [{status}] {task_id} reward={result['total_reward']:+.1f} "
              f"victory={result['victory']} → score={s}")
    print("All tests passed!" if all_ok else "SOME TESTS FAILED")
