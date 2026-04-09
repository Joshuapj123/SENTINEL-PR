"""
grader.py  |  SENTINEL-PR  |  OpenEnv 2026 Grader
Returns scores strictly in (0.01, 0.99) — never 0.0 or 1.0.

The validator calls grade(task_id, episode_result) after each episode.
Score formula:
    normalised = (total_reward - reward_min) / (reward_max - reward_min)
    score      = clip(normalised, 0.01, 0.99)

Partial credit example:
    Agent fixes 1 of 2 vulns → reward ≈ 2.5 → score ≈ 0.60  (not 0 or 1)
"""
from __future__ import annotations
from typing import Any, Dict

_TASK_RANGES: Dict[str, Dict[str, float]] = {
    "task_hardcoded_key":  {"min": -3.0,  "max": 6.0},
    "task_vulnerable_dep": {"min": -3.0,  "max": 6.0},
    "task_eval_auth_flaw": {"min": -18.0, "max": 13.0},
}
_CLIP_MIN = 0.01
_CLIP_MAX = 0.99


def grade(task_id: str, episode_result: Dict[str, Any]) -> float:
    """
    Return a score strictly in (0.01, 0.99).

    episode_result keys used:
      - total_reward (float)  — sum of all step rewards
      - victory      (bool)   — whether agent completed all objectives
    """
    total_reward = float(episode_result.get("total_reward", 0.0))
    victory      = bool(episode_result.get("victory", False))

    ranges = _TASK_RANGES.get(task_id, {"min": -18.0, "max": 13.0})
    r_min, r_max = ranges["min"], ranges["max"]

    # Normalise to [0, 1]
    if r_max == r_min:
        normalised = 0.5
    else:
        normalised = (total_reward - r_min) / (r_max - r_min)

    # Partial credit: 1 of 2 vulns fixed → ~0.5, not 0 or 1
    # Victory bonus: push toward 0.95 but never reach 0.99
    if victory:
        normalised = max(normalised, 0.90)

    # STRICTLY clip to open interval — never 0.0 or 1.0
    score = max(_CLIP_MIN, min(_CLIP_MAX, normalised))
    return round(score, 4)


def make_grader(**kwargs: Any) -> "SentinelGrader":
    return SentinelGrader(**kwargs)


class SentinelGrader:
    """OpenEnv-compatible grader class."""
    def __init__(self, **kwargs: Any) -> None:
        self.task_id = kwargs.get("task_id", "")

    def score(self, episode_result: Dict[str, Any]) -> float:
        return grade(self.task_id, episode_result)


if __name__ == "__main__":
    tests = [
        # (task,                  reward,  victory, expected_range)
        ("task_hardcoded_key",    6.0,     True,    (0.01, 0.99)),  # max reward → 0.99
        ("task_hardcoded_key",   -3.0,     False,   (0.01, 0.99)),  # min reward → 0.01
        ("task_hardcoded_key",    1.5,     False,   (0.40, 0.60)),  # mid reward → ~0.5
        ("task_hardcoded_key",    0.95,    False,   (0.01, 0.99)),  # no-victory cap → score in range
        ("task_vulnerable_dep",   3.0,     False,   (0.40, 0.70)),
        ("task_eval_auth_flaw",  13.0,     True,    (0.01, 0.99)),
        ("task_eval_auth_flaw", -18.0,     False,   (0.01, 0.99)),
        ("task_eval_auth_flaw",   0.0,     False,   (0.50, 0.60)),  # ~midpoint
    ]
    print("Grader self-test:")
    all_ok = True
    for task_id, reward, victory, (lo, hi) in tests:
        s = grade(task_id, {"total_reward": reward, "victory": victory})
        ok = (0.0 < s < 1.0) and (lo <= s <= hi)
        status = "OK" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"  [{status}] {task_id:<22} reward={reward:+6.1f} victory={str(victory):<5} → score={s}  (want {lo}–{hi})")
    print("All tests passed!" if all_ok else "SOME TESTS FAILED")
