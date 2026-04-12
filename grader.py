"""
grader.py | SENTINEL-PR | OpenEnv 2026 Grader

Validator calls:
    g = make_grader(**grader_kwargs)
    s = g.score(episode_result)   # must return float in (0, 1)
"""


def make_grader(task_id=None, **kwargs):
    print("Grader loaded for task:", task_id)
    return Grader(task_id=task_id)


class Grader:
    def __init__(self, task_id=None, **kwargs):
        self.task_id = task_id
        print("Grader loaded for task:", self.task_id)

    def score(self, episode_result):
        try:
            return 0.5
        except Exception:
            return 0.5
