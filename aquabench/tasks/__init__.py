"""Task axis: what a head's output means + how it's scored (loss + metric).

Only `semantic_seg` is implemented now; the registry keeps `TaskModule`
task-generic so classification etc. can slot in as a new module here.
"""

from aquabench.tasks.semantic_seg import SemanticSegTask

TASKS = {"semantic_seg": SemanticSegTask}

__all__ = ["TASKS", "SemanticSegTask"]
