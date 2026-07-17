"""monty_lab: a task-agnostic sensorimotor learning harness (Monty protocol).

Clean-room implementation of the Thousand Brains custom-application contract
(docs.thousandbrains.org/docs/using-monty-in-a-custom-application): an
Environment yields FEATURES AT POSES as a sensor moves; a LearningModule
learns object reference frames from few episodes and recognises object+pose
by evidence accumulation. The real tbp.monty framework pins heavy deps
(torch-sparse et al., conda-only) that conflict with this repo's venv — these
interfaces mirror its protocol so tbp.monty classes can be adapted in behind
them later without touching any task code.

To train a NEW task: subclass Task in monty_lab/tasks/, yield Episodes, and
run `python -m train.monty_lab.runner learn --task <name>`. Nothing else.
"""

from .protocol import Episode, Observation, Task
from .evidence_lm import EvidenceLM

__all__ = ["Episode", "Observation", "Task", "EvidenceLM"]
