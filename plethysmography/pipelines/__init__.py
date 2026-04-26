"""Top-level experiment drivers. Each module has a ``run()`` function that
performs the full preprocess → analyze → stats → plots pipeline for one
experiment and writes outputs to ``Data/experiment N - …/`` and
``results/experiment N - …/``.

  - :mod:`experiment1` — LR vs HR comparison
  - :mod:`experiment2` — Chronic FFA vs vehicle
  - :mod:`experiment4` — P19 het: survivors vs SUDEP (reuses 1+2 outputs)
"""

from . import experiment1, experiment2, experiment4

__all__ = ["experiment1", "experiment2", "experiment4"]
