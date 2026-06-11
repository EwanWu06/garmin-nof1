"""garmin_nof1 — a single-subject (N-of-1) study of cross-sport recovery.

Layers (see README and the plan):
    data/      synthetic generators + (gitignored) real-data loaders
    pipeline/  raw FIT/Connect exports -> tidy, missingness-aware daily panel
    eval/      leakage-safe time-series cross-validation scaffold
    models/    differential-recovery-cost model + demoted AR(1)-baseline prediction
"""

__version__ = "0.1.0"
