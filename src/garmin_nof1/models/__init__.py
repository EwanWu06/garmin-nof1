"""Layer A models: within-person estimators that must recover a known effect on
the synthetic substrate before they are trusted on real Garmin data.

Also hosts the demoted prediction layer (H-P1), whose job is the opposite: to *fail* to
beat an AR(1)/random-walk baseline unless real next-day skill exists."""

from garmin_nof1.models.prediction import (
    PredictionResult,
    build_supervised,
    evaluate_prediction,
    holdout_split,
)
from garmin_nof1.models.recovery_cost import RecoveryCostFit, fit_recovery_cost
from garmin_nof1.models.recovery_tau import RecoveryTauFit, fit_recovery_tau

__all__ = [
    "PredictionResult",
    "RecoveryCostFit",
    "RecoveryTauFit",
    "build_supervised",
    "evaluate_prediction",
    "fit_recovery_cost",
    "fit_recovery_tau",
    "holdout_split",
]
