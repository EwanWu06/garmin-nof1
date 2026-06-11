"""Synthetic daily-panel generator with a KNOWN generative structure.

Why this exists
---------------
Every conclusion in this project is gated by a leakage-safe evaluation scaffold
(`garmin_nof1.eval.cv`) and, later, by models whose job is to *recover* a known
effect. You cannot test either against real Garmin data — there is no ground
truth there. So we generate a daily panel whose data-generating process we wrote
down, and check that the machinery recovers it.

The generative model (kept deliberately simple and documented)
--------------------------------------------------------------
Let ``d[t] = y[t] - b[t]`` be the deviation of nightly ``ln rMSSD`` from a slowly
drifting fitness/seasonal baseline ``b[t]``. Then

    d[t] = phi_regime[t] * d[t-1] - cost[t] + eps[t],   eps ~ N(0, sigma^2)
    cost[t] = beta[sport[t]] * trimp[t] / 100
    y[t] = b[t] + d[t]

* ``phi_regime[t]`` is the AR(1) mean-reversion -> recovery time-constant
  ``tau = -1/ln(phi)``. It can be made **sport-specific** (``phi_soccer`` /
  ``phi_triathlon``): the deviation left by a session decays at that sport's rate
  until the next session, so ``tau`` differs by sport (the H-A2 headline). Both
  default to a single ``phi``, in which case the process is the ordinary single-phi
  AR(1).
* ``beta`` is the next-night ``ln rMSSD`` cost per 100 TRIMP, **sport-specific**.
  The headline ground truth is ``beta_soccer > beta_triathlon > 0`` — i.e. soccer
  costs more vagal HRV per unit of load (the H-A1 headline). Layer A must recover this.

Real-data caveat (documented, not baked in): published evidence is that TRIMP
*underestimates* intermittent-sprint (soccer) load, so on real data part of any
apparent "soccer costs more per TRIMP" effect can be a measurement artifact. Here
the observed ``trimp`` is generated directly and the sport effect lives entirely
in ``beta``, so the synthetic substrate is a clean test of the *estimator*, not a
claim about physiology.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SPORTS = ("rest", "triathlon", "soccer")


@dataclass(frozen=True)
class GroundTruth:
    """The parameters the estimators are supposed to recover.

    Two headline effects can be planted:

    * **H-A1 (cost):** ``beta_soccer > beta_triathlon`` — soccer costs more vagal HRV
      per unit TRIMP. Recovered by :func:`garmin_nof1.models.fit_recovery_cost`.
    * **H-A2 (recovery speed):** ``phi_soccer`` / ``phi_triathlon`` give *sport-specific*
      mean-reversion, so the deviation left by a session decays at a sport-specific rate
      and the recovery time-constant ``tau = -1/ln(phi)`` differs by sport. Recovered by
      :func:`garmin_nof1.models.fit_recovery_tau`. Both default to ``phi`` (no H-A2
      effect), so a default ``GroundTruth()`` reduces exactly to the single-phi process.
    """

    phi: float = 0.75  # AR(1) mean-reversion of ln rMSSD deviation (base/default)
    beta_triathlon: float = 0.06  # next-night ln rMSSD cost per 100 TRIMP (endurance)
    beta_soccer: float = 0.14  # ... after soccer (the headline: soccer > triathlon)
    sigma: float = 0.05  # nightly process noise (ln units)
    baseline: float = 4.0  # mean ln rMSSD  (rMSSD ~ exp(4.0) ~ 55 ms)
    seasonal_amp: float = 0.15  # amplitude of the annual fitness/seasonal drift
    phi_soccer: float | None = None  # post-soccer recovery persistence (defaults to phi)
    phi_triathlon: float | None = None  # post-triathlon recovery persistence (defaults to phi)

    @property
    def tau_days(self) -> float:
        """Recovery time-constant implied by the base phi (days)."""
        return -1.0 / np.log(self.phi)

    def phi_for(self, sport: str) -> float:
        """Recovery persistence governing decay *after* a session of ``sport``.
        Falls back to the base ``phi`` when a sport-specific value is not set (and
        for rest, which never starts a new recovery regime)."""
        if sport == "soccer" and self.phi_soccer is not None:
            return self.phi_soccer
        if sport == "triathlon" and self.phi_triathlon is not None:
            return self.phi_triathlon
        return self.phi

    @property
    def tau_soccer(self) -> float:
        return -1.0 / np.log(self.phi_for("soccer"))

    @property
    def tau_triathlon(self) -> float:
        return -1.0 / np.log(self.phi_for("triathlon"))

    def beta(self, sport: str) -> float:
        return {"rest": 0.0, "triathlon": self.beta_triathlon, "soccer": self.beta_soccer}[sport]


def _draw_sport(rng: np.random.Generator, t: int) -> str:
    """Stochastic daily schedule: triathlon near-daily; soccer ~1-2x/week and
    seasonal (more in-season); the rest are rest days."""
    season = 0.5 * (1 + np.sin(2 * np.pi * (t / 365.0)))  # 0..1 annual cycle
    p_soccer = 0.04 + 0.16 * season  # 4% off-season -> 20% in-season
    u = rng.random()
    if u < p_soccer:
        return "soccer"
    if u < p_soccer + 0.60:
        return "triathlon"
    return "rest"


def _draw_trimp(rng: np.random.Generator, sport: str) -> float:
    if sport == "triathlon":
        return float(rng.gamma(shape=6.0, scale=15.0))  # mean ~ 90
    if sport == "soccer":
        return float(rng.gamma(shape=5.0, scale=14.0))  # mean ~ 70 (observed)
    return 0.0


def make_daily_panel(
    n_days: int = 730,
    seed: int = 0,
    gt: GroundTruth | None = None,
    missing_rate: float = 0.08,
    start: str = "2023-01-01",
) -> pd.DataFrame:
    """Generate a tidy daily panel with known structure.

    Returns a DataFrame indexed 0..n_days-1 with columns:
        date, sport, trimp, sleep_hours, rhr, ln_rmssd, hrv_observed

    ``ln_rmssd`` (and ``rhr``) are set to NaN on a random ``missing_rate`` of
    nights to mimic non-wear; ``hrv_observed`` flags it. ``sport`` and ``trimp``
    are always known (you know what you trained).
    """
    gt = gt or GroundTruth()
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=n_days, freq="D")

    sport = np.empty(n_days, dtype=object)
    trimp = np.zeros(n_days)
    for t in range(n_days):
        sport[t] = _draw_sport(rng, t)
        trimp[t] = _draw_trimp(rng, sport[t])

    t_idx = np.arange(n_days)
    baseline = gt.baseline + gt.seasonal_amp * np.sin(2 * np.pi * t_idx / 365.0)

    d = np.zeros(n_days)  # deviation of ln rMSSD from baseline
    eps = rng.normal(0.0, gt.sigma, size=n_days)
    cost = np.array([gt.beta(sport[t]) * trimp[t] / 100.0 for t in range(n_days)])
    d[0] = eps[0]
    # ``regime_phi`` is the recovery persistence of the most recent session, so a
    # deviation left by a session decays at that sport's rate until the next session.
    # With default (None) per-sport phi this is constant ``gt.phi`` — i.e. the original
    # single-phi AR(1) process, bit-for-bit (no extra RNG draws).
    regime_phi = gt.phi_for(sport[0]) if sport[0] != "rest" else gt.phi
    for t in range(1, n_days):
        d[t] = regime_phi * d[t - 1] - cost[t] + eps[t]
        if sport[t] != "rest":
            regime_phi = gt.phi_for(sport[t])
    ln_rmssd = baseline + d

    # Resting HR rises when recovery (d) is suppressed; sleep mildly load-sensitive.
    rhr = 50.0 - 8.0 * d + rng.normal(0.0, 1.0, size=n_days)
    sleep_hours = 7.6 - 0.004 * trimp + rng.normal(0.0, 0.5, size=n_days)

    df = pd.DataFrame(
        {
            "date": dates,
            "sport": pd.Categorical(sport, categories=SPORTS),
            "trimp": trimp,
            "sleep_hours": sleep_hours,
            "rhr": rhr,
            "ln_rmssd": ln_rmssd,
        }
    )

    # Non-wear missingness on the optical-derived channels.
    miss = rng.random(n_days) < missing_rate
    df["hrv_observed"] = ~miss
    df.loc[miss, ["ln_rmssd", "rhr"]] = np.nan
    return df


def make_rr_series(
    n_beats: int = 300,
    mean_hr: float = 55.0,
    rmssd_target: float = 45.0,
    seed: int = 0,
) -> np.ndarray:
    """Synthetic beat-to-beat RR intervals (milliseconds) for testing the FIT
    HRV parser and RMSSD reconstruction (D-layer). Successive-difference SD is
    tuned so the realised RMSSD is close to ``rmssd_target``."""
    rng = np.random.default_rng(seed)
    mean_rr = 60_000.0 / mean_hr
    # RMSSD = sqrt(mean(successive-diff^2)). For i.i.d. levels with SD s the
    # successive differences have SD sqrt(2)*s, so pick s = target / sqrt(2).
    noise = rng.normal(0.0, rmssd_target / np.sqrt(2.0), size=n_beats)
    rr = mean_rr + noise - noise.mean()
    return np.clip(rr, 300.0, 2000.0)
