# Pre-registration — A single-subject (N-of-1) study of cross-sport recovery

**Author:** Ewan Wu
**Date:** _(to be timestamped on OSF before any outcome-relationship modeling)_
**Status:** Draft. Will be frozen at a named git commit and submitted to OSF **before** the sport×load → HRV relationship is estimated on real data.

---

## 1. Study type and a required disclosure

This is an **observational, non-randomized, single-subject (N-of-1)** study using one
person's own wearable data. It is **not** a randomized N-of-1 trial.

**The data already exist** (~2 years of the author's Garmin Forerunner 955 records; ~3
months of chest-strap beat-to-beat RR). This is therefore a pre-registration of an
*analysis plan on pre-existing data*, which provides weaker inferential protection than
prospective registration. We mitigate this honestly rather than hide it:

- §11 discloses exactly what has and has not been looked at.
- The evaluation scaffold, synthetic-data work, and pipeline were developed and tested on
  **synthetic data only**; at registration the real exposure→outcome relationships have not
  been modeled.
- A temporally **held-out final segment** (the most recent 20% of days) is reserved and
  will not be inspected until the primary models are frozen.

## 2. Background (brief)

Published comparisons of how different sports tax recovery are **between-person** (different
athletes do different sports), so they are confounded by who plays what. This subject does
**both** endurance triathlon and intermittent-sprint soccer, which removes that confound for
a within-person contrast. Nightly HRV is a Garmin-derived RMSSD summary (not raw beats); it
is used only as a within-subject signal, never as ground truth. The prediction literature
shows next-day HRV is dominated by its own recent values (mean reversion) and is hard to
predict beyond an autoregressive baseline; we treat that as the honest prior.

## 3. Hypotheses

| ID | Layer | Statement (directional) |
|----|-------|-------------------------|
| **H-A1** | A (primary) | Within this individual, the next-night drop in lnRMSSD **per unit TRIMP** is **larger after soccer than after triathlon** (sport×TRIMP interaction on the cost scale > 0). |
| **H-A2** | A | The HRV/RHR recovery time-constant **τ differs** between soccer and triathlon. |
| **H-P1** | Prediction (demoted, falsification) | Cross-sport load features add **incremental** next-day-lnRMSSD skill **over an AR(1)/random-walk baseline**. *Pre-registered prior: the increment is small or null.* |
| **H-D1** | D (measurement) | Activity-context Garmin HR / derived HRV agrees with a chest-strap reference **within pre-specified bounds**. |

## 4. Variables (operational definitions)

- **Primary outcome** — nightly `ln(RMSSD)` from Garmin HRV Status. Modeled in **deviation
  form**: residual after subtracting a centered 28-day rolling baseline (detrending is
  mandatory; the raw series is trend-dominated — see the ESS demo).
- **Exposure** — per-session `TRIMP` (primary variant pre-declared: HR-based Banister TRIMP;
  robustness across Edwards and Garmin training-load variants). **Sport label** ∈
  {`triathlon` (endurance: swim/bike/run), `soccer` (intermittent), `rest`}.
- **Covariates** — sleep duration, resting HR, day-of-week, season (annual sinusoid),
  data-source / firmware-era indicator.
- **Negative-control outcome** — sleep duration: must **not** show the sport×TRIMP
  interaction (guards against a generic "hard day" effect masquerading as a sport effect).

## 5. Analysis plan

**A — headline (differential recovery cost).** Bayesian within-person hierarchical model:

```
Δlndev[t+1] ~ sport * TRIMP[t] + lagged_load[t-1..t-k] + sleep[t] + dow + season + AR(1) residual
```

Weakly-informative priors; report the **posterior** of the `sport×TRIMP` interaction and
posterior credible intervals (no p-values). _Indexing convention:_ nightly `lnRMSSD` is the
sleep that **follows** a day's training, so day-*t* load lands on that same night's deviation;
with the AR(1) residual term, modeling the deviation *level* on its lag-1 plus contemporaneous
load is equivalent to the `Δlndev` change form above — both encode "day-*t* load → the
following night's HRV, with multi-day carry-over through the AR persistence." τ per sport from
an exponential decay of the post-session lnRMSSD/RHR deviation back to baseline (a **separate**
estimator from the H-A1 interaction model).

_Decision rule and ROPE are evaluated at the 95% level (§6)._ The H-A1 interaction estimator
is implemented and validated on synthetic data; on real data it adds the `sleep`, `dow`,
`season`, and distributed-lag covariates above. The H-A2 per-sport τ estimator is registered
here but estimated separately.

**Prediction — demoted (falsification).** State-space/Kalman + gradient-boosting (lagged
features) vs **AR(1)** and **random-walk** baselines, evaluated with `PurgedWalkForwardSplit`
(embargo sized from the residual ACF decorrelation time) and **CPCV** as a secondary variance
diagnostic. Report skill relative to baselines alongside the **effective sample size**.

**D — measurement validation.** On the ~3-month chest-strap window: Bland-Altman bias + 95%
limits of agreement, ICC(2,1), MAPE, CCC on paired activity windows; reconstruct RMSSD from
raw RR and compare to Garmin's value against the chest-strap reference.

## 6. Inference / decision criteria (pre-specified)

- **H-A1 supported** iff posterior `P(interaction > 0) ≥ 0.95` **and** the 95% credible
  interval excludes a ROPE of ±0.02 ln-units/100-TRIMP around 0.
- **H-A2 supported** iff the 95% CrI of `(τ_soccer − τ_triathlon)` excludes 0.
- **H-P1 "beats baseline"** iff the CPCV distribution of skill improvement over AR(1) is
  positive with its **5th percentile > 0**; otherwise reported as **null** (the expected,
  honestly-reported outcome).
- **H-D1 "adequate agreement"** iff `ICC ≥ 0.75` **and** `MAPE ≤ 10%`; otherwise reported as
  inadequate (itself a finding, scoped to N=1 and not a device-validation claim).

## 7. Sampling and data window

All available days are used; the two sports are analyzed in their own temporal blocks (no
pooling across regimes). Both **nominal N** and **effective N** are reported. The most recent
20% of days are held out from primary-model development.

## 8. Stopping rules / decision gates

- If the CPCV skill distribution for prediction overlaps the AR(1) baseline → stop adding
  model complexity; report the null and pivot effort to the measurement/EDA contributions.
- Advance to coupling/extension claims only if a baseline is beaten by a margin exceeding the
  CPCV path variance.
- If a module's data are missing (e.g. a repeatable performance proxy for the Banister
  replication, or labeled illness days for anomaly detection) → that module is demoted to
  optional and explicitly reported as not run.

## 9. Confounding and sensitivity analyses

- A confounder DAG is reported, enumerating **measured** (load, sleep, RHR, season, sport)
  vs **unmeasured/noisy** (diet, alcohol, illness onset, acute life stress, time-of-day, the
  social context of recreational soccer).
- Sensitivity: E-value for the primary association, negative-control outcome, and a
  tipping-point analysis.
- Stated plainly: the no-unmeasured-confounding assumption is **untestable** and implausible
  here; causal language is restricted to the sensitivity-analysis level. Sport-type is
  entangled with season, so a cross-sport difference is a within-person association, not a
  causal sport effect.

## 10. Departures from reporting standards

Reported against CENT 2015 and SCRIBE 2016 with the unmet items marked explicitly:
**randomization, allocation concealment, and washout** cannot be satisfied by a passive
observational series. The SCED Scale is used as a self-appraisal checklist.

## 11. Prior knowledge of the data (transparency)

As the device wearer, the author is **not blind** to broad day-to-day Garmin app summaries
(HRV-status bands, training-readiness, Body Battery) seen during normal use. However, at
registration the **formal exposure→outcome models have not been fit**. EDA performed before
registration is limited to: missingness counts, marginal distributions, and sport/session
counts. No `sport×load → HRV` relationship, no τ comparison, and no prediction skill has been
estimated. The held-out final 20% has not been inspected at all.

## 12. Analysis code and freeze

The analysis pipeline and the leakage-safe CV scaffold are versioned in git; the repository
will be frozen at a named commit (hash recorded here on submission) before any outcome model
is fit. The CV scaffold is already validated on synthetic data (test suite passing).
