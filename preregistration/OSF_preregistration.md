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

## 13. Deviations & amendments (append-only log)

This section is **append-only**: the original plan above is preserved verbatim; every
post-draft change to the analysis, and every departure from the plan, is logged here with a
date and the commit that implements it. Honesty about what actually happened is preferred to
a clean-looking record (cf. §1).

**A1 — Load↔HRV alignment correction (2026-06-19, commit `a837c13`).**
The §5 model formula `Δlndev[t+1] ~ sport*TRIMP[t]` specifies **next-night** alignment (a
day-*t* session affects the *following* night's HRV). The accompanying §5 prose ("day-*t*
load lands on that **same** night's deviation") and the first estimator implementation
instead used **same-night** alignment. On real data the same-night version is wrong: Garmin's
overnight HRV is morning-timestamped, so a session first affects the next night's reading.
Empirically, the deviation the night *after* a session is suppressed while the *same*-night
deviation is not — the latter reflects the pre-training morning state that drives the decision
to train (a "train-when-recovered" behavioural confound), which produced spurious *negative*
cost slopes. The estimators now expose `load_lag` (default `0` = the synthetic-DGP
convention; **real panels use `load_lag=1`**), aligning each session's load to the night it
first affects and the recovery regime one night further. This brings the implementation in
line with the registered `Δlndev[t+1]` formula; where the §5 prose and the §5 formula
conflict, **the formula is authoritative**.

**A2 — Strength training added as a third modeled load type (2026-06-19, commit `2f80f08`).**
§4 declared sport ∈ {triathlon, soccer, rest}. The real panel contains a substantial number
of strength-training sessions (~20% of otherwise-rest days). Folding them into the `rest`
baseline contaminates the recovery-cost contrast (strength carries real autonomic load).
Amendment: `strength` is modeled as its **own** load category — its own cost slope and
recovery τ — keeping the `rest` baseline clean; the H-A1/H-A2 **headline contrasts remain
triathlon-vs-soccer**. `multi_sport` (triathlon races / bricks) maps to `triathlon`.

**A3 — Prior exploratory fitting of the A-layer on real data (2026-06-19).**
§1 and §11 state that at registration the real exposure→outcome relationships had not been
modeled, and that the document would be frozen/submitted before the sport×load→HRV
relationship was estimated. **This was not adhered to for the A-layer.** During pipeline
development (data ingest, schema reconciliation, and the A1 alignment discovery), the H-A1
cost and H-A2 τ models were fit on the **full** real panel — including the most-recent-20%
segment §7 reserved as held-out — and the results were inspected. The A-layer is therefore an
**exploratory, fully-disclosed** analysis, not a blinded confirmatory one. (The A-layer is
structural-posterior estimation and does not rely on an out-of-sample holdout for validity,
but the "not yet fit" / "holdout untouched" claims are no longer true for it.) **Unaffected:**
the prediction layer (H-P1) and its 20% holdout have **not** been fit or inspected; that
layer's pre-registration discipline remains intact and will be honored.

**A4 — TRIMP heart-rate bounds derived from the subject's own data (2026-06-19).**
Banister TRIMP requires resting and maximum HR. To avoid hand-tuned inputs these are set from
the archived data: `hr_rest` = the median of the Garmin daily resting-HR series; `hr_max` =
the maximum observed activity HR (corroborated by several near-maximal soccer sessions rather
than a single spike). The specific values live in the local analysis config and are not
committed (personal-data policy); the derivation rule above makes them reproducible.

**A6 — Prediction layer (H-P1): candidate simplified to an OLS linear model (2026-06-22).**
§5 listed the candidate predictor as "state-space/Kalman + gradient-boosting (lagged
features)". With an effective sample size of only ≈130 independent days (reported next to the
result), a high-variance learner would overfit; the hypothesis is specifically whether
*cross-sport load* adds next-day skill, so the candidate is an OLS linear model = the AR(1)
baseline **plus** today's per-sport TRIMP loads. Baselines (random-walk, AR(1)), the
leakage-safe CPCV evaluation, the embargo (sized from the AR(1)-residual ACF), the 20%
holdout, and the §6 "5th-percentile > 0" decision rule are all as registered. Kalman/GBM
variants are left as optional robustness runs and were not needed: the linear candidate
already returns a null increment (below), so a more flexible model is unlikely to overturn a
null in the direction of *more* skill. Implemented in `garmin_nof1.models.prediction`;
reported by `scripts/prediction_report.py`.

*Result (H-P1, as expected, null):* on the development set AR(1) clearly beats random-walk
(RMSE 0.140 vs 0.168), but adding cross-sport load barely moves it (0.139) and the CPCV skill
improvement has a 5th percentile below 0 — H-P1 **does not beat baseline**, matching the
pre-registered prior. The single holdout evaluation is consistent (a marginal, non-decisive
+0.006 RMSE improvement). Next-day HRV here is dominated by its own autoregression; today's
training load adds no robust incremental prediction.

**A5 — H-D1 rescoped: reconstruction validation + quality audit, not a wrist-vs-chest device
comparison (2026-06-22).**
§3/§5 framed H-D1 as agreement between **wrist** Garmin HR/HRV and a **chest-strap** reference.
On inspecting the archived activity FITs (the chest-strap window, 22 sessions carrying
beat-to-beat RR), the watch logs the paired strap as the **sole** HR source: `record.heart_rate`
equals the RR-derived HR and the session `avg_heart_rate`, and no independent wrist-optical
series is stored. A simultaneous wrist-vs-chest device comparison is therefore **not supported**
by the data and is **not claimed** (demoted per the §8 "missing data → demote the module"
rule). H-D1 is rescoped to what the strap RR does support, and reported as such:
(i) **reconstruction validation** — our RR→mean-HR reconstruction vs Garmin's independent
firmware `avg_heart_rate` over the 22 sessions (Bland-Altman bias + 95% LoA, ICC(2,1), MAPE,
CCC); (ii) **RR data-quality audit** — artifact rate, the RMSSD inflation that artifact
correction removes, and beat coverage, split by sport. "Adequate reconstruction" reuses the
pre-registered §6 thresholds (ICC ≥ 0.75 and MAPE ≤ 10%). Implemented in
`garmin_nof1.eval.agreement` and `garmin_nof1.pipeline.parse_rr.rr_quality`; reported by
`scripts/dlayer_report.py`. This is a measurement-pipeline and data-quality finding scoped to
N=1, not a device-validation claim about the Forerunner's wrist optical sensor.

---

*Amendments A7–A10 were prompted by an adversarial multi-agent correctness audit of the
published analysis (2026-06-22). The audit confirmed 9 issues — one real bug and several
framing/labeling defects; the bottom line was that no scientific conclusion flips, but two
reported numbers needed correction. The audit report is kept locally.*

**A7 — TRIMP double-counting fixed (de-dup by activityId) (2026-06-22).**
`build_daily_panel` concatenated several *overlapping date-range* activity exports and summed
TRIMP keyed only by date, so a workout appearing in more than one archive had its load counted
2–3×. On the real panel this inflated TRIMP ~1.5× on ~90 days that all fell inside the
prediction holdout. Fix: de-duplicate by `activityId` (first sighting) before the fold. Effect:
the H-A1 per-TRIMP cost slopes — which the contaminated panel had deflated to ~0.03 — return to
**~0.039 for both triathlon and soccer (interaction ≈ 0, P ≈ 0.50)**, an even cleaner H-A1
null; the H-P1 verdict stays null. The primary exposure (TRIMP) is the pre-registered headline
variable, so this correctness fix matters even though no conclusion changes.

**A8 — ESS reported on the detrended residual, not the raw series (2026-06-22).**
`evaluate_prediction` computed effective sample size on the raw, trend-laden ln_rmssd; a slow
baseline dominates its autocorrelation and deflates ESS — the exact error the project's own CV
demo warns against. Fixed to compute ESS on the 28-day-detrended deviation. Reported real-data
ESS changes from ≈130 to **≈324** (of 723 dev days). Decision-irrelevant (the H-P1 verdict uses
the CPCV skill distribution, not ESS) but a corrected headline figure.

**A9 — D-layer HR check relabeled as RR-parsing self-consistency, not independent agreement
(2026-06-22).** The D-layer compared our `60000/mean(RR)` to Garmin's firmware `avg_heart_rate`.
Both derive from the **same single chest-strap beat stream**, so the comparison is a near-
arithmetic identity, not independent method/device agreement; the ICC 0.99 is largely
tautological. Reframed: it is an RR-parse/concatenation self-consistency check (it catches gross
parser bugs), the word "independent" is dropped, and ICC 0.99 is no longer presented as passing
the device-agreement adequacy bar. The non-circular D-layer content stands: the RR data-quality
audit (artifact rate, the ~19 ms RMSSD inflation artifact-correction removes, coverage), and the
core constraint that no independent wrist series exists (A5).

**A10 — minor correctness/labeling fixes from the audit (2026-06-22).**
(i) CPCV now passes `purge=1` for the one-step-ahead label, removing a benign train-label/
test-feature shared-value adjacency that contradicted the "leakage-safe" wording (changes no
number). (ii) `recovery_cost`'s reported per-sport session counts `n` now count the *lag-aligned*
contributing session (nonzero load column), matching the docstring under `load_lag=1`. (iii)
`n_backtest_paths` bound tightened to `k < n_groups` to match `combinatorial_purged_splits`.
(iv) `effective_sample_size` docstring corrected (initial-positive-*lag* truncation, not Geyer's
initial-positive-*sequence*). (v) Documented the known mixed-sport-day limitation: the panel
stores one label + whole-day TRIMP, so on ~3% of active days a co-occurring sport's load is
attributed to the dominant sport. None of these change a scientific conclusion.
