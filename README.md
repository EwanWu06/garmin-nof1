# Garmin dual-sport N-of-1

A single-subject (N-of-1) study of **cross-sport recovery** built from ~2.5 years of one
athlete's Garmin (Forerunner 955) data spanning two physiologically opposite sports —
**endurance triathlon** and **intermittent-sprint soccer** — with **strength training** as a
third modeled load.

The point of this project is *not* "I discovered X." It is: **use quantitative methods to
honestly characterize one complex individual system, and state clearly where the conclusions
stop.** Rigor, honest boundaries, a pre-registered analysis with an append-only amendment log,
and a validated, reproducible toolchain are the deliverables. Two of the three headline tests
land on a **null** — and that is reported as the finding, not hidden.

> **TL;DR of the science.** On this person: (D) HRV reconstructed from chest-strap RR agrees
> with Garmin's own firmware to **ICC 0.99**; (A) soccer does **not** cost more next-day vagal
> HRV *per unit of training load* than triathlon (**null**), but its recovery time-constant is
> ~2× longer (strong, not decisive at the 95% bar); (P) cross-sport load adds **no** robust
> next-day HRV prediction over an AR(1) baseline (**null**, as pre-registered).

## The three layers

| Layer | Question | Pre-registered prior | Real-data outcome |
|---|---|---|---|
| **D — Measurement** | Does HRV reconstructed from raw beat-to-beat RR agree with a reference, and how clean is the signal? | Should validate | **Validated** — ICC 0.99 vs Garmin firmware HR; quantified artifact inflation of RMSSD |
| **A — Headline (H-A1)** | Does **soccer cost more next-day vagal HRV per unit load** than triathlon? | Expected positive | **Null** — per-TRIMP cost is the same across sports |
| **A — Headline (H-A2)** | Is the **recovery time-constant τ** sport-specific? | Expected positive | **Soccer ~2× slower** to recover than triathlon; strong (P≈0.97) but just shy of the 95% bar |
| **Prediction (demoted, H-P1)** | Does cross-sport load add next-day-HRV skill **over AR(1) / random-walk**? | Expected null | **Null** — AR(1) beats random-walk, load adds nothing robust |

## Results

### D — Measurement validation (chest-strap window, 22 RR-bearing sessions)
- **Reconstruction is correct.** Our RR→mean-HR pipeline vs Garmin's independent firmware
  `avg_heart_rate` over the same heartbeats: **bias −1.2 bpm, ICC(2,1) 0.990, CCC 0.990,
  MAPE 0.93%** — passes the pre-registered adequacy bar (ICC ≥ 0.75, MAPE ≤ 10%).
- **Quality audit.** Artifact correction removes **~19 ms** of motion-inflated RMSSD; soccer
  (2.7% artifact beats) is noisier than running (1.9%) — field sport HRV needs aggressive
  cleaning, which is *why* the study uses Garmin's clean overnight HRV for the daily panel.
- **Honest scope.** When a chest strap is paired the watch logs it as the *sole* HR source,
  so the FITs carry **no independent wrist-optical series** — a simultaneous wrist-vs-chest
  *device* comparison is not supported and is **not claimed** (demoted; see prereg A5).

### A — Differential recovery (904-day panel; triathlon 114 / soccer 48 / strength 90)
- **H-A1 (cost): null.** Next-night ln-RMSSD cost per 100 TRIMP is ~0.03 for all three
  sports; the soccer−triathlon interaction is ≈0 (P 0.57). Per unit of *HR-based* load, the
  sports tax vagal recovery equally. (Caveat: TRIMP underestimates soccer's intermittent
  sprint load, so "equal per TRIMP" ≠ "equal per session.")
- **H-A2 (recovery speed): suggestive.** Recovery time-constant τ: **triathlon 0.39 d,
  soccer 0.77 d, strength 0.88 d**. Soccer recovers ~2× slower than triathlon
  (P(soccer slower) ≈ 0.97), but the 95% credible interval grazes 0 — **not decisive** at the
  pre-registered threshold; soccer's 48 sessions are the power bottleneck.
- **Strength matters.** Modeling strength as its own load (not folding it into "rest") both
  keeps the headline contrast clean and shows strength carries the longest recovery cost — a
  signal that would vanish if it were lumped into the baseline.

### P — Prediction (demoted falsification)
- Predicting next-day ln-RMSSD: random-walk RMSE 0.168, **AR(1) 0.140** (persistence is
  real), candidate AR(1)+load **0.139**. CPCV skill-improvement 5th percentile < 0 →
  **H-P1 does not beat baseline** (null, matching the prior). Effective sample size ≈ **130**
  independent days out of 723 — the true ceiling on power, reported alongside the result.

## A methodological highlight: HRV timestamp alignment

The first real-data fit produced *negative* recovery costs (training appeared to *raise*
next-day HRV). The cause was not physiology but **timing**: Garmin stamps an overnight HRV
reading to the *morning*, so a day's training first affects the *next* night — while the naïve
model aligned load with the *same* night, the night whose good reading actually *drove the
decision to train* ("train-when-recovered" reverse causation). Aligning load to the night it
affects (`load_lag=1`) flips the costs to the physiological sign and matches the pre-registered
`Δlndev[t+1] ~ TRIMP[t]` formula. This is the kind of confound that only shows up on real data,
and it is logged transparently in the pre-registration (amendment A1).

## Honest constraints (these shape everything)

- **Nightly HRV is a Garmin-derived RMSSD summary**, not raw beats — a *within-subject*
  trend/target, never ground truth (cf. Garmin's mid-tier concordance in the wearable-HRV
  literature).
- **True beat-to-beat RR exists only for the chest-strap window** (HRM-Pro + "Log HRV" during
  activities). Wrist optical does not write usable RR, so RR-level work (the D-layer) is scoped
  to that window — 22 sessions.
- **Observational, no randomization** → causal language stays at the association /
  sensitivity-analysis level. Sport is entangled with season, so a cross-sport difference is a
  within-person association, not a causal sport effect.
- **N-of-1.** Everything is about *this person*; nothing here generalizes to a population.

## Pre-registration and honesty log

`preregistration/OSF_preregistration.md` holds the analysis plan (hypotheses, decision rules,
ROPE, holdout). Because the data already existed, §13 is an **append-only amendments log** —
every post-draft change and every departure from the plan is recorded with a date and the
commit that made it:

- **A1** load↔HRV alignment correction · **A2** strength as a 3rd modeled sport ·
  **A3** full disclosure that the A-layer was fit exploratorily on the whole panel ·
  **A4** TRIMP HR bounds derived from data · **A5** H-D1 rescoped (no wrist-vs-chest) ·
  **A6** prediction candidate simplified to OLS (ESS too small for Kalman/GBM).

The prediction layer kept its discipline: developed on the first 80%, the 20% holdout
evaluated exactly **once**.

## Repository layout

```
src/garmin_nof1/
  data/synthetic.py      # daily panel with a KNOWN generative structure (test substrate)
  eval/
    cv.py                # leakage-safe time-series CV: PurgedWalkForward, CPCV, ESS
    agreement.py         # Bland-Altman / ICC(2,1) / MAPE / CCC (D-layer)
  pipeline/
    ingest_garmin.py     # credential-gated Garmin pull; archives raw JSON/FIT (private)
    garth_cn.py          # Garmin China (connect.garmin.cn) OAuth adapter via garth
    garmin_schema.py     # archived Garmin JSON -> tidy records
    parse_rr.py          # activity-FIT hrv messages -> RR (ms) -> RMSSD/SDNN + quality
    trimp.py             # Banister / Edwards / Garmin training-load variants
    build_panel.py       # tidy, missingness-aware daily panel (synthetic-schema compatible)
  models/
    recovery_cost.py     # H-A1: differential next-night recovery cost (Bayesian conjugate)
    recovery_tau.py      # H-A2: per-sport recovery time-constant (Monte-Carlo from posterior)
    prediction.py        # H-P1: AR(1)/random-walk baselines + load candidate, CPCV + holdout
scripts/
  pull_garmin.py         # one command to pull your own data (US via garminconnect, CN via garth)
  check_panel.py         # build + sanity-check the panel (privacy-safe summary)
  dlayer_report.py       # D-layer reconstruction validation + RR quality audit
  prediction_report.py   # H-P1 holdout-safe skill report
tests/                   # pytest, 132 tests — incl. leakage-injection + estimator recovery
data/                    # gitignored: raw FIT/JSON archive + derived panel (never committed)
preregistration/         # OSF pre-registration + append-only amendment log
```

## Setup

```bash
mamba env create -f environment.yml     # or: conda env create -f environment.yml
conda activate garmin-nof1
pip install -e .
pytest                                  # 132 tests, all on synthetic data — no private data needed
```

Everything that gates a conclusion (the leakage-safe CV, the estimators, the agreement stats)
is validated on a synthetic substrate with known ground truth, so the test suite runs without
any personal data.

## Reproduce on your own Garmin data

Your password is typed interactively and never written to disk; raw data stays under
`data/` (gitignored) and is analyzed locally.

```bash
# 1. Pull (US/global account via garminconnect; China account via garth with --cn)
python scripts/pull_garmin.py --start 2024-01-01 --end 2026-06-15
python scripts/pull_garmin.py --cn --start 2024-01-01 --end 2025-12-15   # China account
python scripts/pull_garmin.py --start 2026-03-01 --end 2026-06-15 --fits # FITs (RR) for D-layer

# 2. Build + sanity-check the combined daily panel
python scripts/check_panel.py

# 3. Run the layers
python scripts/dlayer_report.py        # D: HRV reconstruction validation + RR quality
python scripts/prediction_report.py    # P: H-P1 skill vs baseline (holdout-safe)
```

The A-layer models (`fit_recovery_cost`, `fit_recovery_tau`) take the built panel directly;
HR bounds for TRIMP are derived from your own data (median resting HR; max activity HR) per
prereg A4.

## Status

**Complete.** All three layers are built, validated on synthetic ground truth, and run on the
real combined panel; 132 tests pass. The scientific story is two honest nulls (H-A1, H-P1), one
strong-but-not-decisive effect (H-A2: soccer recovers slower), and a clean measurement
validation (D) — with every analytical decision logged in the pre-registration.
