# Garmin dual-sport N-of-1

A single-subject (N-of-1) study of **cross-sport recovery** using ~2 years of one
athlete's Garmin (Forerunner 955) data spanning two physiologically opposite sports —
**endurance triathlon** and **intermittent-sprint soccer**.

The point of this project is *not* "I discovered X." It is: **use quantitative methods to
honestly characterize one complex individual system, and state clearly where the
conclusions stop.** Rigor, honest boundaries, and a validated, reproducible toolchain are
the deliverables.

## The core idea (three layers)

| Layer | Role | Can it fail to a null? |
|---|---|---|
| **D — Measurement layer** | A validated, missingness-aware pipeline (raw FIT + chest-strap RR + logs → tidy daily panel) **plus** a pre-registered validation of derived metrics against transparent references. The artifact *is* the result. | No — it ships a tool + a data-quality report card. |
| **A — Headline science** | Sport-type as a within-person alternating "treatment": does **soccer cost more next-day vagal HRV per unit of training load** than triathlon, and is the recovery time-constant sport-specific? | Near-structurally positive. |
| **Prediction (demoted)** | The incremental predictive value of cross-sport load **over a proper AR(1) / random-walk baseline**, pre-registered as a falsification test. | A null here is an honest finding, not a failure. |

Optional, data-permitting extensions: exploratory directed-coupling discovery (PCMCI+,
surrogate-gated), a Banister fitness–fatigue ill-posedness replication, anomaly/early-warning
detection. See `docs/PLAN.md`.

## Honest constraints (these shape everything)

- **Nightly HRV is a Garmin-derived RMSSD summary**, not raw beats — fine as a *within-subject*
  trend/target, never as ground truth (cf. Porter & Flatt 2026; Dial et al. 2025: Garmin
  mid-to-low concordance, ahead of Polar wrist, behind Oura/Whoop).
- **True beat-to-beat RR exists only for the last ~3 months** (chest strap + "Log HRV" on,
  during activities). Wrist optical does not write usable RR; nightly Enhanced BBI and the
  Garmin Health API are not available to individuals. So RR-level work (coupling, the D-layer
  HRV validation) is scoped to that window.
- **Observational, no randomization** → causal language stays at the sensitivity-analysis /
  association level (Daza's framework as estimand scaffold, not a causal license).

## Repository layout

```
src/garmin_nof1/
  data/synthetic.py     # daily panel with a KNOWN generative structure (test substrate)
  pipeline/             # raw Garmin exports -> tidy daily panel  (touches private data)
  eval/cv.py            # leakage-safe time-series CV (built first — gates every conclusion)
  models/               # recovery-cost model + AR(1)-baseline prediction
tests/                  # pytest; includes a leakage-injection test for the CV scaffold
data/                   # gitignored: raw FIT/JSON archive + derived DB/panel
docs/PLAN.md            # the full phased plan
preregistration/        # OSF pre-registration draft
```

## Setup

```bash
mamba env create -f environment.yml     # or: conda env create -f environment.yml
conda activate garmin-nof1
pip install -e .
pytest                                  # runs on synthetic data — no private data needed
```

## Status

Bootstrapping. Built first (and independent of any private data): the reproducible
environment, the synthetic data substrate, and the **leakage-safe CV scaffold** — because
it decides whether every later conclusion can be trusted.

## Pipeline (Phase 0)

`src/garmin_nof1/pipeline/`:
- `parse_rr.py` — parse activity-FIT `hrv` messages → RR (ms) → RMSSD/SDNN (D-layer).
- `trimp.py` — Banister (primary) / Edwards / Garmin-load training-load variants.
- `build_panel.py` — tidy, missingness-aware daily panel (same schema as the synthetic
  panel, so real data is drop-in for the recovery models).
- `ingest_garmin.py` — credential-gated Garmin Connect pull; archives raw JSON/FIT to
  `data/raw/` (gitignored). Set `GARMIN_EMAIL` / `GARMIN_PASSWORD` in `.env` and run it
  yourself — it analyzes your data locally; nothing personal is committed.
