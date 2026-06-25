# Project guide & collaboration model

Single-subject (N-of-1) study of cross-sport recovery from the author's own Garmin data.
This file records how the project is worked on, and the division of labor in its
AI-assisted workflow. It is the source the README's "Notes on tooling" section points to.

## Collaboration model (who does what)

This project is built with the **Claude Code CLI**, with an explicit, honest division of
labor:

- **The author (Ewan Wu) directs and decides.** Goals, scope, constraints, the key modeling
  and design calls (e.g. modeling strength training as its own load type; running the analysis
  on real personal data; commissioning the adversarial correctness audit), the standard of
  rigor (pre-registration, leakage-safe evaluation), and the requirement that results —
  including nulls — be reported honestly. Proposed interpretations are treated as hypotheses
  to verify, not conclusions to adopt.
- **Claude Code proposes methods and implements.** It translates the author's specifications
  into code, proposes specific methods, runs analyses, produces figures and documentation, and
  commits with `Co-authored-by` trailers under the author's direction and review.

The aim is not to hide the AI's role but to make it legible: directing and auditing a capable
AI toward a trustworthy, honestly-reported result is treated as a first-class skill here.

## Working conventions

- **TDD.** New behavior gets a failing test first, then the minimal implementation. The suite
  runs entirely on a synthetic substrate with known ground truth — no personal data needed.
- **Lint clean.** `ruff` with `E,F,I,W,UP,B` at line length 100; no `# noqa` suppressions.
- **Privacy.** Never commit anything under `data/` (raw FIT/JSON, derived panels) or any
  credentials/tokens; personal physiology stays local. The analysis is run locally.
- **Pre-registration is append-only.** `preregistration/OSF_preregistration.md` §13 logs every
  deviation with a date and commit; the original plan (§1–§12) is never rewritten.
- **Honesty over polish.** Null and inconclusive results are reported as findings; corrections
  (including bugs found by self-audit) are logged, not quietly fixed.
- **Environment.** conda env `garmin-nof1` (`environment.yml`); `pip install -e ".[dev]"` for
  the test path. Figures: `scripts/make_figures.py` (the `figures` extra).
