#!/usr/bin/env python
"""Generate the README figures from the real combined panel (privacy-safe aggregates only).

Reads local archived data, fits the A-layer, and writes two PNGs to ``docs/figures/``. Both
show only aggregate effect estimates / group means, no raw daily physiology. Usage:

    python scripts/make_figures.py --hr-rest <resting HR> --hr-max <max HR>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from garmin_nof1.models import fit_recovery_cost, fit_recovery_tau
from garmin_nof1.models._common import deviation
from garmin_nof1.pipeline.build_panel import build_daily_panel

RAW_DIRS = ["data/raw", "data/raw_cn"]
OUT = Path("docs/figures")
COLOR = {"rest": "#9e9e9e", "triathlon": "#2c7fb8", "soccer": "#e6550d", "strength": "#6a51a3"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 140})


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def fig_a_layer(panel):
    """Two panels: H-A1 per-TRIMP cost (null) and H-A2 recovery tau (soccer slower)."""
    cost = fit_recovery_cost(panel, load_lag=1)
    tau = fit_recovery_tau(panel, load_lag=1)
    sports = ["triathlon", "soccer", "strength"]
    fig, (axc, axt) = plt.subplots(1, 2, figsize=(9.5, 4.2))

    for i, s in enumerate(sports):
        lo, hi = cost.cost_slope_ci[s]
        m = cost.cost_slope[s]
        axc.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="o", color=COLOR[s], capsize=5, ms=8)
    axc.axhline(0, color="k", lw=0.8, ls="--", alpha=0.6)
    axc.set_xticks(range(len(sports)))
    axc.set_xticklabels(sports)
    axc.set_ylabel("ln-rMSSD cost per 100 TRIMP\n(positive = HRV suppressed)")
    axc.set_title(
        f"H-A1: recovery cost — NULL\n"
        f"posterior P(soc−tri>0)={cost.prob_interaction_positive:.2f}"
    )

    for i, s in enumerate(sports):
        lo, hi = tau.tau_ci[s]
        m = tau.tau[s]
        axt.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="o", color=COLOR[s], capsize=5, ms=8)
    axt.set_xticks(range(len(sports)))
    axt.set_xticklabels(sports)
    axt.set_ylabel("recovery time-constant τ (days)")
    axt.set_title(
        f"H-A2: soccer recovers ~2× slower\n"
        f"posterior P(soccer slower)={tau.prob_tau_soccer_longer:.2f} (95% CrI grazes 0)"
    )
    _save(fig, "a_layer.png")


def fig_alignment(panel):
    """Same-night vs next-night HRV deviation by sport — the timestamp-alignment confound."""
    dev = deviation(panel, "ln_rmssd", None, 28)
    sport = panel["sport"].astype(object).to_numpy()
    nxt = dev.shift(-1)
    sports = ["rest", "triathlon", "soccer", "strength"]
    same = [float(dev[sport == s].mean()) for s in sports]
    after = [float(nxt[sport == s].mean()) for s in sports]

    x = np.arange(len(sports))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.bar(x - w / 2, same, w, label="same night  dev[t]  (drove the decision to train)",
           color="#bdbdbd", edgecolor="k", lw=0.5)
    ax.bar(x + w / 2, after, w, label="next night  dev[t+1]  (the actual recovery cost)",
           color=[COLOR[s] for s in sports], edgecolor="k", lw=0.5)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(sports)
    ax.set_ylabel("mean HRV deviation (ln-rMSSD)")
    ax.set_title("Why same-night alignment gave spurious negative costs\n"
                 "training days read HIGH the same night (train-when-recovered), LOW the next")
    ax.legend(fontsize=9, loc="lower left")
    _save(fig, "alignment.png")


def main():
    parser = argparse.ArgumentParser(description="Regenerate the README figures.")
    parser.add_argument("--hr-rest", type=float, required=True, help="resting HR for TRIMP")
    parser.add_argument("--hr-max", type=float, required=True, help="maximum HR for TRIMP")
    parser.add_argument("--sex", default="M", choices=["M", "F"], help="Banister TRIMP weighting")
    args = parser.parse_args()
    panel = build_daily_panel(RAW_DIRS, hr_rest=args.hr_rest, hr_max=args.hr_max, sex=args.sex)
    print(f"panel: {len(panel)} days")
    fig_a_layer(panel)
    fig_alignment(panel)
    print("done.")


if __name__ == "__main__":
    main()
