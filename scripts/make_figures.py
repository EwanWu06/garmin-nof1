#!/usr/bin/env python
"""Generate the README figures from the real combined panel (privacy-safe aggregates only).

Reads local archived data, fits the layers, and writes four publication-style PNGs to
``docs/figures/``. Every figure shows only aggregate effect estimates / group summaries /
distributions — no raw daily physiology — so the outputs are safe to commit. Usage:

    python scripts/make_figures.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import glob

import matplotlib.pyplot as plt
import numpy as np

from garmin_nof1.eval.cv import combinatorial_purged_splits
from garmin_nof1.models import fit_recovery_cost, fit_recovery_tau
from garmin_nof1.models._common import deviation
from garmin_nof1.models.prediction import _fold_rmse, build_supervised, holdout_split
from garmin_nof1.pipeline.build_panel import build_daily_panel
from garmin_nof1.pipeline.parse_rr import reconstruct_rr_ms, rr_quality

HR_REST, HR_MAX = 49.0, 211.0
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
        f"interaction(soc−tri)≈0, P={cost.prob_interaction_positive:.2f}"
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
        f"P(soccer slower)={tau.prob_tau_soccer_longer:.2f} (CI grazes 0)"
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


def fig_d_quality():
    """D-layer: motion artifact inflates activity RMSSD (raw vs corrected), by sport."""
    warnings.filterwarnings("ignore")
    from fitparse import FitFile

    by_sport: dict[str, list] = {}
    for d in RAW_DIRS:
        for p in glob.glob(f"{d}/activities/*.fit"):
            ff = FitFile(p)
            rr = reconstruct_rr_ms([m.get_value("time") for m in ff.get_messages("hrv")])
            if rr.size < 2:
                continue
            sess = next(iter(ff.get_messages("session")), None)
            sport = (sess.get_value("sport") if sess else None) or "?"
            secs = sess.get_value("total_timer_time") if sess else None
            by_sport.setdefault(sport, []).append(rr_quality(rr, session_seconds=secs))

    sports = [s for s in ["running", "soccer"] if s in by_sport]
    raw = [float(np.mean([q.rmssd_raw for q in by_sport[s]])) for s in sports]
    corr = [float(np.mean([q.rmssd_corrected for q in by_sport[s]])) for s in sports]
    art = [float(np.mean([q.artifact_rate for q in by_sport[s]]) * 100) for s in sports]
    n = [len(by_sport[s]) for s in sports]

    x = np.arange(len(sports))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar(x - w / 2, raw, w, label="raw RMSSD (motion-inflated)", color="#fdae6b", ec="k", lw=0.5)
    ax.bar(x + w / 2, corr, w, label="artifact-corrected RMSSD", color="#31a354", ec="k", lw=0.5)
    for i in range(len(sports)):
        ax.annotate(f"{art[i]:.1f}% beats\nflagged\n(n={n[i]})", (i, max(raw[i], corr[i])),
                    textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(sports)
    ax.set_ylabel("RMSSD (ms)")
    ax.set_title(
        "D-layer: motion artifact inflates activity RMSSD ~3×\n"
        "(field-sport HRV needs aggressive cleaning)"
    )
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(raw) * 1.35)
    _save(fig, "d_layer_quality.png")


def fig_prediction(panel):
    """H-P1: AR(1) beats random-walk; cross-sport load adds no robust skill (CPCV straddles 0)."""
    dev_df, _ = holdout_split(panel, frac=0.2)
    sup = build_supervised(dev_df)
    load_cols = [c for c in sup.columns if c.startswith("load_")]
    splits = combinatorial_purged_splits(len(sup), n_groups=6, n_test_groups=2, embargo=1, purge=1)
    rmse = {"random_walk": [], "ar1": [], "candidate": []}
    imp = []
    for tr, te in splits:
        fold = _fold_rmse(sup, load_cols, tr, te)
        for m in rmse:
            rmse[m].append(fold[m])
        imp.append(fold["ar1"] - fold["candidate"])
    means = {m: float(np.mean(v)) for m, v in rmse.items()}
    imp = np.asarray(imp)
    p05 = float(np.quantile(imp, 0.05))

    fig, (axr, axd) = plt.subplots(1, 2, figsize=(9.5, 4.2))
    labels = ["random\nwalk", "AR(1)", "AR(1)+\nload"]
    axr.bar(range(3), [means["random_walk"], means["ar1"], means["candidate"]],
            color=["#9e9e9e", "#2c7fb8", "#e6550d"], edgecolor="k", lw=0.5)
    axr.set_xticks(range(3))
    axr.set_xticklabels(labels)
    axr.set_ylabel("next-day ln-rMSSD RMSE (CPCV mean)")
    axr.set_ylim(min(means.values()) * 0.95, max(means.values()) * 1.02)
    axr.set_title("AR(1) beats random-walk;\nload barely moves it")

    axd.hist(imp, bins=10, color="#9ecae1", edgecolor="k", lw=0.5)
    axd.axvline(0, color="k", lw=1.2, label="no improvement")
    axd.axvline(p05, color="#e6550d", lw=1.5, ls="--", label=f"5th pct = {p05:+.4f}")
    axd.set_xlabel("skill improvement  RMSE(AR1) − RMSE(candidate)")
    axd.set_ylabel("CPCV splits")
    axd.set_title("H-P1: NULL\n5th percentile < 0 → load adds no robust skill")
    axd.legend(fontsize=8.5)
    _save(fig, "prediction.png")


def main():
    panel = build_daily_panel(RAW_DIRS, hr_rest=HR_REST, hr_max=HR_MAX, sex="M")
    print(f"panel: {len(panel)} days")
    fig_a_layer(panel)
    fig_alignment(panel)
    fig_d_quality()
    fig_prediction(panel)
    print("done.")


if __name__ == "__main__":
    main()
