"""Efficiency summary from already-saved KO/KI results — no model calls.

Vocabulary: the only two effects are necessity (KO) and sufficiency (KI); the
% of WT versions divide by wt_signal. On disk the sufficiency-% column is still
named `recovery` for back-compat, but every figure/label reads "sufficiency".

Reads results/per_sequence.csv (the `recovery` column = sufficiency % of WT,
plus total_bp per sequence per method) and adds the two scalars the per_method
table never wrote out:

    suff_per_bp = sufficiency / total_bp      (per 1 bp; x10 for fly's per-10bp)
    closeness   = |sufficiency - 1|           (0 = reproduces WT exactly)

Figure (results/figures/efficiency_summary.png):
    A  bp-vs-sufficiency tradeoff, annotated seqs — median marker + IQR cross per method
    B  median sufficiency / bp  (efficiency), horizontal bars + IQR
    C  median closeness |suff-1|, horizontal bars + IQR

Style (palette, markers, names, order) reused verbatim from visualize.py so the
figure sits next to the summary_<bin>.png panels.

    python efficiency_summary.py
"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config as C

# Style mirrored from visualize.py so this figure sits next to summary_<bin>.png.
FIGURE_ORDER = [
    "two_stage", "two_stage_ism", "naive", "recursive_seqlets", "tfmodisco_seqlets",
    "modiscolite_seqlets", "jaspar_fimo", "modisco_hits_mm10",
]
DISPLAY_NAMES = {
    "two_stage":           "Two-stage · foreground (ours)",
    "two_stage_ism":       "Two-stage · raw ISM (ours)",
    "naive":               "Naive thresholding",
    "recursive_seqlets":   "Recursive seqlets (tangermeme)",
    "tfmodisco_seqlets":   "TF-MoDISco seqlets (tangermeme)",
    "modiscolite_seqlets": "TF-MoDISco seqlets (modiscolite)",
    "jaspar_fimo":         "JASPAR motif scan",
    "modisco_hits_mm10":   "TF-MoDISco patterns (min 10)",
}
COLORS = {
    "two_stage":           "#111111",
    "two_stage_ism":       "#b22222",
    "naive":               "#9467bd",
    "recursive_seqlets":   "#1f77b4",
    "tfmodisco_seqlets":   "#ff7f0e",
    "modiscolite_seqlets": "#2ca02c",
    "jaspar_fimo":         "#8c564b",
    "modisco_hits_mm10":   "#e377c2",
}
MARKERS = {
    "two_stage": "*", "two_stage_ism": "X", "naive": "o", "recursive_seqlets": "s",
    "tfmodisco_seqlets": "^", "modiscolite_seqlets": "D", "jaspar_fimo": "v",
    "modisco_hits_mm10": "P",
}
FS_LABEL, FS_TICK, DPI = 15, 13, 150

FIGDIR = C.RESULTS / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)


def load():
    df = pd.read_csv(C.RESULTS / "per_sequence.csv")
    for m in ("two_stage", "two_stage_ism"):     # our runs, each in its own file
        ts = C.RESULTS / m / f"per_sequence_{m}.csv"
        if ts.exists():
            df = pd.concat([df, pd.read_csv(ts)], ignore_index=True)
    df = df[df["method"].isin(FIGURE_ORDER)].copy()
    ann = df[df["total_bp"] > 0].copy()          # annotated sequences only
    ann["rec_per_bp"] = ann["recovery"] / ann["total_bp"]
    ann["closeness"] = (ann["recovery"] - 1.0).abs()
    return df, ann


def _med_iqr(g, col):
    return g[col].median(), g[col].quantile(.25), g[col].quantile(.75)


def _tradeoff(ax, ann):
    """Median annotation length (x) vs median sufficiency % of WT (y), IQR cross per
    method. Over ANNOTATED sequences only. Plain names → shared bottom legend."""
    for m in FIGURE_ORDER:
        g = ann[ann.method == m]
        if g.empty:
            continue
        mx, mb_lo, mb_hi = _med_iqr(g, "total_bp")
        my, r_lo, r_hi = _med_iqr(g, "recovery")
        ax.plot([mb_lo, mb_hi], [my, my], color=COLORS[m], lw=1.5, alpha=0.5, zorder=4)
        ax.plot([mx, mx], [r_lo, r_hi], color=COLORS[m], lw=1.5, alpha=0.5, zorder=4)
        ax.scatter([mx], [my], s=200, color=COLORS[m], marker=MARKERS[m],
                   edgecolor="white", linewidth=1.8, zorder=6, label=DISPLAY_NAMES[m])
    ax.axhline(0.9, color="#aaaaaa", ls="--", lw=1.0, zorder=2)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("Median annotation length (bp)", fontsize=FS_LABEL - 1)
    ax.set_ylabel("Median sufficiency (% of WT)", fontsize=FS_LABEL - 1)
    ax.set_title("Sufficiency vs length  ·  upper-left = efficient", fontsize=FS_LABEL, pad=10)
    ax.tick_params(labelsize=FS_TICK)
    ax.margins(x=0.12, y=0.14)
    ax.grid(alpha=0.18)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _bars(ax, ann, col, xlabel, title, pct=False, fmt="{:.4f}", counts=None,
          ylabels=True):
    """Horizontal median bars + IQR whisker, one row per method (FIGURE_ORDER).

    Over ANNOTATED sequences only (total_bp > 0). `counts` maps method ->
    "n/n_total" and is shown once (on the labelled left panel) so coverage is
    explicit; set ylabels=False on aligned sibling panels to drop the repeat."""
    order = [m for m in FIGURE_ORDER if (ann.method == m).any()]
    y = np.arange(len(order))[::-1]
    meds, los, his = [], [], []
    for m in order:
        md, lo, hi = _med_iqr(ann[ann.method == m], col)
        meds.append(md); los.append(lo); his.append(hi)
    meds, los, his = map(np.array, (meds, los, his))
    ax.barh(y, meds, color=[COLORS[m] for m in order], alpha=0.9, zorder=3,
            height=0.6)
    ax.errorbar(meds, y, xerr=[np.clip(meds - los, 0, None), np.clip(his - meds, 0, None)],
                fmt="none", ecolor="#555555", elinewidth=1.1, capsize=2.5, zorder=4)
    xmax = his.max() if his.size else 1.0
    for yi, md, hi in zip(y, meds, his):            # value sits past the whisker, not on it
        lab = f"{md:.0%}" if pct else fmt.format(md)
        ax.text(hi + 0.03 * xmax, yi, lab, va="center", fontsize=FS_TICK - 2,
                color="#444444")
    ax.set_yticks(y)
    if ylabels:
        labels = [f"{DISPLAY_NAMES[m]}  ({counts[m]})" if counts else DISPLAY_NAMES[m]
                  for m in order]
        ax.set_yticklabels(labels, fontsize=FS_TICK - 1)
    else:
        ax.set_yticklabels([])
    if pct:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel(xlabel, fontsize=FS_LABEL - 1)
    ax.set_title(title, fontsize=FS_LABEL, pad=10)
    ax.tick_params(labelsize=FS_TICK)
    ax.margins(x=0.22)
    ax.grid(axis="x", alpha=0.18)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


def main():
    df, ann = load()
    n_total = df["seq_id"].nunique()
    # per-method "n_annotated/n_total" so every bar/point states its support.
    counts = {m: f"{int((ann.method == m).sum())}/{n_total}" for m in FIGURE_ORDER}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6),
                             gridspec_kw={"width_ratios": [1.25, 1.0, 1.0], "wspace": 0.18})
    # All panels over ANNOTATED seqs only; the (n/84) on the left panel is the
    # coverage, so a silent-on-most method no longer reads a false 0%.
    _bars(axes[0], ann, "recovery", "Median sufficiency (% of WT)",
          "Sufficiency recovered", pct=True, counts=counts)        # labels live here only
    axes[0].axvline(0.9, color="#aaaaaa", ls="--", lw=1.0, zorder=2)
    _bars(axes[1], ann, "rec_per_bp", "Median sufficiency per bp",
          "Efficiency (per base)", counts=counts, ylabels=False)
    _tradeoff(axes[2], ann)

    # compact key inside the tradeoff panel's empty lower-right — no figure-wide
    # legend (the left panel's colored rows already name every method).
    axes[2].legend(loc="lower right", fontsize=FS_TICK - 3, frameon=False,
                   handletextpad=0.3, labelspacing=0.3, borderpad=0.2)
    fig.suptitle(f"K562 AGFT — sufficiency & efficiency   (annotated seqs only, n of {n_total})",
                 fontsize=FS_LABEL + 2, y=1.0)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))

    out = FIGDIR / "efficiency_summary.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)

    # also dump the aggregated numbers next to the plot
    rows = []
    for m in FIGURE_ORDER:
        g = ann[ann.method == m]                       # annotated seqs only (matches plot)
        rows.append({
            "method": m, "n_annotated": len(g), "n_total": n_total,
            "med_recovery": g["recovery"].median() if len(g) else np.nan,
            "med_bp": g["total_bp"].median() if len(g) else np.nan,
            "med_rec_per_bp": g["rec_per_bp"].median() if len(g) else np.nan,
            "med_rec_per_10bp": 10 * g["rec_per_bp"].median() if len(g) else np.nan,
            "med_closeness": g["closeness"].median() if len(g) else np.nan,
        })
    tab = pd.DataFrame(rows)
    tab.to_csv(C.RESULTS / "per_method_efficiency.csv", index=False)
    print("wrote", C.RESULTS / "per_method_efficiency.csv")
    print(tab.to_string(index=False))


if __name__ == "__main__":
    main()
