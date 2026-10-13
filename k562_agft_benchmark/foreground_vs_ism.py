"""Foreground vs raw-ISM attribution, per method — does seeding from the SEAM
denoised foreground help, or do the methods do just as well on the raw ISM map?

Reads the two parallel runs (foreground = the default results/; ISM = results/ism/
plus our two_stage_ism), pairs each method foreground-vs-ISM on the SHARED
sequences, and plots median sufficiency (% of WT, annotated seqs) and median bp.

    uv run python foreground_vs_ism.py
"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config as C

ORDER = ["two_stage", "naive", "recursive_seqlets", "tfmodisco_seqlets",
         "modiscolite_seqlets", "jaspar_fimo", "modisco_hits_mm10"]
DISPLAY = {
    "two_stage": "Two-stage (ours)", "naive": "Naive thresholding",
    "recursive_seqlets": "Recursive seqlets", "tfmodisco_seqlets": "TF-MoDISco (tangermeme)",
    "modiscolite_seqlets": "TF-MoDISco (modiscolite)", "jaspar_fimo": "JASPAR scan",
    "modisco_hits_mm10": "TF-MoDISco patterns",
}
FG_COLOR, ISM_COLOR = "#4c78a8", "#b22222"
FIGDIR = C.RESULTS / "figures"


def _concat(base, extra_path, rename=None):
    df = pd.read_csv(base)
    if extra_path.exists():
        ex = pd.read_csv(extra_path)
        if rename:
            ex["method"] = ex["method"].replace(rename)
        df = pd.concat([df, ex], ignore_index=True)
    return df


def load():
    fg = _concat(C.RESULTS / "per_sequence.csv",
                 C.RESULTS / "two_stage" / "per_sequence_two_stage.csv")
    ism = _concat(C.RESULTS / "ism" / "per_sequence.csv",
                  C.RESULTS / "two_stage_ism" / "per_sequence_two_stage_ism.csv",
                  rename={"two_stage_ism": "two_stage"})
    return fg, ism


def _med(df, method, col, annotated=True):
    g = df[df.method == method]
    if annotated:
        g = g[g.total_bp > 0]
    return g[col].median() if len(g) else np.nan


def main():
    fg, ism = load()
    present = [m for m in ORDER if (fg.method == m).any() and (ism.method == m).any()]
    y = np.arange(len(present))[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"wspace": 0.42})

    # Panel A: median sufficiency (% of WT), annotated seqs — foreground vs ISM
    axA = axes[0]
    for yi, m in zip(y, present):
        for col, off, color in (("foreground", +0.19, FG_COLOR), ("ism", -0.19, ISM_COLOR)):
            src = fg if col == "foreground" else ism
            v = _med(src, m, "recovery")
            axA.barh(yi + off, v, height=0.36, color=color, zorder=3)
            axA.text(v + 0.01, yi + off, f"{v:.0%}", va="center", fontsize=11, color="#333")
    axA.axvline(0.9, color="#aaa", ls="--", lw=1)
    axA.set_yticks(y); axA.set_yticklabels([DISPLAY[m] for m in present], fontsize=12)
    axA.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    axA.set_xlabel("Median sufficiency (% of WT, annotated)", fontsize=13)
    axA.set_title("Recovery: foreground vs raw ISM", fontsize=14)
    axA.margins(x=0.18)

    # Panel B: median annotation length (bp) — foreground vs ISM
    axB = axes[1]
    for yi, m in zip(y, present):
        for col, off, color in (("foreground", +0.19, FG_COLOR), ("ism", -0.19, ISM_COLOR)):
            src = fg if col == "foreground" else ism
            v = _med(src, m, "total_bp")
            axB.barh(yi + off, v, height=0.36, color=color, zorder=3)
            axB.text(v + 1.5, yi + off, f"{v:.0f}", va="center", fontsize=11, color="#333")
    axB.set_yticks(y); axB.set_yticklabels([])
    axB.set_xlabel("Median annotation length (bp, annotated)", fontsize=13)
    axB.set_title("Length: foreground vs raw ISM", fontsize=14)
    axB.margins(x=0.18)

    for ax in axes:
        ax.grid(axis="x", alpha=0.2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=FG_COLOR, label="SEAM foreground"),
                        Patch(color=ISM_COLOR, label="raw ISM")],
               loc="lower center", ncol=2, fontsize=12, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Attribution source per method — does the foreground help?  "
                 "(shared seqs, annotated-only)", fontsize=15, y=1.0)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))

    out = FIGDIR / "foreground_vs_ism.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)

    # table
    rows = []
    for m in present:
        rows.append({"method": m,
                     "fg_recovery": _med(fg, m, "recovery"), "ism_recovery": _med(ism, m, "recovery"),
                     "fg_bp": _med(fg, m, "total_bp"), "ism_bp": _med(ism, m, "total_bp")})
    tab = pd.DataFrame(rows)
    tab.to_csv(C.RESULTS / "foreground_vs_ism.csv", index=False)
    print(tab.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
