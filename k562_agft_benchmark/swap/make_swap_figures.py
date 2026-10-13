"""Regenerate every swap figure from results/swap_scores_by_method.npz.

Ported verbatim from swap_experiment.ipynb so the figures update from a plain
`uv run python swap/make_swap_figures.py` (no notebook). Writes to results/figures/:

    binxbin_by_method.png            bin x bin mean swap, per method
    swapbox_<method>.png             donor syntax into each context (+ shuffle ctrls)
    method_correlation.png           swap-landscape Pearson r across methods
    binxbin_log2fc.png               bin x bin mean log2FC (vs donor WT)
    swapbox_log2fc_<method>.png      log2FC boxes grouped by donor
    method_correlation_log2fc.png    log2FC-landscape correlation
    mcs_k562_binxbin_swap_heatmap.png  3x3 donor-bin x context-bin heatplot from the
                                       mcs_k562 all-by-all swap (external npz, if present)
"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
RES = HERE / "results" / "swap_scores_by_method.npz"
FIGDIR = HERE / "results" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

s = np.load(RES, allow_pickle=True)
wt, bn = s["wt"], s["bins"]
methods = [k[len("swap__"):] for k in s.files if k.startswith("swap__")]
SHORT = {"naive": "naive", "recursive_seqlets": "recursive", "tfmodisco_seqlets": "tfmodisco",
         "modiscolite_seqlets": "modiscolite", "jaspar_fimo": "jaspar", "modisco_hits_mm10": "modisco_hits"}
short = [SHORT.get(m, m) for m in methods]
ORDER = ["activating", "neutral", "repressive"]
COL = {"activating": "#d62728", "neutral": "#2ca02c", "repressive": "#1f77b4"}
IDX = {b: np.where(bn == b)[0] for b in ORDER}
DON_OK = {m: s[f"motif_bp__{m}"] > 0 for m in methods}
NDON = len(wt)


def donors(m, b):
    return np.array([i for i in IDX[b] if DON_OK[m][i]], dtype=int)


ncol = 3
nrow = int(np.ceil(len(methods) / ncol))


# ---------------------------------------------------------------- bin x bin mean swap
def binmat(m):
    sw = s[f"swap__{m}"]
    return np.array([[sw[np.ix_(donors(m, db), IDX[rb])].mean() if len(donors(m, db)) else np.nan
                      for rb in ORDER] for db in ORDER])


def fig_binxbin(valfn, vlabel, suptitle, out, namer=lambda m: m):
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.8 * nrow))
    axes = np.atleast_1d(axes).ravel()
    vmax = np.nanmax([np.abs(valfn(m)) for m in methods])
    im = None
    for ax, m in zip(axes, methods):
        A = valfn(m)
        im = ax.imshow(A, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(ORDER, rotation=0, ha="center", fontsize=8)
        ax.set_yticklabels(ORDER, fontsize=8)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, "" if np.isnan(A[i, j]) else f"{A[i, j]:+.2f}",
                        ha="center", va="center", fontsize=8)
        ax.set_title(namer(m), fontsize=9)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in axes[len(methods):]:
        ax.axis("off")
    fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.02, label=vlabel)
    fig.suptitle(suptitle, fontsize=12)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- swap boxes (abs)
def swap_box(ax, m, title):
    sw = s[f"swap__{m}"]
    motif_shuf, ctx_shuf = s[f"motif_shuffle__{m}"], s[f"context_shuffle__{m}"]
    groups = [(f"WT ({b})", [(b, wt[IDX[b]])]) for b in ORDER]
    for rb in ORDER:
        groups.append((f"{rb} ctx", [(db, sw[np.ix_(donors(m, db), IDX[rb])].ravel()) for db in ORDER]))
    groups.append(("motif shuffle", [(b, motif_shuf[IDX[b]]) for b in ORDER]))
    groups.append(("context shuffle", [(b, ctx_shuf[IDX[b]]) for b in ORDER]))
    rng = np.random.RandomState(0)
    ticks, labels, x = [], [], 0.0
    for label, items in groups:
        start = x
        for b, v in items:
            if len(v):
                bp = ax.boxplot([v], positions=[x], widths=0.26, patch_artist=True,
                                showfliers=False, medianprops=dict(color="black", lw=1.2))
                bp["boxes"][0].set(facecolor=COL[b], alpha=0.55, edgecolor="black")
                for wk in bp["whiskers"] + bp["caps"]:
                    wk.set_color("black")
                ax.scatter(x + (rng.rand(len(v)) - 0.5) * 0.18, v, s=9, color=COL[b],
                           alpha=0.6, lw=0, zorder=3)
            x += 0.32
        ticks.append((start + x - 0.32) / 2); labels.append(label); x += 0.4
    ax.axvline(ticks[len(ORDER)] - 0.55, color="0.7", lw=1, ls="--")
    ax.axvline(ticks[len(ORDER) * 2] - 0.55, color="0.7", lw=1, ls="--")
    ax.set_xticks(ticks); ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=8)
    ax.set_ylabel("K562 pred"); ax.set_title(title, fontsize=10, loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(handles=[Patch(facecolor=COL[b], alpha=0.55, edgecolor="black", label=b) for b in ORDER],
              frameon=False, fontsize=8, loc="upper right", title="donor / bin")


# ---------------------------------------------------------------- log2FC boxes
L2 = {m: s[f"swap__{m}"] - wt[:, None] for m in methods}


def binmat_fc(m):
    L = L2[m]
    return np.array([[L[np.ix_(donors(m, db), IDX[rb])].mean() if len(donors(m, db)) else np.nan
                      for rb in ORDER] for db in ORDER])


def swap_box_fc(ax, m, title):
    L = L2[m]
    ms, cs = s[f"motif_shuffle__{m}"] - wt, s[f"context_shuffle__{m}"] - wt
    groups = []
    for db in ORDER:
        dset = donors(m, db)
        groups.append((f"{db}\ndonors",
                       [(rb, L[np.ix_(dset, IDX[rb])].ravel() if len(dset) else np.array([]))
                        for rb in ORDER]))
    groups.append(("motif\nshuffle", [(rb, ms[IDX[rb]]) for rb in ORDER]))
    groups.append(("context\nshuffle", [(rb, cs[IDX[rb]]) for rb in ORDER]))
    rng = np.random.RandomState(0)
    ticks, labels, x = [], [], 0.0
    ax.axhline(0, color="0.5", lw=1, zorder=1)
    for label, items in groups:
        start = x
        for b, v in items:
            if len(v):
                bp = ax.boxplot([v], positions=[x], widths=0.26, patch_artist=True,
                                showfliers=False, medianprops=dict(color="black", lw=1.2))
                bp["boxes"][0].set(facecolor=COL[b], alpha=0.55, edgecolor="black")
                for wk in bp["whiskers"] + bp["caps"]:
                    wk.set_color("black")
                ax.scatter(x + (rng.rand(len(v)) - 0.5) * 0.18, v, s=9, color=COL[b],
                           alpha=0.6, lw=0, zorder=3)
            x += 0.32
        ticks.append((start + x - 0.32) / 2); labels.append(label); x += 0.5
    ax.axvline(ticks[len(ORDER)] - 0.6, color="0.7", lw=1, ls="--")
    ax.set_xticks(ticks); ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=9)
    ax.set_ylabel("log2FC vs donor WT  (controls vs recipient WT)", fontsize=9)
    ax.set_title(title, fontsize=10, loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(handles=[Patch(facecolor=COL[b], alpha=0.55, edgecolor="black", label=b) for b in ORDER],
              frameon=False, fontsize=8, loc="upper right", title="recipient context bin")


# ---------------------------------------------------------------- correlation
def fig_corr(landscape, suptitle, out):
    common = np.where(np.all([DON_OK[m] for m in methods], axis=0))[0]
    dd, tt = np.meshgrid(common, np.arange(len(wt)), indexing="ij")
    pm = dd != tt
    flat = np.array([landscape[m][dd[pm], tt[pm]] for m in methods])
    R = np.corrcoef(flat)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    lo = R[~np.eye(len(methods), dtype=bool)].min()
    im = ax.imshow(R, cmap="viridis", vmin=lo, vmax=1)
    ax.set_xticks(range(len(methods))); ax.set_yticks(range(len(methods)))
    ax.set_xticklabels(short, rotation=0, ha="center", fontsize=8)
    ax.set_yticklabels(short, fontsize=8)
    for i in range(len(methods)):
        for j in range(len(methods)):
            ax.text(j, i, f"{R[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if R[i, j] < (lo + 1) / 2 else "black")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    ax.set_title(suptitle, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- mcs_k562 3x3 heatplot
def fig_mcs_heatmap():
    """3x3 donor-bin x context-bin mean swap from the external mcs_k562 all-by-all
    swap (agft_swap_full.npz). Skipped if the file isn't present."""
    npz = Path("/grid/koo/home/pmantill/projects/SEAM_validate/mcs_k562/results/agft_swap_full.npz")
    if not npz.exists():
        print("  [mcs heatmap] skipped (no agft_swap_full.npz)")
        return
    d = np.load(npz, allow_pickle=True)
    high = set(int(i) for i in d["high"]); neutral = set(int(i) for i in d["neutral"])
    pd_, pc_, swap = d["pair_donor"], d["pair_ctx"], d["swap"]
    val = swap.mean(1)
    order = ["activating", "neutral", "repressing"]
    binof = lambda i: "activating" if i in high else "neutral" if i in neutral else "repressing"
    dbin = np.array([binof(int(x)) for x in pd_]); cbin = np.array([binof(int(x)) for x in pc_])
    A = np.array([[val[(dbin == db) & (cbin == rb)].mean() for rb in order] for db in order])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    vmax = np.abs(A).max()
    im = ax.imshow(A, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mean K562 pred")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(order); ax.set_yticklabels(order)
    ax.set_xlabel("recipient context bin"); ax.set_ylabel("donor motif-syntax bin")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{A[i, j]:+.2f}", ha="center", va="center", fontsize=10)
    ax.set_title("Donor-bin x recipient-bin mean swap")
    fig.tight_layout()
    fig.savefig(FIGDIR / "mcs_k562_binxbin_swap_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    print(f"{len(wt)} sequences | methods: {methods}")
    fig_binxbin(binmat, "mean K562 pred",
                "Donor-bin x recipient-bin mean swap, by annotation method",
                FIGDIR / "binxbin_by_method.png")
    print("  wrote binxbin_by_method.png")
    for m in methods:
        fig, ax = plt.subplots(figsize=(15, 4.6))
        swap_box(ax, m, f"{m}  --  donor syntax into each context "
                        f"(+ shuffle controls, {int(DON_OK[m].sum())}/{NDON} donors)")
        fig.tight_layout(); fig.savefig(FIGDIR / f"swapbox_{m}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    print("  wrote swapbox_<method>.png")
    fig_corr({m: s[f"swap__{m}"] for m in methods},
             "Swap-landscape correlation across methods", FIGDIR / "method_correlation.png")
    print("  wrote method_correlation.png")

    fig_binxbin(binmat_fc, "mean log2FC vs donor WT",
                "Donor-bin x recipient-bin mean log2FC (vs donor WT), by method",
                FIGDIR / "binxbin_log2fc.png", namer=lambda m: SHORT.get(m, m))
    print("  wrote binxbin_log2fc.png")
    for m in methods:
        fig, ax = plt.subplots(figsize=(10, 4.6))
        swap_box_fc(ax, m, f"{SHORT.get(m, m)}  --  log2FC from donor, grouped by donor "
                           f"({int(DON_OK[m].sum())}/{NDON} donors)")
        fig.tight_layout(); fig.savefig(FIGDIR / f"swapbox_log2fc_{m}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    print("  wrote swapbox_log2fc_<method>.png")
    fig_corr(L2, "log2FC swap-landscape correlation across methods",
             FIGDIR / "method_correlation_log2fc.png")
    print("  wrote method_correlation_log2fc.png")

    fig_mcs_heatmap()
    print("  wrote mcs_k562_binxbin_swap_heatmap.png")


if __name__ == "__main__":
    main()
