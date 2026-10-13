"""Generate benchmark assessment figures as PNGs — no notebook needed.

    uv run python visualize.py example                 # auto-pick a rich activating seq
    uv run python visualize.py example --seq peak109    # pin a sequence
    uv run python visualize.py example --refresh        # recompute the annotation cache

Annotations (the model-free stage, incl. the slow TF-MoDISco discovery) are cached
to results/annotations.pkl, so re-running to restyle a figure is instant. Edit the
STYLE block below and re-run; just open the PNG.
"""

import argparse
import csv
import pickle
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
import data
import annotate as A
import metrics as M
from run_benchmark import annotate_all, METHODS

sys.path.insert(0, str(C.EI_DIR))
from fast_logo import fast_logo  # noqa: E402

# ----------------------------- STYLE (edit me) -----------------------------
# Methods ordered least -> most computationally heavy (top -> bottom in the figure).
FIGURE_ORDER = [
    "two_stage",             # AutoSituate two-stage annotator on the foreground (OURS)
    "two_stage_ism",         # same, but seeded from the raw ISM map (OURS)
    "naive",                 # adaptive threshold on the CWM  (trivial)
    "recursive_seqlets",     # tangermeme recursive seqlets
    "tfmodisco_seqlets",     # tangermeme TF-MoDISco-style seqlets
    "modiscolite_seqlets",   # real TF-MoDISco (modiscolite) seqlets
    "jaspar_fimo",           # FIMO scan of the full JASPAR database
    "modisco_hits_mm10",     # de novo TF-MoDISco discovery (min 10) -> hit-calling
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
    "two_stage":           "*",
    "two_stage_ism":       "X",
    "naive":               "o",
    "recursive_seqlets":   "s",
    "tfmodisco_seqlets":   "^",
    "modiscolite_seqlets": "D",
    "jaspar_fimo":         "v",
    "modisco_hits_mm10":   "P",
}

# Each of our runs writes to its own dir so the 6-method run (and each other) is
# never overwritten; loaders concatenate whichever variants exist.
TS_RUNS = ["two_stage", "two_stage_ism"]


def _ts_csv(method):
    return C.RESULTS / method / f"per_sequence_{method}.csv"


def read_per_sequence():
    """per_sequence.csv for the 6 benchmark methods, plus our two-stage variants
    (foreground / raw ISM) appended if their separate runs have produced them."""
    import pandas as pd
    df = pd.read_csv(C.RESULTS / "per_sequence.csv")
    for m in TS_RUNS:
        p = _ts_csv(m)
        if p.exists():
            df = pd.concat([df, pd.read_csv(p)], ignore_index=True)
    return df
FIG_W = 15
LOGO_H = 1.7          # height ratio of each logo panel
TRACK_H = 2.6         # height ratio of the annotation-track panel
DPI = 150
# text sizes
FS_TITLE = 16
FS_LABEL = 15
FS_TICK = 13
FS_ANNOT = 11
# KO / KI metric bar colors
KO_COLOR = "#c44e52"   # necessity (knock-out)
KI_COLOR = "#4c72b0"   # sufficiency (knock-in)
# ---------------------------------------------------------------------------

CACHE = C.RESULTS / "annotations.pkl"
FIGDIR = C.RESULTS / "figures"


# (method, seq_id) -> {"core":[...], "flank":[...], "context":[...]} for our runs,
# so the example figures can denote the three region types instead of one flat bar.
TS_CLASS = {}


def _attach_two_stage(regions, ids):
    """Add our two-stage variants' regions from their separate runs (if present),
    aligned to the dataset order. In-memory only — annotations.pkl (the 6-method
    cache) is never rewritten. Also loads each run's core/flank/context split."""
    regions = dict(regions)
    TS_CLASS.clear()
    for m in TS_RUNS:
        pkl = C.RESULTS / m / f"annotations_{m}.pkl"
        if not pkl.exists():
            continue
        d = pickle.load(open(pkl, "rb"))
        by_id = dict(zip(d["ids"], d["regions"][m]))
        regions[m] = [by_id.get(sid, []) for sid in ids]
        for sid, cls in d.get("classified", {}).items():
            TS_CLASS[(m, sid)] = cls
    return regions


# region-type styling for our method's track row
TS_SUB_COLORS = {"core": "#111111", "flank": "#9e9e9e", "context": "#1aa3a3"}
TS_SUB_LABELS = {"core": "core motif (necessity)", "flank": "flank (sufficiency ext.)",
                 "context": "context (buffered)"}


def load_annotations(refresh=False):
    recs = data.load_dataset()
    bins, _ = data.activity_bins(recs)
    ids = [r["id"] for r in recs]
    regions = None
    if CACHE.exists() and not refresh:
        cached = pickle.load(open(CACHE, "rb"))
        if cached.get("ids") == ids:
            regions = cached["regions"]
        else:
            print("cache stale (sequence set changed) -> recomputing", flush=True)
    if regions is None:
        tracks = A.projected_tracks(recs)
        regions = annotate_all(recs, tracks)
        C.RESULTS.mkdir(parents=True, exist_ok=True)
        pickle.dump({"ids": ids, "regions": regions}, open(CACHE, "wb"))
        print(f"cached annotations -> {CACHE}", flush=True)
    regions = _attach_two_stage(regions, ids)
    return recs, bins, regions


def load_metrics():
    """{(seq_id, method): {necessity, sufficiency, nec_fraction, recovery, ...}}
    from the benchmark's per_sequence.csv (empty dict if the job hasn't run)."""
    path = C.RESULTS / "per_sequence.csv"
    if not path.exists():
        return {}
    out = {}
    for p in [path] + [_ts_csv(m) for m in TS_RUNS]:
        if not p.exists():
            continue
        with open(p) as f:
            for r in csv.DictReader(f):
                out[(r["seq_id"], r["method"])] = r
    return out


def _pick_example(recs, bins, regions):
    cand = [i for i, b in enumerate(bins) if b == "activating"] or list(range(len(recs)))
    i = max(cand, key=lambda i: sum(bool(regions[m][i]) for m in METHODS))
    return recs[i]["id"]


def _example_setup(recs, bins, regions, seq_id):
    """Resolve seq_id -> (i, rec, logos, methods). Shared by the example figures."""
    id2i = {r["id"]: i for i, r in enumerate(recs)}
    if seq_id is None:
        seq_id = _pick_example(recs, bins, regions)
    if seq_id not in id2i:
        raise SystemExit(f"{seq_id} not in dataset. Options e.g.: {list(id2i)[:5]} ...")
    i = id2i[seq_id]
    rec = recs[i]
    wt_path = C.FOREGROUNDS / seq_id / "wt_attribution.npy"
    wt = np.load(wt_path).astype(float) if wt_path.exists() else None
    logos = [("WT attribution", wt)] if wt is not None else []
    logos.append(("foreground", rec["fg"].astype(float)))
    methods = [m for m in FIGURE_ORDER if m in regions]
    return seq_id, i, rec, logos, methods


def _draw_logos_track(fig, gs, logos, methods, regions, i, seq_id=None):
    """Left column: attribution logos (aligned) + the per-method annotation track.
    Our method's row is split into core / flank / context (TS_CLASS); every other
    method stays a single bar. Returns (track_ax, shared_x_ax)."""
    ax0 = None
    for row, (label, vals) in enumerate(logos):
        ax = fig.add_subplot(gs[row, 0], sharex=ax0) if ax0 else fig.add_subplot(gs[row, 0])
        ax0 = ax0 or ax
        fast_logo(vals, ax)
        ax.set_ylabel(label, fontsize=FS_LABEL)
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", labelsize=FS_TICK)
        ax.tick_params(axis="x", bottom=False, labelbottom=False)

    ax = fig.add_subplot(gs[-1, 0], sharex=ax0)
    ts_drawn = False
    for yi, m in enumerate(methods):
        ts_cls = TS_CLASS.get((m, seq_id)) if seq_id is not None else None
        if ts_cls:                               # our runs: split core/flank/context
            # context (teal) under flank (gray) under core (black); non-overlapping
            for kind in ("context", "flank", "core"):
                for s, e in M.merge_regions(ts_cls.get(kind, [])):
                    ax.broken_barh([(s, e - s)], (yi - 0.4, 0.8),
                                   facecolors=TS_SUB_COLORS[kind], edgecolors="none")
            ts_drawn = True
        else:
            regs = M.merge_regions(regions[m][i])
            if regs:
                ax.broken_barh([(s, e - s) for s, e in regs], (yi - 0.4, 0.8),
                               facecolors=COLORS.get(m, "#444444"), edgecolors="none")
    if ts_drawn:
        from matplotlib.patches import Patch
        # figure-level, parked in the empty top-right (logos only use the left column)
        fig.legend(handles=[Patch(color=TS_SUB_COLORS[k], label=TS_SUB_LABELS[k])
                            for k in ("core", "flank", "context")],
                   title="Two-stage (ours)", fontsize=FS_ANNOT - 1,
                   title_fontsize=FS_ANNOT - 1, loc="upper right",
                   bbox_to_anchor=(0.995, 0.97), framealpha=0.0,
                   handlelength=1.3, borderpad=0.3)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([DISPLAY_NAMES.get(m, m) for m in methods], fontsize=FS_TICK)
    ax.set_ylim(-0.6, len(methods) - 0.4)
    ax.invert_yaxis()
    ax.set_xlim(0, C.L_VAR)
    ax.set_xlabel("position (bp)", fontsize=FS_LABEL)
    ax.tick_params(axis="x", labelsize=FS_TICK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", alpha=0.2)
    return ax, ax0


def fig_example(recs, bins, regions, seq_id=None, out=None):
    seq_id, i, rec, logos, methods = _example_setup(recs, bins, regions, seq_id)
    metrics = load_metrics()

    ROW_H = 0.66                                   # vertical space per method row
    track_h = ROW_H * len(methods)
    heights = [LOGO_H] * len(logos) + [track_h]
    fig = plt.figure(figsize=(FIG_W, LOGO_H * len(logos) + track_h + 2.6))
    gs = fig.add_gridspec(len(heights), 2, height_ratios=heights,
                          width_ratios=[3.0, 1.15], hspace=0.6, wspace=0.28)

    # small sequence tag top-left (not a title)
    fig.text(0.01, 0.99, f"{seq_id}   ·   {bins[i]}   ·   true={rec['true']:.2f}",
             fontsize=FS_LABEL, ha="left", va="top")

    ax, ax0 = _draw_logos_track(fig, gs, logos, methods, regions, i, seq_id=seq_id)

    # --- KO / KI test results (right) ---
    # Necessity (knock-out) and Sufficiency (knock-in), each as a fraction of the
    # WT signal. Bars share the track's y so each pair sits beside its method row.
    axm = fig.add_subplot(gs[-1, 1], sharey=ax)
    rows = {m: metrics.get((seq_id, m)) for m in methods}
    vals = [float(r[k]) for r in rows.values() if r is not None
            and float(r["total_bp"]) > 0 for k in ("nec_fraction", "recovery")]
    if vals:
        # Data-driven window, but cap rare outliers (e.g. nec=-13, +4) so the
        # ordinary 0..1 range stays readable; out-of-range bars are clamped and
        # flagged with an arrow next to their true value.
        lo = max(-1.0, min(0.0, min(vals)))
        hi = min(2.0, max(1.1, max(vals)))
        span = hi - lo
        pad = span * 0.015
        for yi, m in enumerate(methods):
            row = rows[m]
            if row is None or float(row["total_bp"]) == 0:    # no regions -> no test
                axm.text(lo + span * 0.03, yi, "no annotation", va="center",
                         ha="left", fontsize=FS_ANNOT - 2, color="#bbbbbb",
                         style="italic")
                continue
            for val, color, off in ((float(row["nec_fraction"]), KO_COLOR, -0.2),
                                    (float(row["recovery"]),     KI_COLOR, +0.2)):
                drawn = min(max(val, lo), hi)                  # clamp into view
                axm.barh(yi + off, drawn, height=0.36, color=color, zorder=3)
                arrow = "▸" if val > hi else ("◂" if val < lo else "")
                if drawn >= 0:
                    axm.text(drawn + pad, yi + off, f"{val:.2f}{arrow}",
                             va="center", ha="left", fontsize=FS_ANNOT - 1,
                             color="#222222")
                else:
                    axm.text(drawn - pad, yi + off, f"{arrow}{val:.2f}",
                             va="center", ha="right", fontsize=FS_ANNOT - 1,
                             color="#222222")
        if lo < 0:
            axm.axvline(0.0, color="#cccccc", lw=0.8, zorder=1)
        axm.axvline(1.0, color="#888888", lw=1, ls="--", zorder=2)  # full WT
        axm.set_xlim(lo - span * 0.02, hi + span * 0.22)   # headroom for labels
        axm.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        axm.set_xlabel("Fraction of WT", fontsize=FS_LABEL)
        axm.tick_params(axis="x", labelsize=FS_TICK)
        from matplotlib.patches import Patch
        axm.legend(handles=[Patch(color=KO_COLOR, label="Necessity"),
                            Patch(color=KI_COLOR, label="Sufficiency")],
                   fontsize=FS_ANNOT, loc="lower center", bbox_to_anchor=(0.5, 1.01),
                   ncol=2, framealpha=0.0, handlelength=1.2, columnspacing=1.2)
    else:
        axm.text(0.5, 0.5, "run benchmark\nfor KO/KI", ha="center", va="center",
                 fontsize=FS_LABEL, color="#999999", transform=axm.transAxes)
        axm.set_xticks([])
    axm.tick_params(axis="y", labelleft=False)
    for side in ("top", "right"):
        axm.spines[side].set_visible(False)
    axm.grid(axis="x", alpha=0.2)

    if out is None:
        bin_dir = FIGDIR / "examples" / bins[i]
        bin_dir.mkdir(parents=True, exist_ok=True)
        out = bin_dir / f"example_{seq_id}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


SIG_EXAMPLE_DIR = FIGDIR / "significance_examples"


def fig_example_significance(recs, bins, regions, seq_id=None, out=None):
    """Per-sequence view: each method's annotation track (left) beside THAT method's
    significance on this sequence (right) -- necessity (KO) and sufficiency (KI) as
    a marker, against the window-matched null band (null mean +/- std, both as % of
    WT), with the actual z and two-sided empirical p printed per method. Saved under
    results/figures/significance_examples/<bin>/."""
    seq_id, i, rec, logos, methods = _example_setup(recs, bins, regions, seq_id)
    metrics = load_metrics()

    ROW_H = 0.78
    track_h = ROW_H * len(methods)
    heights = [LOGO_H] * len(logos) + [track_h]
    fig = plt.figure(figsize=(FIG_W, LOGO_H * len(logos) + track_h + 2.8))
    gs = fig.add_gridspec(len(heights), 2, height_ratios=heights,
                          width_ratios=[2.3, 1.7], hspace=0.6, wspace=0.26)
    fig.text(0.01, 0.99, f"{seq_id}   ·   {bins[i]}   ·   true={rec['true']:.2f}   "
             f"·  significance vs window-matched null",
             fontsize=FS_LABEL, ha="left", va="top")

    ax, ax0 = _draw_logos_track(fig, gs, logos, methods, regions, i, seq_id=seq_id)

    # --- significance panel (right): real effect vs null band, as % of WT ---
    axm = fig.add_subplot(gs[-1, 1], sharey=ax)
    rows = {m: metrics.get((seq_id, m)) for m in methods}

    def _f(x):                                   # empty cells (NaN) -> nan, never crash
        try:
            return float(x)
        except (ValueError, TypeError):
            return float("nan")

    def fracs(r):
        w = _f(r["wt_signal"])
        if not np.isfinite(w) or abs(w) < 1e-6:
            return None
        nd = int(_f(r["null_draws"])) if np.isfinite(_f(r["null_draws"])) else 0
        return dict(
            nec=_f(r["nec_fraction"]), suff=_f(r["recovery"]),
            nnull=_f(r["null_necessity"]) / w, nnull_sd=_f(r["null_nec_std"]) / abs(w),
            snull=_f(r["null_sufficiency"]) / w, snull_sd=_f(r["null_suff_std"]) / abs(w),
            nz=_f(r["nec_z"]), np_=_f(r["nec_p"]),
            sz=_f(r["suff_z"]), sp=_f(r["suff_p"]), nd=nd,
            outside=_f(r.get("null_outside_frac", "nan")))

    data = {m: (fracs(rows[m]) if rows[m] and float(rows[m]["total_bp"]) > 0 else None)
            for m in methods}
    xs = [v for d in data.values() if d for v in
          (d["nec"], d["suff"], d["nnull"], d["snull"]) if np.isfinite(v)]
    lo = max(-1.0, min(0.0, min(xs))) if xs else 0.0
    hi = min(2.5, max(1.1, max(xs))) if xs else 1.1
    span = hi - lo

    for yi, m in enumerate(methods):
        d = data[m]
        if d is None:
            axm.text(lo + span * 0.03, yi, "no annotation", va="center", ha="left",
                     fontsize=FS_ANNOT - 2, color="#bbbbbb", style="italic")
            continue
        # tan band = null had to be placed ANYWHERE (annotation too big to fit outside)
        fallback = np.isfinite(d["outside"]) and d["outside"] < 0.5
        band_c = "#d9a066" if fallback else "#9a9a9a"
        for real, nullmean, nullsd, color, off, z, p, tag in (
                (d["nec"], d["nnull"], d["nnull_sd"], KO_COLOR, -0.22, d["nz"], d["np_"], "KO"),
                (d["suff"], d["snull"], d["snull_sd"], KI_COLOR, +0.22, d["sz"], d["sp"], "KI")):
            if d["nd"] > 0 and np.isfinite(nullmean):          # null band (mean +/- std)
                axm.plot([nullmean - nullsd, nullmean + nullsd], [yi + off, yi + off],
                         color=band_c, lw=7, alpha=0.5, solid_capstyle="round", zorder=2)
                axm.scatter([nullmean], [yi + off], marker="|", color="#5a5a5a", s=60, zorder=3)
            axm.scatter([min(max(real, lo), hi)], [yi + off], color=color, s=85,
                        edgecolor="white", lw=1.0, zorder=5)
            sig = "*" if (np.isfinite(p) and p < 0.05) else ""
            ptxt = f"p={p:.3f}" if np.isfinite(p) else "p=n/a"
            ztxt = f"z={z:+.1f}" if np.isfinite(z) else "z=n/a"
            axm.text(hi + span * 0.03, yi + off, f"{tag} {ztxt} {ptxt}{sig}",
                     va="center", ha="left", fontsize=FS_ANNOT - 2, color=color)

    axm.axvline(0.0, color="#cccccc", lw=0.8, zorder=1)
    axm.axvline(1.0, color="#888888", lw=1, ls="--", zorder=1)       # 100% = WT
    axm.set_xlim(lo - span * 0.02, hi + span * 0.42)                 # headroom for z/p text
    axm.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    axm.set_xlabel("Effect (% of WT) vs null band", fontsize=FS_LABEL)
    axm.tick_params(axis="x", labelsize=FS_TICK)
    axm.tick_params(axis="y", labelleft=False)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    axm.legend(handles=[Line2D([], [], marker="o", color=KO_COLOR, ls="", label="Necessity (KO)"),
                        Line2D([], [], marker="o", color=KI_COLOR, ls="", label="Sufficiency (KI)"),
                        Line2D([], [], color="#9a9a9a", lw=7, alpha=0.5, label="null (outside motifs)"),
                        Line2D([], [], color="#d9a066", lw=7, alpha=0.5, label="null (anywhere: too big)"),
                        Line2D([], [], marker="*", color="#222222", ls="", label="p < 0.05")],
               fontsize=FS_ANNOT - 2, loc="lower center", bbox_to_anchor=(0.5, 1.01),
               ncol=3, framealpha=0.0, handlelength=1.4, columnspacing=1.0)
    for side in ("top", "right"):
        axm.spines[side].set_visible(False)
    axm.grid(axis="x", alpha=0.2)

    if out is None:
        bin_dir = SIG_EXAMPLE_DIR / bins[i]
        bin_dir.mkdir(parents=True, exist_ok=True)
        out = bin_dir / f"example_{seq_id}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def _bp_vs_capture(ax, df, order, col, ylabel, target=None):
    """Annotation length (x) vs Sufficiency/Necessity (y), fly-style scatter.

    Faint point = one sequence; big marker = that method's median, with IQR
    bars on both axes. y is % of WT, so 100% = reproduces WT.
    """
    for m in order:
        g = df[df.method == m]
        ax.scatter(g["total_bp"], g[col], s=22, alpha=0.28, lw=0,
                   color=COLORS[m], marker=MARKERS[m], zorder=3)
    for m in order:
        g = df[df.method == m]
        mx, my = g["total_bp"].median(), g[col].median()
        ax.plot([g["total_bp"].quantile(.25), g["total_bp"].quantile(.75)],
                [my, my], color=COLORS[m], lw=1.8, alpha=0.8, zorder=5)
        ax.plot([mx, mx], [g[col].quantile(.25), g[col].quantile(.75)],
                color=COLORS[m], lw=1.8, alpha=0.8, zorder=5)
        ax.scatter([mx], [my], s=230, color=COLORS[m], marker=MARKERS[m],
                   edgecolor="white", linewidth=2.0, zorder=6,
                   label=DISPLAY_NAMES[m])
    if target is not None:
        ax.axhline(target, color="#888888", ls="--", lw=1.2, zorder=2)
    ax.axhline(1.0, color="#bbbbbb", ls=":", lw=1.0, zorder=2)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("Annotation length (bp)", fontsize=FS_LABEL)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.margins(x=0.08, y=0.10)
    ax.grid(alpha=0.2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _summary_bars(ax, df, order, col, xlabel, target=None):
    """Median across sequences as a horizontal bar, IQR as a whisker.

    Same method order as the scatter above it. x is % of WT.
    """
    y = np.arange(len(order))[::-1]
    med = np.array([df.loc[df.method == m, col].median() for m in order])
    q25 = np.array([df.loc[df.method == m, col].quantile(.25) for m in order])
    q75 = np.array([df.loc[df.method == m, col].quantile(.75) for m in order])

    ax.barh(y, med, height=0.62, lw=0, zorder=3, color=[COLORS[m] for m in order])
    for yi, lo, hi in zip(y, q25, q75):
        ax.plot([lo, hi], [yi, yi], color="#333333", lw=1.3, alpha=0.55,
                zorder=5, solid_capstyle="butt")
    if target is not None:
        ax.axvline(target, color="#888888", ls="--", lw=1.2, zorder=4)
    ax.axvline(1.0, color="#bbbbbb", ls=":", lw=1.0, zorder=2)

    span = max(np.nanmax(q75), np.nanmax(med), 0.1)
    for yi, v, hi in zip(y, med, q75):
        ax.text(max(v, hi) + span * 0.03, yi, f"{v:.0%}", va="center",
                fontsize=FS_ANNOT, color="#222222")
    ax.set_yticks(y)
    ax.set_yticklabels([DISPLAY_NAMES[m] for m in order], fontsize=FS_TICK)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel(xlabel, fontsize=FS_LABEL)
    ax.tick_params(axis="x", labelsize=FS_TICK)
    ax.grid(axis="x", alpha=0.2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def fig_summary(activity_bin, out=None, target=0.90):
    """One activity bin per figure. Top row: Annotation length vs Sufficiency
    (knock-in) and Necessity (knock-out) scatter. Bottom row: the same two as
    median+IQR bars per method."""
    import pandas as pd
    df = read_per_sequence()
    df = df[df["activity_bin"] == activity_bin].copy()
    present = [m for m in FIGURE_ORDER if m in set(df["method"])]
    # one method order everywhere, set by median Sufficiency
    order = sorted(present, key=lambda m: df.loc[df.method == m, "recovery"].median(),
                   reverse=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    _bp_vs_capture(axes[0, 0], df, order, "recovery",
                   "Sufficiency (% of WT)", target=target)
    axes[0, 0].set_title("Sufficiency", fontsize=FS_LABEL + 1, loc="left")
    _bp_vs_capture(axes[0, 1], df, order, "nec_fraction",
                   "Necessity (% of WT)", target=None)
    axes[0, 1].set_title("Necessity", fontsize=FS_LABEL + 1, loc="left")
    _summary_bars(axes[1, 0], df, order, "recovery",
                  "Sufficiency (% of WT)", target=target)
    _summary_bars(axes[1, 1], df, order, "nec_fraction",
                  "Necessity (% of WT)", target=None)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(3, len(labels)),
               fontsize=FS_ANNOT, frameon=False, bbox_to_anchor=(0.5, -0.01))
    n = df["seq_id"].nunique()
    fig.suptitle(f"{activity_bin.capitalize()} sequences (n={n})",
                 fontsize=FS_TITLE, x=0.09, ha="left")
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = out or (FIGDIR / f"summary_{activity_bin}.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


ANNOT_COLOR = "#2a6f97"     # the method's annotation (one color everywhere)
NULL_COLOR = "#b0b0b0"      # window-matched null (one color everywhere)


def fig_significance(out=None):
    """Necessity AND sufficiency vs the window-matched null, per method.

    Top row  (effect size, as % of WT): median annotation effect vs the median
      matched-null effect, with IQR whiskers. Sufficiency/necessity are divided by
      wt_signal = WT - shuffled_mean, so 100% = reproduces the WT activity change.
    Bottom row (significance): the per-sequence z of the real effect over the null
      distribution (box = IQR, whisker = 5-95%, points = sequences). The actual
      median z and median two-sided empirical p are printed per method -- no cutoff.
    Left column = Necessity (KO), right column = Sufficiency (KI). Same method order
      and the same two colors (annotation / null) throughout, so the legend matches."""
    import pandas as pd
    df = read_per_sequence()
    df = df[df["total_bp"] > 0].copy()                        # annotated sequences only
    # effects relative to WT activity change (wt_signal = WT - shuffled_mean)
    df["null_nec_fraction"] = df["null_necessity"] / df["wt_signal"]
    df["null_recovery"] = df["null_sufficiency"] / df["wt_signal"]
    order = [m for m in FIGURE_ORDER if m in set(df["method"])]
    y = np.arange(len(order))[::-1]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11),
                             gridspec_kw={"height_ratios": [1.1, 1.0]})
    SPECS = [  # column -> (real %WT col, null %WT col, z col, p col, title)
        ("nec_fraction", "null_nec_fraction", "nec_z", "nec_p", "Necessity (knock-out)"),
        ("recovery",     "null_recovery",     "suff_z", "suff_p", "Sufficiency (knock-in)"),
    ]

    for c, (real_c, null_c, zc, pc, title) in enumerate(SPECS):
        # --- top: effect as % of WT, annotation vs matched null ---
        axT = axes[0, c]
        for yi, m in zip(y, order):
            g = df[df.method == m]
            for col, off, color in ((real_c, +0.19, ANNOT_COLOR), (null_c, -0.19, NULL_COLOR)):
                med = g[col].median()
                axT.barh(yi + off, med, height=0.36, color=color, zorder=3)
                axT.plot([g[col].quantile(.25), g[col].quantile(.75)], [yi + off, yi + off],
                         color="#333333", lw=1.2, alpha=0.55, zorder=5, solid_capstyle="butt")
        axT.axvline(0, color="#888888", lw=1, zorder=2)
        axT.axvline(1.0, color="#bbbbbb", ls=":", lw=1.0, zorder=2)   # 100% = WT
        axT.set_yticks(y)
        axT.set_yticklabels([DISPLAY_NAMES.get(m, m) for m in order] if c == 0 else [],
                            fontsize=FS_TICK)
        axT.set_ylim(-0.6, len(order) - 0.4)
        axT.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        axT.set_xlabel("Effect (% of WT activity change)", fontsize=FS_LABEL)
        axT.set_title(title, fontsize=FS_LABEL + 2, loc="left")
        axT.tick_params(axis="x", labelsize=FS_TICK)
        axT.grid(axis="x", alpha=0.2)
        for side in ("top", "right"):
            axT.spines[side].set_visible(False)

        # --- bottom: per-sequence z over the null, with actual median z & p ---
        axB = axes[1, c]
        zdata = [df.loc[df.method == m, zc].replace([np.inf, -np.inf], np.nan).dropna().values
                 for m in order]
        bp = axB.boxplot(zdata, positions=y, vert=False, widths=0.6, showfliers=False,
                         patch_artist=True, whis=(5, 95),
                         medianprops=dict(color="black", lw=1.4))
        for patch, m in zip(bp["boxes"], order):
            patch.set(facecolor=COLORS[m], alpha=0.5, edgecolor="black")
        for wk in bp["whiskers"] + bp["caps"]:
            wk.set_color("0.4")
        rng = np.random.RandomState(0)
        for yi, zv in zip(y, zdata):
            axB.scatter(zv, yi + (rng.rand(len(zv)) - 0.5) * 0.3, s=10, color="#333333",
                        alpha=0.4, lw=0, zorder=4)
        for z in (-1.96, 1.96):                                        # |z|~1.96 <-> p~0.05
            axB.axvline(z, color="#c44e52", ls="--", lw=1.0, zorder=2)
        axB.axvline(0, color="#888888", lw=1, zorder=2)
        xmax = axB.get_xlim()[1]
        for yi, m in zip(y, order):
            g = df[df.method == m]
            mz, mp = g[zc].median(), g[pc].median()
            txt = f"z̃={mz:+.1f}  p̃={mp:.3f}" if np.isfinite(mz) else "n/a"
            axB.text(xmax, yi + 0.34, txt, va="center", ha="right", fontsize=FS_ANNOT - 1,
                     color="#222222")
        axB.set_yticks(y)
        axB.set_yticklabels([DISPLAY_NAMES.get(m, m) for m in order] if c == 0 else [],
                            fontsize=FS_TICK)
        axB.set_ylim(-0.6, len(order) - 0.4)
        axB.set_xlabel("z of effect vs matched null  (per sequence)", fontsize=FS_LABEL)
        axB.tick_params(axis="x", labelsize=FS_TICK)
        axB.grid(axis="x", alpha=0.2)
        for side in ("top", "right"):
            axB.spines[side].set_visible(False)

    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=ANNOT_COLOR, label="annotation"),
                        Patch(color=NULL_COLOR, label="window-matched null"),
                        plt.Line2D([], [], color="#c44e52", ls="--", label="|z|=1.96 (p≈0.05)")],
               loc="lower center", ncol=3, fontsize=FS_ANNOT, frameon=False,
               bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Annotation significance over the window-matched null  "
                 "(effect as % of WT, with actual z and p)", fontsize=FS_TITLE, x=0.09, ha="left")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = out or (FIGDIR / "summary_significance.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("figure", nargs="?", default="example",
                    choices=["example", "sig_example", "summary", "significance", "all"],
                    help="which figure(s) to generate ('all' = every example "
                         "(annotation + per-method significance) + summaries + significance)")
    ap.add_argument("--seq", default=None, help="sequence id (default: auto-pick)")
    ap.add_argument("--out", default=None, help="output PNG path")
    ap.add_argument("--refresh", action="store_true", help="recompute annotation cache")
    args = ap.parse_args()

    def do_summaries():
        import pandas as pd
        present = read_per_sequence()["activity_bin"].unique()
        for b in ["activating", "neutral", "repressive"]:
            if b in present:
                print("wrote", fig_summary(b))
        print("wrote", fig_significance())

    if args.figure == "summary":
        do_summaries()
        return
    if args.figure == "significance":
        print("wrote", fig_significance())
        return

    recs, bins, regions = load_annotations(refresh=args.refresh)
    if args.figure == "example":
        print("wrote", fig_example(recs, bins, regions, seq_id=args.seq, out=args.out))
    elif args.figure == "sig_example":
        print("wrote", fig_example_significance(recs, bins, regions, seq_id=args.seq, out=args.out))
    elif args.figure == "all":
        for rec in recs:                                      # annotation + significance per seq
            print("wrote", fig_example(recs, bins, regions, seq_id=rec["id"]))
            print("wrote", fig_example_significance(recs, bins, regions, seq_id=rec["id"]))
        do_summaries()


if __name__ == "__main__":
    main()
