"""Systematic annotation benchmark on the K562 AlphaGenome encoder-only model.

For every usable SEAM foreground (84 sequences):
    1. annotate it with each of the 6 methods (seqlet callers + discovery are
       pooled across all sequences; naive + JASPAR-FIMO are per sequence),
    2. score each method's region set with joint knock-out (necessity) and joint
       knock-in (sufficiency) against a single shared dinuc-shuffled background,
       plus a window-matched null DISTRIBUTION (same #windows / bp, placed OUTSIDE
       the motifs) giving the significance (z, empirical p) of the annotation's
       sufficiency over random off-motif windows,
    3. write one tidy row per (sequence, method).

Outputs (results/):
    per_sequence.csv   one row per (sequence, method)
    per_method.csv     method-level medians
    run_meta.txt       config echo
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
import data
import annotate as A
import metrics as M

# Attribution source for the 6 methods: the SEAM denoised foreground (default) or
# the raw ISM map (wt_attribution.npy). ISM writes to results/ism/ so the
# foreground benchmark is never overwritten.
BENCH_ATTR = os.environ.get("BENCH_ATTR", "foreground").lower()   # "foreground" | "ism"
OUTDIR = C.RESULTS if BENCH_ATTR == "foreground" else C.RESULTS / "ism"

METHODS = [
    "recursive_seqlets",
    "tfmodisco_seqlets",
    "modiscolite_seqlets",
    "naive",
    "jaspar_fimo",
] + [f"modisco_hits_mm{mm}" for mm in C.MODISCO_MIN_METACLUSTERS]


def annotate_all(recs, tracks):
    """Return {method: list[n] of region lists}. Model-free stage."""
    out = {}

    def safe(name, fn):
        t0 = time.time()
        try:
            out[name] = fn()
            n = sum(len(r) for r in out[name])
            print(f"  [{name}] {n} regions total ({time.time()-t0:.1f}s)", flush=True)
        except Exception as e:  # a failing method must not sink the whole run
            print(f"  [{name}] FAILED: {type(e).__name__}: {e}", flush=True)
            out[name] = A._empty(len(recs))

    safe("recursive_seqlets", lambda: A.m_recursive(recs, tracks))
    safe("tfmodisco_seqlets", lambda: A.m_tfmodisco_seqlets(recs, tracks))
    safe("modiscolite_seqlets", lambda: A.m_modiscolite_seqlets(recs, tracks))

    safe("naive", lambda: A.m_naive(recs, tracks))
    safe("jaspar_fimo", lambda: A.m_jaspar_fimo(recs))

    for mm in C.MODISCO_MIN_METACLUSTERS:
        name = f"modisco_hits_mm{mm}"
        print(f"  [{name}] running TF-MoDISco discovery (min_metacluster_size={mm}) ...",
              flush=True)
        try:
            patterns = A.discover_patterns(recs, min_cluster=mm)
            print(f"  [{name}] discovered {len(patterns)} patterns", flush=True)
        except Exception as e:
            print(f"  [{name}] discovery FAILED: {type(e).__name__}: {e}", flush=True)
            patterns = []
        safe(name, lambda patterns=patterns: A.m_modisco_hits(recs, patterns))
    return out


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    recs = data.load_dataset()
    if BENCH_ATTR == "ism":                       # swap foreground -> raw ISM map
        kept = []
        for r in recs:
            p = C.FOREGROUNDS / r["id"] / "wt_attribution.npy"
            if p.exists():
                r["fg"] = np.load(p).astype(np.float32)
                kept.append(r)
            else:
                print(f"  skip {r['id']} (no wt_attribution)", flush=True)
        recs = kept
    bins, (lo, hi) = data.activity_bins(recs)
    print(f"Loaded {len(recs)} seqs | attribution={BENCH_ATTR} | out={OUTDIR} | "
          f"activity cutoffs: <= {lo:.3f} rep, >= {hi:.3f} act", flush=True)

    tracks = A.projected_tracks(recs)

    print("Annotating (model-free stage):", flush=True)
    regions = annotate_all(recs, tracks)

    print("Scoring KO / KI (loading model) ...", flush=True)
    from model import K562Model
    model = K562Model(device="cuda")

    rows = []
    for ri, rec in enumerate(recs):
        t0 = time.time()
        ctx = M.ShuffleContext(model, rec)
        for method in METHODS:
            regs = M.merge_regions(regions[method][ri])
            total_bp = sum(e - s for s, e in regs)
            nec = M.joint_ko(ctx, regs)
            suff = M.joint_ki(ctx, regs)
            ko_null, ki_null, null_outside = M.matched_null_scores(
                ctx, regs, seed=C.SEED + ri * 131 + METHODS.index(method))
            suff_z, suff_p = M.null_significance(suff, ki_null)   # the requested test
            nec_z, nec_p = M.null_significance(nec, ko_null)
            null_nec = float(ko_null.mean()) if len(ko_null) else 0.0
            null_suff = float(ki_null.mean()) if len(ki_null) else 0.0
            wsig = ctx.wt_signal if abs(ctx.wt_signal) > 1e-6 else float("nan")
            rows.append({
                "seq_id": rec["id"],
                "activity_bin": bins[ri],
                "true": rec["true"],
                "pred": rec["pred"],
                "wt_pred": ctx.wt_pred,
                "shuf_mean": ctx.shuf_mean,
                "wt_signal": ctx.wt_signal,
                "method": method,
                "n_regions": len(regs),
                "total_bp": total_bp,
                "necessity": nec,
                "sufficiency": suff,
                "nec_fraction": nec / wsig,
                "recovery": suff / wsig,
                "null_sufficiency": null_suff,
                "null_suff_std": float(ki_null.std()) if len(ki_null) else float("nan"),
                "suff_z": suff_z,
                "suff_p": suff_p,
                "suff_sig": bool(suff_p < 0.05) if np.isfinite(suff_p) else False,
                "null_necessity": null_nec,
                "null_nec_std": float(ko_null.std()) if len(ko_null) else float("nan"),
                "nec_z": nec_z,
                "nec_p": nec_p,
                "null_draws": len(ki_null),
                "null_outside_frac": null_outside,
            })
        print(f"  [{ri+1}/{len(recs)}] {rec['id']} ({bins[ri]}) "
              f"wt_signal={ctx.wt_signal:.3f} ({time.time()-t0:.1f}s)", flush=True)

    df = pd.DataFrame(rows)
    per_seq = OUTDIR / "per_sequence.csv"
    df.to_csv(per_seq, index=False)

    agg = (df.groupby("method")
             .agg(n_seq=("seq_id", "nunique"),
                  med_necessity=("necessity", "median"),
                  med_sufficiency=("sufficiency", "median"),
                  med_null_sufficiency=("null_sufficiency", "median"),
                  med_suff_z=("suff_z", "median"),
                  frac_suff_sig=("suff_sig", "mean"),
                  med_nec_fraction=("nec_fraction", "median"),
                  med_recovery=("recovery", "median"),
                  med_bp=("total_bp", "median"),
                  med_regions=("n_regions", "median"))
             .reindex(METHODS))
    per_method = OUTDIR / "per_method.csv"
    agg.to_csv(per_method)

    (OUTDIR / "run_meta.txt").write_text(
        f"n_sequences={len(recs)}\nnum_shuffles={C.NUM_SHUFFLES}\nseed={C.SEED}\n"
        f"foreground_map={C.FOREGROUND_MAP}\nckpt={C.CKPT}\n"
        f"jaspar={C.JASPAR_MEME}\nactivity_tertiles=({lo:.4f},{hi:.4f})\n")

    print(f"\nWrote {per_seq}\nWrote {per_method}\n", flush=True)
    print(agg.round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
