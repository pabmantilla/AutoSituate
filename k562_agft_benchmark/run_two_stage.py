"""Run AutoSituate's two-stage annotator (our method) on the K562 benchmark.

For every usable SEAM foreground (84 sequences):
    1. run the two-stage pipeline (stage-1 necessity boundaries + null pruning,
       stage-2a sufficiency extension, stage-2b buffered-feature search) driven
       by the K562 encoder model, using the SEAM foreground as the attribution
       map — this yields a final region set per sequence,
    2. score that region set with the SAME metrics.py KO/KI + window-matched
       null as the 6-method benchmark (same SEED / NUM_SHUFFLES, so the numbers
       are directly comparable and can be concatenated into the figures),
    3. write one tidy row per sequence for method "two_stage".

Writes ONLY new files (nothing from the 6-method run is touched):
    results/two_stage/per_sequence_two_stage.csv   same schema as per_sequence.csv
    results/two_stage/annotations_two_stage.pkl    regions + native status/recovery
    results/two_stage/run_meta.txt                 config echo

    uv run python run_two_stage.py
"""

import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
import data
import metrics as M
from two_stage_adapter import K562Adapter
from two_stage_agent import run_two_stage

# Attribution source for stage-1: the SEAM denoised foreground (default) OR the
# raw ISM map (wt_attribution.npy). Each is a SEPARATE method so they compare.
ATTR = os.environ.get("TS_ATTR", "foreground").lower()     # "foreground" | "ism"
METHOD = "two_stage" if ATTR == "foreground" else "two_stage_ism"
# distinct null-seed offset so matched-null draws never collide with the 6
# benchmark methods (run_benchmark used METHODS.index in 0..5) or each other.
TS_SEED_INDEX = 6 if ATTR == "foreground" else 7

OUTDIR = C.RESULTS / METHOD


def _attr_map(rec):
    """Stage-1 attribution map for this record (foreground or raw ISM)."""
    if ATTR == "ism":
        p = C.FOREGROUNDS / rec["id"] / "wt_attribution.npy"
        return np.load(p).astype(np.float32) if p.exists() else None
    return rec["fg"]


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    recs = data.load_dataset()
    bins, (lo, hi) = data.activity_bins(recs)
    print(f"Loaded {len(recs)} foregrounds | activity cutoffs: <= {lo:.2f} rep, "
          f">= {hi:.2f} act", flush=True)

    print(f"Attribution source = {ATTR} (method '{METHOD}')", flush=True)
    print("Loading K562 model ...", flush=True)
    from model import K562Model
    model = K562Model(device="cuda")

    rows, ann_ids, ann_regions, native, classified = [], [], [], [], {}
    for ri, rec in enumerate(recs):
        t0 = time.time()
        attr = _attr_map(rec)
        if attr is None:                         # no raw ISM map for this seq -> skip
            print(f"  [{ri+1}/{len(recs)}] {rec['id']} skipped (no {ATTR} map)", flush=True)
            continue
        adapter = K562Adapter(model, rec["left"], rec["right"])

        # --- run our method (its own ShuffleContext drives the search) -------
        res = run_two_stage(adapter, rec["var"], attr, head_index=0,
                            target_recovery=0.90, num_shuffles=C.NUM_SHUFFLES,
                            seed=C.SEED, verbose=False)
        regs = M.merge_regions(res["regions"])
        ann_ids.append(rec["id"])
        ann_regions.append(regs)
        native.append({"id": rec["id"], "status": res["status"],
                       "recovery": res["recovery"], "n_experiments": res["n_experiments"],
                       "n_motifs": len(res["motifs"]), "n_buffered": len(res["buffered"])})

        # --- classify regions for the example figures ------------------------
        # core   = stage-1 necessity span of each motif
        # flank  = the sufficiency-extension / absorbed-buffered part of a motif
        # context= standalone buffered features (invisible to KO, found by KI)
        core, flank, context = [], [], []
        for m in res["motifs"]:
            s, e = int(m["start"]), int(m["end"])
            cs, ce = int(m.get("stage1_start", s)), int(m.get("stage1_end", e))
            cs, ce = max(s, cs), min(e, ce)
            if ce > cs:
                core.append((cs, ce))
            if s < cs:
                flank.append((s, cs))
            if ce < e:
                flank.append((ce, e))
        for b in res["buffered"]:
            context.append((int(b["start"]), int(b["end"])))
        classified[rec["id"]] = {"core": core, "flank": flank, "context": context}

        # --- score the final regions through the shared benchmark metrics ----
        ctx = M.ShuffleContext(model, rec)
        total_bp = sum(e - s for s, e in regs)
        nec = M.joint_ko(ctx, regs)
        suff = M.joint_ki(ctx, regs)
        ko_null, ki_null, null_outside = M.matched_null_scores(
            ctx, regs, seed=C.SEED + ri * 131 + TS_SEED_INDEX)
        suff_z, suff_p = M.null_significance(suff, ki_null)
        nec_z, nec_p = M.null_significance(nec, ko_null)
        wsig = ctx.wt_signal if abs(ctx.wt_signal) > 1e-6 else float("nan")

        rows.append({
            "seq_id": rec["id"], "activity_bin": bins[ri],
            "true": rec["true"], "pred": rec["pred"],
            "wt_pred": ctx.wt_pred, "shuf_mean": ctx.shuf_mean,
            "wt_signal": ctx.wt_signal, "method": METHOD,
            "n_regions": len(regs), "total_bp": total_bp,
            "necessity": nec, "sufficiency": suff,
            "nec_fraction": nec / wsig, "recovery": suff / wsig,
            "null_sufficiency": float(ki_null.mean()) if len(ki_null) else 0.0,
            "null_suff_std": float(ki_null.std()) if len(ki_null) else float("nan"),
            "suff_z": suff_z, "suff_p": suff_p,
            "suff_sig": bool(suff_p < 0.05) if np.isfinite(suff_p) else False,
            "null_necessity": float(ko_null.mean()) if len(ko_null) else 0.0,
            "null_nec_std": float(ko_null.std()) if len(ko_null) else float("nan"),
            "nec_z": nec_z, "nec_p": nec_p,
            "null_draws": len(ki_null), "null_outside_frac": null_outside,
            # two-stage-native cross-check columns (ignored by the shared figures)
            "ts_status": res["status"],
            "ts_recovery_final": res["recovery"]["final"],
            "ts_n_experiments": res["n_experiments"],
        })
        print(f"  [{ri+1}/{len(recs)}] {rec['id']} ({bins[ri]}) "
              f"status={res['status']} regs={len(regs)} bp={total_bp} "
              f"recovery={rows[-1]['recovery']:.2%} ({time.time()-t0:.1f}s)", flush=True)

    df = pd.DataFrame(rows)
    out_csv = OUTDIR / f"per_sequence_{METHOD}.csv"
    df.to_csv(out_csv, index=False)
    print("wrote", out_csv, flush=True)

    pkl_path = OUTDIR / f"annotations_{METHOD}.pkl"
    with open(pkl_path, "wb") as fh:
        pickle.dump({"ids": ann_ids, "regions": {METHOD: ann_regions},
                     "native": native, "classified": classified}, fh)
    print("wrote", pkl_path, flush=True)

    with open(OUTDIR / "run_meta.txt", "w") as fh:
        fh.write(f"method={METHOD}\nn_seqs={len(recs)}\n"
                 f"target_recovery=0.90\nnum_shuffles={C.NUM_SHUFFLES}\nseed={C.SEED}\n"
                 f"ckpt={C.CKPT}\n")

    # status breakdown
    sc = df["ts_status"].value_counts().to_dict()
    print("\nstatus breakdown:", sc, flush=True)
    print(f"median recovery (metrics.py): {df['recovery'].median():.2%}  "
          f"median bp: {df['total_bp'].median():.0f}", flush=True)


if __name__ == "__main__":
    main()
