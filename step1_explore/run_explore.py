"""Stage-1 exploration: run our two-stage annotator on the RAW ISM maps, varying
ONLY the initial seeding step, and compare.

Four Stage-1 variants (see seeders.py): naive_pos, naive_abs, adaptive_nec,
seqlets. For each variant we monkeypatch the pipeline's Stage 1 and run the
otherwise-unchanged two_stage_agent.run_two_stage on every sequence that has a
raw ISM map (wt_attribution.npy), on the K562 AlphaGenome-encoder model.

Scoring is the pipeline's OWN ShuffleContext (same SEED / NUM_SHUFFLES as the
benchmark), so `final` recovery here is directly comparable to results/two_stage.
We deliberately skip the heavy window-matched null (not needed to compare seeds).

Outputs (results/):
    <variant>/per_sequence.csv    one row per sequence, for that variant
    <variant>/annotations.pkl     regions + core/flank/context split + native status
    seed_comparison.csv           tidy long table: every (variant, sequence) with
                                  seed count/bp, recovery at each stage, final bp,
                                  core vs context bp, status, #experiments
    explore_summary.csv           one row per variant: the medians that answer
                                  "does a better Stage 1 localise more and lean
                                  less on the buffered search?"

Env knobs (optional):
    VARIANTS=naive_pos,seqlets    run a subset (default: all four)
    LIMIT=5                       first N sequences only (quick smoke test)

    uv run python run_explore.py
"""

import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the benchmark's harness verbatim (config, data, model, adapter, agent).
BENCH = Path("/grid/koo/home/pmantill/projects/AutoSituate/k562_agft_benchmark")
sys.path.insert(0, str(BENCH))

import config as C          # noqa: E402
import data                 # noqa: E402
import metrics as M         # noqa: E402  (merge_regions only)
import two_stage_agent as TSA  # noqa: E402
from two_stage_adapter import K562Adapter  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from seeders import SEEDERS          # noqa: E402
from stage1_variants import make_stage1  # noqa: E402

OUT = HERE / "results"
TARGET = 0.90


def attr_ism(rec):
    """Raw ISM attribution map (wt_attribution.npy) for this record, or None."""
    p = C.FOREGROUNDS / rec["id"] / "wt_attribution.npy"
    return np.load(p).astype(np.float32) if p.exists() else None


def classify(res):
    """core = stage-1 necessity span, flank = sufficiency extension, context =
    standalone buffered features (same split as run_two_stage.py's figures)."""
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
    return {"core": core, "flank": flank, "context": context}


def _bp(regs):
    return sum(e - s for s, e in M.merge_regions(regs))


def main():
    variants = (os.environ["VARIANTS"].split(",")
                if os.environ.get("VARIANTS") else list(SEEDERS))
    limit = int(os.environ.get("LIMIT", "0")) or None

    recs = data.load_dataset()
    bins, (lo, hi) = data.activity_bins(recs)
    usable = [(ri, rec) for ri, rec in enumerate(recs) if attr_ism(rec) is not None]
    if limit:
        usable = usable[:limit]
    print(f"Variants: {variants}", flush=True)
    print(f"{len(usable)}/{len(recs)} sequences have a raw ISM map | "
          f"activity cutoffs <= {lo:.2f} rep, >= {hi:.2f} act", flush=True)

    print("Loading K562 model ...", flush=True)
    from model import K562Model
    model = K562Model(device="cuda")

    OUT.mkdir(parents=True, exist_ok=True)
    seed_rows = []
    summary_rows = []

    for vname in variants:
        if vname not in SEEDERS:
            print(f"  !! unknown variant {vname!r}, skipping", flush=True)
            continue
        TSA.stage1_necessity = make_stage1(SEEDERS[vname])   # swap ONLY Stage 1
        vout = OUT / vname
        vout.mkdir(parents=True, exist_ok=True)
        print(f"\n===== variant: {vname} =====", flush=True)

        rows, ann_ids, ann_regions, native, classified = [], [], [], [], {}
        for ri, rec in usable:
            t0 = time.time()
            attr = attr_ism(rec)
            adapter = K562Adapter(model, rec["left"], rec["right"])

            res = TSA.run_two_stage(adapter, rec["var"], attr, head_index=0,
                                    target_recovery=TARGET,
                                    num_shuffles=C.NUM_SHUFFLES, seed=C.SEED,
                                    verbose=False)

            regs = M.merge_regions(res["regions"])
            cls = classify(res)
            ctx = res["ctx"]                              # the pipeline's own context
            wsig = res["wt_signal"] if abs(res["wt_signal"]) > 1e-6 else float("nan")
            final_nec = ctx.knock_out(regs)["nec"] if regs else 0.0
            seed_bp = _bp([(m["start"], m["end"]) for m in res["naive"]])
            core_bp, flank_bp, ctx_bp = (_bp(cls["core"]), _bp(cls["flank"]),
                                         _bp(cls["context"]))
            rec_row = {
                "variant": vname, "seq_id": rec["id"], "activity_bin": bins[ri],
                "true": rec["true"], "pred": rec["pred"], "wt_signal": res["wt_signal"],
                "n_seeds": len(res["naive"]), "seed_bp": seed_bp,
                "n_regions": len(regs), "total_bp": _bp(regs),
                "core_bp": core_bp, "flank_bp": flank_bp, "context_bp": ctx_bp,
                "n_motifs": len(res["motifs"]), "n_buffered": len(res["buffered"]),
                "rec_naive": res["recovery"]["naive"],
                "rec_stage1": res["recovery"]["stage1"],
                "rec_extended": res["recovery"]["extended"],
                "rec_final": res["recovery"]["final"],
                "necessity": final_nec, "nec_fraction": final_nec / wsig,
                "recovery": res["recovery"]["final"],      # alias (benchmark column name)
                "status": res["status"], "n_experiments": res["n_experiments"],
            }
            rows.append(rec_row)
            seed_rows.append(rec_row)

            ann_ids.append(rec["id"])
            ann_regions.append(regs)
            native.append({"id": rec["id"], "status": res["status"],
                           "recovery": res["recovery"],
                           "n_experiments": res["n_experiments"]})
            classified[rec["id"]] = cls
            print(f"  [{rec['id']}] {bins[ri]} seeds={len(res['naive'])} "
                  f"core={core_bp}bp ctx={ctx_bp}bp final={res['recovery']['final']:.0%} "
                  f"{res['status']} ({time.time()-t0:.1f}s)", flush=True)

        df = pd.DataFrame(rows)
        df.to_csv(vout / "per_sequence.csv", index=False)
        with open(vout / "annotations.pkl", "wb") as fh:
            pickle.dump({"ids": ann_ids, "regions": {vname: ann_regions},
                         "native": native, "classified": classified}, fh)

        # per-variant medians (annotated seqs = all; some may be empty-core)
        summary_rows.append({
            "variant": vname, "n_seq": len(df),
            "med_n_seeds": df["n_seeds"].median(),
            "med_seed_bp": df["seed_bp"].median(),
            "med_rec_naive": df["rec_naive"].median(),
            "med_rec_stage1": df["rec_stage1"].median(),
            "med_rec_final": df["rec_final"].median(),
            "med_total_bp": df["total_bp"].median(),
            "med_core_bp": df["core_bp"].median(),
            "med_context_bp": df["context_bp"].median(),
            "frac_zero_core": float((df["core_bp"] == 0).mean()),
            "med_n_experiments": df["n_experiments"].median(),
            "pct_captured_stage1": float((df["status"] == "captured_at_stage1").mean()),
            "pct_captured_buffered": float((df["status"] == "captured_by_buffered").mean()),
        })
        print(f"  -> wrote {vout/'per_sequence.csv'}", flush=True)

    pd.DataFrame(seed_rows).to_csv(OUT / "seed_comparison.csv", index=False)
    summ = pd.DataFrame(summary_rows)
    summ.to_csv(OUT / "explore_summary.csv", index=False)
    print("\nwrote", OUT / "seed_comparison.csv", flush=True)
    print("wrote", OUT / "explore_summary.csv", flush=True)
    print("\n" + summ.round(3).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
