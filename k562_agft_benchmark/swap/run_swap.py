"""All-by-all motif-syntax x context swap, per annotation method.

Same chimera protocol as mcs_k562/large_scale_swap/scripts/score_swap.py, but the
motif mask comes from each benchmark annotation method's called regions (results/
annotations.pkl) instead of a single SEAM annotation. Everything stays in the 200bp
variable-region frame and reuses the benchmark's model / dinuc-shuffle, so scores are
on the same footing as the KO/KI figures.

For each method, for every ordered pair (donor d, recipient t):
  1. context(t): dinuc-shuffle recipient var, paste the shuffle onto t's motif
     positions -> recipient with its own motif syntax knocked out. NREP reps.
  2. inject donor d's motif columns (from d's WT var) onto d's motif positions.
  chimera = recipient context + donor motif syntax; adapters held WT (recipient's).
  score = mean K562 pred over NREP reps.

Out (results/swap_scores_by_method.npz):
  swap__<method>            (K,K) float32  [donor, recipient] mean pred
  motif_shuffle__<method>   (K,)  float32  recipient w/ MOTIF shuffled, context WT (motif KO)
  context_shuffle__<method> (K,)  float32  recipient w/ CONTEXT shuffled, motif WT (motif only)
  wt (K,), names (K,), bins (K,)
"""

import pickle
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))          # benchmark package (config/data/model/...)

import config as C          # noqa: E402
import data                 # noqa: E402
import metrics as M         # noqa: E402

NREP = 30
PER_BIN = 10        # sequences kept per activity bin (balanced subset)
RESULTS = HERE / "results"


def select_subset(masks, bins, per_bin=PER_BIN):
    """Balanced per-bin subset, prioritizing sequences annotated across methods.

    coverage = how many methods called >=1 motif bp on that sequence; ties broken
    by total annotated bp summed over methods. Returns sorted global indices.
    """
    stack = np.stack([masks[m] for m in masks])          # (M,K,200)
    coverage = (stack.sum(2) > 0).sum(0)                 # (K,) methods with a motif
    total_bp = stack.sum(2).sum(0)                       # (K,) tiebreak
    keep = []
    for b in ["activating", "neutral", "repressive"]:
        idx = np.where(bins == b)[0]
        idx = sorted(idx, key=lambda i: (-coverage[i], -total_bp[i], i))[:per_bin]
        keep += idx
        print(f"  {b}: kept {len(idx)} (coverage {[int(coverage[i]) for i in idx]})",
              flush=True)
    return np.array(sorted(keep))


def build_masks(recs, method_regions):
    """{method: (K,200) bool mask} from merged per-sequence regions (var frame)."""
    masks = {}
    for method, per_seq in method_regions.items():
        mm = np.zeros((len(recs), C.L_VAR), dtype=bool)
        for i, regs in enumerate(per_seq):
            for s, e in M.merge_regions(regs):
                mm[i, s:e] = True
        masks[method] = mm
    return masks


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    recs = data.load_dataset()
    bins, _ = data.activity_bins(recs)
    names = np.array([r["id"] for r in recs])
    bins = np.array(bins)
    K = len(recs)

    ann = pickle.load(open(C.RESULTS / "annotations.pkl", "rb"))
    assert ann["ids"] == names.tolist(), "annotations.pkl order != dataset order"
    # keep only the figure methods, in figure order (drops any stale cached method)
    from visualize import FIGURE_ORDER
    method_regions = {m: ann["regions"][m] for m in FIGURE_ORDER if m in ann["regions"]}
    masks = build_masks(recs, method_regions)
    methods = list(method_regions)
    print(f"{K} sequences (full) | methods: {methods}", flush=True)

    # balanced per-bin subset, prioritizing cross-method annotation coverage
    sel = select_subset(masks, bins, PER_BIN)
    recs = [recs[i] for i in sel]
    bins = bins[sel]
    names = names[sel]
    masks = {m: masks[m][sel] for m in methods}
    K = len(recs)
    print(f"-> subset to {K} sequences ({PER_BIN}/bin)", flush=True)
    for m in methods:
        print(f"  [{m}] seqs with >=1 motif bp: {(masks[m].sum(1) > 0).sum()}/{K}",
              flush=True)

    var = np.stack([r["var"] for r in recs]).astype(np.float32)   # (K,200,4)
    lefts = [r["left"] for r in recs]
    rights = [r["right"] for r in recs]

    # one fixed set of NREP dinuc shuffles per recipient (seed 1000+t), reused by
    # every method -- the context carrier only differs by which positions are pasted.
    shuffles = np.stack([
        np.stack([M.dinuc_shuffle(var[t], np.random.default_rng(1000 + t))
                  for _ in range(NREP)])
        for t in range(K)
    ])                                                            # (K,NREP,200,4)

    from model import K562Model
    model = K562Model(device="cuda")

    wt = np.array([float(model.predict_var(var[t][None], lefts[t], rights[t])[0])
                   for t in range(K)], np.float32)

    out = {"wt": wt, "names": names, "bins": bins}
    t0 = time.time()
    for m in methods:
        mask = masks[m]
        swap = np.zeros((K, K), np.float32)
        motif_shuffle = np.zeros(K, np.float32)      # shuffle MOTIF, keep context (motif KO'd)
        context_shuffle = np.zeros(K, np.float32)    # shuffle CONTEXT, keep motif (motif only)
        for t in range(K):                                        # recipient
            mt = mask[t]
            # context carrier = recipient with its MOTIF positions shuffled (motif KO)
            ctx = shuffles[t].copy()                              # (NREP,200,4)
            if not mt.any():
                ctx[:] = var[t][None]                             # no motifs -> context is WT
            else:
                ctx[:, ~mt, :] = var[t][None, ~mt, :]             # WT everywhere but motifs
            motif_shuffle[t] = model.predict_var(ctx, lefts[t], rights[t]).mean()

            # control: shuffle CONTEXT, keep the motif WT (motif on a dead scaffold)
            cs = shuffles[t].copy()
            if mt.any():
                cs[:, mt, :] = var[t][None, mt, :]
            context_shuffle[t] = model.predict_var(cs, lefts[t], rights[t]).mean()

            chim = np.repeat(ctx[None], K, axis=0)                # (K_donor,NREP,200,4)
            for d in range(K):                                    # donor motif injection
                md = mask[d]
                if md.any():
                    chim[d][:, md, :] = var[d][None, md, :]
            preds = model.predict_var(chim.reshape(K * NREP, C.L_VAR, 4),
                                      lefts[t], rights[t]).reshape(K, NREP)
            swap[:, t] = preds.mean(1)
        out[f"swap__{m}"] = swap
        out[f"motif_shuffle__{m}"] = motif_shuffle
        out[f"context_shuffle__{m}"] = context_shuffle
        out[f"motif_bp__{m}"] = mask.sum(1).astype(np.int32)   # 0 = method found no motif
        print(f"  [{m}] done ({time.time()-t0:.1f}s)  diag mean={np.diag(swap).mean():+.3f}"
              f"  motif_shuf={motif_shuffle.mean():+.3f} ctx_shuf={context_shuffle.mean():+.3f}",
              flush=True)

    outp = RESULTS / "swap_scores_by_method.npz"
    np.savez(outp, **out)
    print(f"\nwrote {outp}  ({len(methods)} methods x {K}x{K})", flush=True)


if __name__ == "__main__":
    main()
