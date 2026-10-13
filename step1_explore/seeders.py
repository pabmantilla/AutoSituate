"""Stage-1 seeders — the ONLY thing this exploration varies.

Each seeder maps one sequence's (attribution_map, one-hot) to a list of candidate
seed regions, in the exact format the two-stage pipeline's naive step produces:

    [{"start": int, "end": int, "sum_attr": float}, ...]

Everything downstream (necessity boundary expansion, length-matched null pruning,
stage-2 sufficiency extension, stage-2b buffered search) is the unchanged
two_stage_agent code — see stage1_variants.make_stage1.

Uniform signature so they are interchangeable:
    seeder(attr_map, onehot, ctx, num_std=1.5, min_len=1) -> list[dict]
`attr_map` and `onehot` are (200, 4). `ctx` is the pipeline's ShuffleContext
(gives model knock-out access); only `adaptive_nec` uses it.

The four variants
-----------------
naive_pos     mean + k*std on the CWM, POSITIVE tail only. What the pipeline does
              today — blind to repressive (negative-attribution) positions.
naive_abs     mean +/- k*std, i.e. threshold |CWM - mean|. Two-sided, so strong
              NEGATIVE positions (repressor cores) are seeded too.
adaptive_nec  sweep the threshold from strict -> loose; at each level score the
              seed set's necessity (model knock-out) and keep the threshold with
              the best necessity-PER-BP. Lets the model, not a fixed cutoff, say
              how deep to threshold.
seqlets       tangermeme recursive seqlet calling on |CWM| instead of a global
              mean/std cutoff.
"""

import numpy as np

# merge_regions lives in the (unchanged) agent; reuse it so seeds dedupe exactly
# as the rest of the pipeline expects.
from two_stage_agent import merge_regions


def _runs(mask, scores, min_len):
    """Contiguous True-runs of `mask` (len L) -> seed dicts carrying summed attr."""
    regs, i, L = [], 0, len(mask)
    while i < L:
        if mask[i]:
            j = i
            while j < L and mask[j]:
                j += 1
            if j - i >= min_len:
                regs.append({"start": i, "end": j, "sum_attr": float(scores[i:j].sum())})
            i = j
        else:
            i += 1
    return regs


def _cwm_track(attr_map, onehot):
    """Per-position contribution of the actual base = (attr * one-hot).sum(channels)."""
    return (attr_map * onehot).sum(axis=1)


# --- 1. naive positive-tail (today's behaviour) -----------------------------
def naive_pos(attr_map, onehot, ctx=None, num_std=1.5, min_len=1):
    s = _cwm_track(attr_map, onehot)
    thr = s.mean() + num_std * s.std()
    return _runs(s > thr, s, min_len)


# --- 2. naive on |CWM| (two-sided, catches repressors) ----------------------
def naive_abs(attr_map, onehot, ctx=None, num_std=1.5, min_len=1):
    s = _cwm_track(attr_map, onehot)
    mask = np.abs(s - s.mean()) > num_std * s.std()
    return _runs(mask, s, min_len)


# --- 3. adaptive: lower the threshold to optimise necessity ------------------
# Strict -> loose. Two-sided at every level (so repressors are in play), and the
# level kept is the one whose seed set has the highest necessity PER BP, i.e. the
# most necessity-dense cut rather than simply the loosest.
ADAPTIVE_STD_GRID = (3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0)


def adaptive_nec(attr_map, onehot, ctx, num_std=1.5, min_len=1):
    s = _cwm_track(attr_map, onehot)
    mu, sd = s.mean(), s.std()
    best = None  # (nec_per_bp, regions)
    for k in ADAPTIVE_STD_GRID:
        regs = _runs(np.abs(s - mu) > k * sd, s, min_len)
        if not regs:
            continue
        merged = merge_regions([(r["start"], r["end"]) for r in regs])
        bp = sum(e - b for b, e in merged)
        if bp == 0:
            continue
        nec = ctx.knock_out(merged)["nec"]      # one model forward pass (100 shuffles)
        nec_per_bp = nec / bp
        if nec > 0 and (best is None or nec_per_bp > best[0]):
            best = (nec_per_bp, regs)
    if best is not None:
        return best[1]
    # nothing scored positive necessity -> fall back to the two-sided 1.5-std cut
    return _runs(np.abs(s - mu) > 1.5 * sd, s, min_len)


# --- 4. seqlet calling to init ----------------------------------------------
def seqlets(attr_map, onehot, ctx=None, num_std=1.5, min_len=1, threshold=0.01):
    from tangermeme.seqlet import recursive_seqlets
    s = _cwm_track(attr_map, onehot).astype(np.float32)
    # recursive_seqlets is positive-only; call on |track| so repressor spans count.
    df = recursive_seqlets(np.abs(s)[None, :], threshold=threshold)
    regs = []
    for row in df.itertuples(index=False):
        if int(row.example_idx) != 0:
            continue
        a, b = int(row.start), int(row.end)
        if b - a >= min_len:
            regs.append({"start": a, "end": b, "sum_attr": float(s[a:b].sum())})
    return regs


SEEDERS = {
    "naive_pos":    naive_pos,
    "naive_abs":    naive_abs,
    "adaptive_nec": adaptive_nec,
    "seqlets":      seqlets,
}
