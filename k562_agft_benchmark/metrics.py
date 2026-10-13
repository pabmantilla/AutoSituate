"""Motif knock-out (necessity) and knock-in (sufficiency) metrics.

Protocol (per Pablo): dinucleotide-shuffle the WHOLE variable region once per
replicate, then
    KO (necessity)  : take the WT variable region and overwrite the called
                      positions with the shuffled sequence's positions.
    KI (sufficiency): take the shuffled variable region and overwrite the called
                      positions with the WT positions.
One fixed background set per sequence is shared by both tests and by every method,
so the numbers are directly comparable.

    necessity    = mean(WT_pred - KO_preds)
    sufficiency  = mean(KI_preds - shuffled_preds)
    wt_signal    = WT_pred - mean(shuffled_preds)
    nec_fraction = necessity / wt_signal
    recovery     = sufficiency / wt_signal     (knock-in recovery of WT activity)
"""

import numpy as np

import config as C


def dinuc_shuffle(sequence, rng):
    """Dinucleotide-preserving shuffle of a one-hot (L,4) sequence.

    Altschul & Erickson (1985): Eulerian path over the dinucleotide graph.
    (Same implementation as AutoSituate's input_handler.dinuc_shuffle.)
    """
    idx = np.argmax(sequence, axis=1)
    L, A = len(idx), sequence.shape[1]

    edges = {a: [] for a in range(A)}
    for i in range(L - 1):
        edges[idx[i]].append(idx[i + 1])
    for a in edges:
        rng.shuffle(edges[a])

    ptr = {a: 0 for a in range(A)}
    stack, path = [idx[0]], []
    while stack:
        v = stack[-1]
        if ptr[v] < len(edges[v]):
            stack.append(edges[v][ptr[v]])
            ptr[v] += 1
        else:
            path.append(stack.pop())
    path.reverse()

    out = np.zeros_like(sequence)
    out[np.arange(L), path] = 1.0
    return out


class ShuffleContext:
    """Fixed dinuc-shuffled backgrounds + cached WT/shuffled preds for one seq."""

    def __init__(self, model, rec, num_shuffles=C.NUM_SHUFFLES, seed=C.SEED):
        self.model = model
        self.rec = rec
        rng = np.random.default_rng(seed)
        var = rec["var"]
        self.shuffled = np.stack([dinuc_shuffle(var, rng) for _ in range(num_shuffles)])
        self.wt_pred = float(model.predict_var(var[None], rec["left"], rec["right"])[0])
        self.shuffled_preds = model.predict_var(self.shuffled, rec["left"], rec["right"])
        self.shuf_mean = float(self.shuffled_preds.mean())
        self.wt_signal = self.wt_pred - self.shuf_mean


def joint_ko(ctx, regions):
    """Knock OUT all regions jointly. Returns necessity = mean(WT - KO)."""
    if not regions:
        return 0.0
    var = ctx.rec["var"]
    chim = np.tile(var[None], (ctx.shuffled.shape[0], 1, 1))
    for s, e in regions:
        chim[:, s:e, :] = ctx.shuffled[:, s:e, :]
    preds = ctx.model.predict_var(chim, ctx.rec["left"], ctx.rec["right"])
    return float((ctx.wt_pred - preds).mean())


def joint_ki(ctx, regions):
    """Knock IN all regions jointly. Returns sufficiency = mean(KI - shuffled)."""
    if not regions:
        return 0.0
    var = ctx.rec["var"]
    impl = ctx.shuffled.copy()
    for s, e in regions:
        impl[:, s:e, :] = var[None, s:e, :]
    preds = ctx.model.predict_var(impl, ctx.rec["left"], ctx.rec["right"])
    return float((preds - ctx.shuffled_preds).mean())


def _rejection_place(lengths, occupied, L, rng, max_tries):
    """Place windows (desc lengths) in FREE positions (occupied==False), non-overlapping.
    Returns list[(s,e)] or None if some window couldn't be placed."""
    placed = []
    for w in lengths:
        for _ in range(max_tries):
            start = int(rng.integers(0, L - w + 1))
            if not occupied[start:start + w].any():
                occupied[start:start + w] = True
                placed.append((start, start + w))
                break
        else:
            return None
    return placed


def _gap_partition(lengths, L, rng):
    """Lay windows anywhere in [0,L], non-overlapping, with the leftover space (slack)
    split into random gaps. Always succeeds for sum(lengths) <= L -- so a matched null
    exists even when the motifs cover most of the window."""
    order = list(lengths)
    rng.shuffle(order)
    slack = L - sum(order)
    gaps = (rng.multinomial(slack, np.ones(len(order) + 1) / (len(order) + 1))
            if slack > 0 else np.zeros(len(order) + 1, dtype=int))
    placed, pos = [], int(gaps[0])
    for w, g in zip(order, gaps[1:]):
        placed.append((pos, pos + w))
        pos += w + int(g)
    return sorted(placed)


def matched_null_regions(merged, rng, L=C.L_VAR, max_tries=200):
    """Window-matched null placement: SAME number of windows and SAME lengths as
    `merged` (hence same total bp). Tries to place them entirely OUTSIDE the motif
    positions (clean specificity control); if the motifs cover too much of the window
    for that to fit, FALLS BACK to random placement anywhere (windows may overlap the
    motifs). A matched null therefore always exists -- essential for over-annotating
    methods, where it shows the effect is just from knocking out that many bp, not from
    the motifs. Returns (list[(s,e)], placed_outside: bool)."""
    if not merged:
        return [], True
    lengths = sorted((e - s for s, e in merged), reverse=True)
    occupied = np.zeros(L, dtype=bool)
    for s, e in merged:
        occupied[s:e] = True
    if sum(lengths) <= L - int(occupied.sum()):             # room to fit outside motifs
        placed = _rejection_place(lengths, occupied.copy(), L, rng, max_tries)
        if placed is not None:
            return sorted(placed), True
    return _gap_partition(lengths, L, rng), False           # fallback: anywhere


def matched_null_scores(ctx, merged, n_draws=C.NULL_DRAWS, seed=C.SEED):
    """Window-matched null DISTRIBUTION: `n_draws` placements of same-size windows
    (outside the motifs where possible, else anywhere), each scored by both joint_ko
    and joint_ki (same shuffled backgrounds as the real tests). Returns
    (ko_vals, ki_vals, frac_outside) -- the empirical null distributions of necessity
    and sufficiency, and the fraction of draws placed strictly outside the motifs."""
    if not merged:
        return np.array([]), np.array([]), float("nan")
    rng = np.random.default_rng(seed)
    ko, ki, outside = [], [], 0
    for _ in range(n_draws):
        null, is_out = matched_null_regions(merged, rng)
        ko.append(joint_ko(ctx, null))
        ki.append(joint_ki(ctx, null))
        outside += int(is_out)
    return np.array(ko, float), np.array(ki, float), outside / n_draws


def null_significance(observed, null_vals):
    """Significance of `observed` against the window-matched null distribution.

    z     = (observed - null_mean) / null_std            signed: + for activating
            annotations (KI raises activity above random windows), - for repressive
            ones (KI lowers it below them).
    p_emp = (1 + #{|null - null_mean| >= |observed - null_mean|}) / (n + 1)
            TWO-sided: an annotation is significant when its effect is extreme in
            EITHER direction vs random off-motif windows, so repressive motifs (lower
            tail) are caught as well as activating ones. Returns (z, p_emp)."""
    n = len(null_vals)
    if n == 0 or not np.isfinite(observed):
        return float("nan"), float("nan")
    mu, sd = null_vals.mean(), null_vals.std()
    z = (observed - mu) / sd if sd > 0 else float("nan")
    p_emp = (1 + int((np.abs(null_vals - mu) >= abs(observed - mu)).sum())) / (n + 1)
    return float(z), float(p_emp)


def merge_regions(regions, L=C.L_VAR):
    """Clip to [0,L], drop empties, and merge overlapping/abutting spans.

    KO/KI act on a position set, so overlaps would otherwise double-apply; merging
    also makes `total_bp` an honest count of distinct annotated positions.
    """
    spans = sorted((max(0, int(s)), min(L, int(e))) for s, e in regions if int(e) > int(s))
    merged = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged
