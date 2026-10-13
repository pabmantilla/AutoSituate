"""Necessity-path primitives for the two-stage annotator, vendored from
AutoSituate's hackathon agent (annotator_agent.py / virtual_experimentalist.py)
and decoupled from TensorFlow so they run in the PyTorch K562 benchmark env.

Only the necessity path that `two_stage_agent` actually uses is kept:
    naive_motif_detection  — CWM thresholding
    necessity_test         — marginalized knock-out
    optimize_boundaries    — expand-only necessity boundaries, merge-on-collision
    optimize_all           — all motifs, strongest-first, merge-on-collision

The algorithm is unchanged; the only edit vs the original is the import of
`dinuc_shuffle` (reused from the benchmark's metrics module) and dropping the
unused _suff / window-KO / paired-test machinery. `model` here is any object
exposing `model.predict(x, verbose=0)[head_index]` on (B, L, 4) input — the
K562 adapter supplies that (see two_stage_adapter.py).
"""

import numpy as np

from metrics import dinuc_shuffle


# ----------------------------------------------------------------------
# Naive motif detection — CWM thresholding
# ----------------------------------------------------------------------

def naive_motif_detection(attribution_map, onehot, num_std=2.0, min_len=1):
    """Detect candidate motifs via thresholding on the CWM.

    Threshold = mean + num_std * std of per-position CWM scores; contiguous
    above-threshold positions of length >= min_len become regions.
    """
    cwm = attribution_map * onehot
    pos_scores = cwm.sum(axis=1)

    threshold = pos_scores.mean() + num_std * pos_scores.std()
    above = pos_scores > threshold

    regions = []
    start = None
    for i in range(len(above)):
        if above[i] and start is None:
            start = i
        elif not above[i] and start is not None:
            if i - start >= min_len:
                regions.append({"start": start, "end": i,
                                "sum_attr": float(pos_scores[start:i].sum())})
            start = None
    if start is not None and len(above) - start >= min_len:
        regions.append({"start": start, "end": len(above),
                        "sum_attr": float(pos_scores[start:].sum())})
    return regions


# ----------------------------------------------------------------------
# Necessity test — marginalized knock-out
# ----------------------------------------------------------------------

def necessity_test(model, sequence, start, end, head_index=0,
                   num_shuffles=100, seed=None):
    """necessity_score = mean(WT_pred - chimeric_preds), where each chimera is
    WT with [start:end] replaced by that of a dinuc-shuffle of the full WT."""
    L, A = sequence.shape

    wt_input = sequence[np.newaxis]
    wt_pred = float(model.predict(wt_input, verbose=0)[head_index][0, 0])

    rng = np.random.default_rng(seed)
    shuffled_seqs = np.array([dinuc_shuffle(sequence, rng)
                              for _ in range(num_shuffles)])

    chimeras = np.tile(sequence[np.newaxis], (num_shuffles, 1, 1))
    chimeras[:, start:end, :] = shuffled_seqs[:, start:end, :]

    chimeric_preds = model.predict(chimeras, verbose=0)[head_index].flatten()
    necessity_score = float(np.mean(wt_pred - chimeric_preds))

    return {"necessity_score": necessity_score, "wt_pred": wt_pred,
            "chimeric_preds": chimeric_preds, "start": start, "end": end}


# ----------------------------------------------------------------------
# Single-motif boundary expansion — NECESSITY objective
# ----------------------------------------------------------------------

def optimize_boundaries(model, sequence, motif, head_index=0,
                        num_shuffles=100, seed=None, verbose=True,
                        claimed_owner=None, finalized=None, max_expand=15,
                        naive_index=None):
    """Expand-only necessity boundaries with merge-on-collision.

    Accept a 1bp flank only if necessity strictly increases. Running into a bp
    already claimed by a stronger, already-finalized motif MERGES the two (free,
    does not consume max_expand) and expansion continues from the merged span.
    """
    L = sequence.shape[0]
    start, end = motif["start"], motif["end"]
    claimed_owner = claimed_owner if claimed_owner is not None else {}
    finalized = finalized if finalized is not None else {}
    history = []
    naive_indices = [naive_index] if naive_index is not None else []
    merged_naive_start, merged_naive_end = start, end

    def _nec(s, e):
        return necessity_test(model, sequence, s, e, head_index=head_index,
                              num_shuffles=num_shuffles, seed=seed)["necessity_score"]

    def _absorb(owner_id):
        owner = finalized.pop(owner_id)
        for p in range(owner["start"], owner["end"]):
            claimed_owner.pop(p, None)
        return owner

    def _fold_in(owner):
        nonlocal merged_naive_start, merged_naive_end
        merged_naive_start = min(merged_naive_start, owner["naive_start"])
        merged_naive_end = max(merged_naive_end, owner["naive_end"])
        naive_indices.extend(owner["naive_indices"])

    # Absorb any pre-existing overlap in the naive span itself
    cur_s, cur_e = start, end
    changed = True
    while changed:
        changed = False
        for p in range(cur_s, cur_e):
            if p in claimed_owner:
                owner = _absorb(claimed_owner[p])
                cur_s = min(cur_s, owner["start"])
                cur_e = max(cur_e, owner["end"])
                _fold_in(owner)
                changed = True
                break

    naive_nec = _nec(cur_s, cur_e)
    naive_len = cur_e - cur_s
    naive_nec_per_bp = naive_nec / naive_len if naive_len > 0 else 0.0

    cur_nec = naive_nec
    n_added = 0
    history.append((cur_s, cur_e, naive_nec, naive_nec_per_bp))

    stop_L = stop_R = None
    while n_added < max_expand:
        expanded_this_round = False

        lp = cur_s - 1
        if lp < 0:
            stop_L = stop_L or "edge"
        elif lp in claimed_owner:
            owner = _absorb(claimed_owner[lp])
            cur_s = min(cur_s, owner["start"]); cur_e = max(cur_e, owner["end"])
            _fold_in(owner); cur_nec = _nec(cur_s, cur_e); stop_L = None
            history.append((cur_s, cur_e, cur_nec, cur_nec / (cur_e - cur_s)))
            expanded_this_round = True
            if verbose:
                print(f"    [{motif['start']:3d},{motif['end']:3d}) MERGED left "
                      f"with [{owner['start']:3d},{owner['end']:3d}) -> "
                      f"[{cur_s:3d},{cur_e:3d}) nec={cur_nec:.4f}")
        else:
            new_nec = _nec(lp, cur_e)
            if new_nec > cur_nec:
                cur_s = lp; cur_nec = new_nec; n_added += 1; stop_L = None
                history.append((cur_s, cur_e, cur_nec, cur_nec / (cur_e - cur_s)))
                expanded_this_round = True
            else:
                stop_L = f"nec {new_nec:.4f}<={cur_nec:.4f}"

        rp = cur_e
        if rp >= L:
            stop_R = stop_R or "edge"
        elif rp in claimed_owner:
            owner = _absorb(claimed_owner[rp])
            cur_s = min(cur_s, owner["start"]); cur_e = max(cur_e, owner["end"])
            _fold_in(owner); cur_nec = _nec(cur_s, cur_e); stop_R = None
            history.append((cur_s, cur_e, cur_nec, cur_nec / (cur_e - cur_s)))
            expanded_this_round = True
            if verbose:
                print(f"    [{motif['start']:3d},{motif['end']:3d}) MERGED right "
                      f"with [{owner['start']:3d},{owner['end']:3d}) -> "
                      f"[{cur_s:3d},{cur_e:3d}) nec={cur_nec:.4f}")
        else:
            new_nec = _nec(cur_s, cur_e + 1)
            if new_nec > cur_nec:
                cur_e += 1; cur_nec = new_nec; n_added += 1; stop_R = None
                history.append((cur_s, cur_e, cur_nec, cur_nec / (cur_e - cur_s)))
                expanded_this_round = True
            else:
                stop_R = f"nec {new_nec:.4f}<={cur_nec:.4f}"

        if not expanded_this_round:
            break

    final_nec_per_bp = cur_nec / (cur_e - cur_s) if cur_e > cur_s else 0.0

    return {
        "start": cur_s, "end": cur_e, "nec": cur_nec,
        "nec_per_bp": final_nec_per_bp, "naive_nec": naive_nec,
        "naive_nec_per_bp": naive_nec_per_bp,
        "naive_start": merged_naive_start, "naive_end": merged_naive_end,
        "naive_indices": naive_indices, "n_expanded": n_added, "history": history,
    }


# ----------------------------------------------------------------------
# Multi-motif optimization with merge-on-collision — NECESSITY objective
# ----------------------------------------------------------------------

def optimize_all(model, sequence, naive_motifs, head_index=0,
                 num_shuffles=100, seed=None, verbose=True, max_expand=15):
    """Optimize every motif, strongest-first, merging on collision.

    Returns finalized motif dicts (possibly fewer than input if merges occurred),
    sorted by start, each carrying 'naive_indices' and 'merged'.
    """
    n = len(naive_motifs)

    naive_scores = []
    for m in naive_motifs:
        r = necessity_test(model, sequence, m["start"], m["end"],
                           head_index=head_index, num_shuffles=num_shuffles, seed=seed)
        naive_scores.append(r["necessity_score"])

    order = np.argsort(naive_scores)[::-1]
    claimed_owner = {}
    finalized = {}
    next_id = 0

    for idx in order:
        m = naive_motifs[idx]
        opt = optimize_boundaries(
            model, sequence, m, head_index=head_index,
            num_shuffles=num_shuffles, seed=seed, verbose=verbose,
            claimed_owner=claimed_owner, finalized=finalized,
            max_expand=max_expand, naive_index=int(idx),
        )
        entry_id = next_id
        next_id += 1
        finalized[entry_id] = {
            "start": opt["start"], "end": opt["end"],
            "nec": opt["nec"], "nec_per_bp": opt["nec_per_bp"],
            "naive_nec": opt["naive_nec"], "naive_nec_per_bp": opt["naive_nec_per_bp"],
            "naive_start": opt["naive_start"], "naive_end": opt["naive_end"],
            "naive_indices": opt["naive_indices"], "n_expanded": opt["n_expanded"],
            "merged": len(opt["naive_indices"]) > 1,
        }
        for p in range(opt["start"], opt["end"]):
            claimed_owner[p] = entry_id

    return sorted(finalized.values(), key=lambda r: r["start"])
