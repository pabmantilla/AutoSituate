"""Post-sufficiency counter-element (repressor) rescue — EXPERIMENTAL, step1_explore only.

Motivation (see analysis in seed_comparison): varying ONLY Stage-1 seeding does not
fix the repressor problem. Two-sided seeders place seeds on repressor positions, but
the Stage-1 `nec > 0` gate discards them, so recovery stays > 100% on ~35% of
repressive sequences regardless of seeder. The overshoot is most obvious AFTER the
sufficiency optimisation: the knocked-in set of annotated motifs reproduces MORE than
the whole WT effect, so WT must contain a counter-element pushing the opposite way that
the annotation missed.

This module is a logic gate applied to the FINISHED `run_two_stage` result, using the
ShuffleContext it returns (`res["ctx"]`). It does NOT touch the main algorithm and does
NOT move any existing annotation:

    if final recovery overshoots 1.0 by > tol:
        seed candidates from the OPPOSITE-sign CWM tail (repressor for an over-
        activating seq, activator for an over-repressing seq),
        greedily add the candidate that pulls |recovery - 1| down the most,
        keeping the existing annotated set fixed,
        stop when within tol of 1.0, no candidate helps, or max_add reached.

Returns the counter-elements found and the rescued recovery; the caller records both
alongside the untouched native annotation.
"""

import numpy as np

# Reuse the exact helpers the rest of the pipeline uses (imported by the driver's
# sys.path setup: step1_explore + k562_agft_benchmark are both importable).
from seeders import _runs, _cwm_track          # step1_explore
from two_stage_agent import merge_regions, _overlaps  # k562_agft_benchmark


def rescue_counter_element(res, attr_map, onehot, num_std=1.5, tol=0.05,
                           max_add=2, min_len=2):
    """Find counter-elements that pull an overshooting recovery back toward 1.0.

    `res`        : a completed two_stage_agent.run_two_stage() result (carries 'ctx',
                   'regions', 'wt_signal', 'recovery').
    `attr_map`   : (L, 4) raw ISM map for this sequence (same one Stage-1 saw).
    `onehot`     : (L, 4) WT one-hot.
    `tol`        : how close to 1.0 counts as "recovered" (gate is recovery > 1 + tol).
    `max_add`    : cap on counter-elements added (one forward-pass batch each).

    Does not mutate `res`. Returns a dict:
        triggered, counter_elements[{start,end,sign,rec_after}], rec_before,
        rec_after, n_experiments_added.
    """
    ctx = res["ctx"]
    wt_signal = res["wt_signal"]
    fixed = [tuple(r) for r in merge_regions(res["regions"])]
    rec0 = res["recovery"]["final"]

    out = {"triggered": False, "counter_elements": [], "rec_before": rec0,
           "rec_after": rec0, "n_experiments_added": 0}

    # Gate: only when the annotated set reproduces MORE than the whole WT effect.
    if not (rec0 > 1.0 + tol):
        return out
    out["triggered"] = True

    # The missing element pushes OPPOSITE to the net WT direction:
    #   over-activating (wt_signal > 0) -> a repressor lives in the negative CWM tail
    #   over-repressing (wt_signal < 0) -> an activator lives in the positive CWM tail
    s = _cwm_track(attr_map, onehot)
    mu, sd = s.mean(), s.std()
    if wt_signal >= 0:
        mask, sign = (s - mu) < -num_std * sd, "repressor"
    else:
        mask, sign = (s - mu) > num_std * sd, "activator"

    cands = [(c["start"], c["end"]) for c in _runs(mask, s, min_len)]
    cands = [c for c in cands if not _overlaps(c, fixed)]

    e0 = ctx.n_experiments
    current = list(fixed)
    cur_rec = rec0
    for _ in range(max_add):
        if abs(cur_rec - 1.0) <= tol or not cands:
            break
        scored = ctx.knock_in_many([current + [c] for c in cands])
        dists = [abs(sc["recovery"] - 1.0) for sc in scored]
        best_i = int(np.argmin(dists))
        # Accept only a candidate that STRICTLY reduces the distance to 1.0.
        if dists[best_i] < abs(cur_rec - 1.0) - 1e-6:
            c = cands.pop(best_i)
            cur_rec = scored[best_i]["recovery"]
            current.append(c)
            out["counter_elements"].append(
                {"start": int(c[0]), "end": int(c[1]), "sign": sign,
                 "rec_after": float(cur_rec)})
        else:
            break

    out["rec_after"] = float(cur_rec)
    out["n_experiments_added"] = ctx.n_experiments - e0
    return out
