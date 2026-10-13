"""A seeder-pluggable Stage 1, so only the initial step changes.

`make_stage1(seeder)` returns a drop-in replacement for
`two_stage_agent.stage1_necessity` with an IDENTICAL body except for one line:
the naive-threshold seed is replaced by `seeder(...)`. Everything else — the
expand-only necessity boundary optimisation (`optimize_all`) and the
length-matched-null spurious-call pruning — is byte-for-byte the original.

The driver monkeypatches `two_stage_agent.stage1_necessity` with this, then calls
the unchanged `two_stage_agent.run_two_stage`; stages 2a/2b are untouched.
"""

import numpy as np

from two_stage_core import optimize_all
from two_stage_agent import _null_necessity, merge_regions


def make_stage1(seeder):
    def stage1_necessity(model, sequence, attribution_map, ctx, num_std=1.5,
                         min_len=1, max_expand=15, n_null=16, null_z=2.0,
                         verbose=True):
        # >>> the only change vs two_stage_agent.stage1_necessity <<<
        naive = seeder(attribution_map, sequence, ctx,
                       num_std=num_std, min_len=min_len)
        # <<< everything below is the original, unchanged >>>
        if verbose:
            print(f"  seed ({seeder.__name__}): {len(naive)} calls")

        if not naive:
            return {"naive": [], "motifs": [], "pruned": [], "null_by_width": {}}

        opt = optimize_all(model, sequence, naive, head_index=ctx.head_index,
                           num_shuffles=ctx.num_shuffles, seed=ctx.seed,
                           verbose=False, max_expand=max_expand)

        occupied = merge_regions([(m["start"], m["end"]) for m in opt])
        rng = np.random.default_rng(ctx.seed)
        null_by_width = {}
        motifs, pruned = [], []

        for m in opt:
            width = m["end"] - m["start"]
            if width not in null_by_width:
                mean, std, scores = _null_necessity(ctx, width, occupied, n_null, rng)
                null_by_width[width] = {"mean": mean, "std": std, "scores": scores}
            null = null_by_width[width]
            z = ((m["nec"] - null["mean"]) / null["std"]
                 if null["std"] > 1e-9 else np.inf)

            rec = dict(m)
            rec.update({
                "opt_start": m["start"], "opt_end": m["end"],
                "stage1_start": m["start"], "stage1_end": m["end"],
                "null_nec": null["mean"], "null_std": null["std"], "z": float(z),
            })
            (motifs if (z >= null_z and m["nec"] > 0) else pruned).append(rec)

        motifs.sort(key=lambda r: r["start"])
        pruned.sort(key=lambda r: r["start"])
        return {"naive": naive, "motifs": motifs, "pruned": pruned,
                "null_by_width": null_by_width}

    return stage1_necessity
