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


# ---------------------------------------------------------------------------
# Sign-aware necessity (5th variant) — capture repressors
# ---------------------------------------------------------------------------
# The baseline gate keeps a motif only if `nec > 0`, and the boundary optimiser
# (`optimize_boundaries`) grows a flank only if necessity strictly INCREASES.
# For a REPRESSIVE sequence a real repressor has NEGATIVE necessity (knocking it
# out raises the prediction), so both steps throw it away — the dominant repressor
# is seeded, then deleted, and a weaker region sneaks in via the sufficiency
# back-door (see peak55453).
#
# Minimal fix that reuses optimize_all VERBATIM: for a sequence whose WT signal is
# negative, run the necessity path against a sign-flipped view of the model. Then
# the repressor's necessity is positive, the expand-only optimiser grows it on
# |necessity|, and the `nec > 0` gate keeps it — all unchanged. We flip the stored
# `nec` back to its true sign before returning, so the rest of the pipeline (stage
# 2, reporting) sees the real signed necessity. Stages 2a/2b are untouched.


class _SignedModel:
    """Model view whose predictions are multiplied by `sgn` (used only for the
    necessity computations inside this Stage 1)."""

    def __init__(self, model, sgn):
        self._m, self.sgn = model, sgn

    def predict(self, x, verbose=0):
        out = self._m.predict(x, verbose=verbose)
        return [self.sgn * o for o in out]


class _SignedCtx:
    """Proxy over a ShuffleContext that negates knock-out necessity, so the
    length-matched null is in the same flipped space as _SignedModel."""

    def __init__(self, ctx, sgn):
        self._c, self.sgn = ctx, sgn
        self.head_index = ctx.head_index
        self.num_shuffles = ctx.num_shuffles
        self.seed = ctx.seed
        self.L = ctx.L

    def _flip(self, r):
        r = dict(r)
        r["nec"] = self.sgn * r["nec"]
        r["nec_per_bp"] = self.sgn * r["nec_per_bp"]
        return r

    def knock_out(self, regions):
        return self._flip(self._c.knock_out(regions))

    def knock_out_many(self, region_sets):
        return [self._flip(r) for r in self._c.knock_out_many(region_sets)]


def make_stage1_signed(seeder):
    """Like make_stage1, but necessity is evaluated in the WT-signal direction, so
    repressors in repressive sequences are kept instead of pruned."""

    def stage1_necessity(model, sequence, attribution_map, ctx, num_std=1.5,
                         min_len=1, max_expand=15, n_null=16, null_z=2.0,
                         verbose=True):
        sgn = 1.0 if ctx.wt_signal >= 0 else -1.0
        smodel = _SignedModel(model, sgn)
        sctx = _SignedCtx(ctx, sgn)

        naive = seeder(attribution_map, sequence, ctx,
                       num_std=num_std, min_len=min_len)
        if verbose:
            print(f"  signed seed ({seeder.__name__}, sgn={sgn:+.0f}): "
                  f"{len(naive)} calls")
        if not naive:
            return {"naive": [], "motifs": [], "pruned": [], "null_by_width": {}}

        # optimize_all unchanged, but on the sign-flipped model -> expands on |nec|
        opt = optimize_all(smodel, sequence, naive, head_index=ctx.head_index,
                           num_shuffles=ctx.num_shuffles, seed=ctx.seed,
                           verbose=False, max_expand=max_expand)

        occupied = merge_regions([(m["start"], m["end"]) for m in opt])
        rng = np.random.default_rng(ctx.seed)
        null_by_width = {}
        motifs, pruned = [], []

        for m in opt:
            width = m["end"] - m["start"]
            if width not in null_by_width:
                mean, std, scores = _null_necessity(sctx, width, occupied, n_null, rng)
                null_by_width[width] = {"mean": mean, "std": std, "scores": scores}
            null = null_by_width[width]
            # m["nec"] is flipped (positive == effect in the WT direction); gate and
            # z use that flipped space, consistent with the flipped null.
            z = ((m["nec"] - null["mean"]) / null["std"]
                 if null["std"] > 1e-9 else np.inf)
            keep = (z >= null_z and m["nec"] > 0)

            rec = dict(m)
            rec["nec"] = sgn * m["nec"]                       # restore true sign
            rec["nec_per_bp"] = sgn * m.get("nec_per_bp", 0.0)
            rec["sign"] = "repressor" if sgn < 0 else "activator"
            rec.update({
                "opt_start": m["start"], "opt_end": m["end"],
                "stage1_start": m["start"], "stage1_end": m["end"],
                "null_nec": sgn * null["mean"], "null_std": null["std"],
                "z": float(z),
            })
            (motifs if keep else pruned).append(rec)

        motifs.sort(key=lambda r: r["start"])
        pruned.sort(key=lambda r: r["start"])
        if verbose:
            print(f"  signed null-gate (|effect| in WT direction, keep z>={null_z}): "
                  f"{len(motifs)} kept, {len(pruned)} pruned")
        return {"naive": naive, "motifs": motifs, "pruned": pruned,
                "null_by_width": null_by_width}

    return stage1_necessity


# ---------------------------------------------------------------------------
# Abs necessity (per-region) — capture repressors AND activators in one pass
# ---------------------------------------------------------------------------
# make_stage1_signed flips sign ONCE per sequence off `wt_signal`, so it only
# works when the whole sequence has one dominant direction. The abs variant is
# per-REGION: score each seed's raw necessity, split the seeds into a positive
# group (activators) and a negative group (repressors), and run the UNCHANGED
# optimize_all on each group against the matching sign-flipped model view. Then
# both the expand-only boundary optimiser and the `nec > 0` gate operate on the
# magnitude of each region's own effect, so a repressor is grown and kept even
# inside an activating sequence (and vice-versa). optimize_all / optimize_boundaries
# / _null_necessity / the gate are all reused verbatim; stages 2a/2b are untouched.


def make_stage1_abs(seeder):
    def stage1_necessity(model, sequence, attribution_map, ctx, num_std=1.5,
                         min_len=1, max_expand=15, n_null=16, null_z=2.0,
                         verbose=True):
        naive = seeder(attribution_map, sequence, ctx,
                       num_std=num_std, min_len=min_len)
        if verbose:
            print(f"  abs seed ({seeder.__name__}): {len(naive)} calls")
        if not naive:
            return {"naive": [], "motifs": [], "pruned": [], "null_by_width": {}}

        # 1. raw necessity of each seed -> orientation (+1 activator, -1 repressor)
        raw = [ctx.knock_out([(m["start"], m["end"])])["nec"] for m in naive]
        groups = {1.0: [m for m, v in zip(naive, raw) if v >= 0],
                  -1.0: [m for m, v in zip(naive, raw) if v < 0]}

        # 2. optimize each orientation's seeds against its sign-flipped model view,
        #    so optimize_all expands on |nec| and reports nec>0 in that direction.
        opt = []
        for sgn, grp in groups.items():
            if not grp:
                continue
            smodel = _SignedModel(model, sgn)
            for r in optimize_all(smodel, sequence, grp, head_index=ctx.head_index,
                                  num_shuffles=ctx.num_shuffles, seed=ctx.seed,
                                  verbose=False, max_expand=max_expand):
                r = dict(r); r["_sgn"] = sgn
                opt.append(r)

        # 3. the two passes claim bp independently -> resolve cross-orientation
        #    overlaps greedily, strongest |nec| wins.
        opt.sort(key=lambda r: abs(r["nec"]), reverse=True)
        kept = []
        for r in opt:
            if any(r["start"] < k["end"] and r["end"] > k["start"] for k in kept):
                continue
            kept.append(r)
        opt = sorted(kept, key=lambda r: r["start"])

        occupied = merge_regions([(m["start"], m["end"]) for m in opt])
        rng = np.random.default_rng(ctx.seed)
        null_by_width = {}
        motifs, pruned = [], []

        for m in opt:
            sgn = m["_sgn"]
            sctx = _SignedCtx(ctx, sgn)
            width = m["end"] - m["start"]
            key = (width, sgn)                       # null floor is sign-specific
            if key not in null_by_width:
                mean, std, scores = _null_necessity(sctx, width, occupied, n_null, rng)
                null_by_width[key] = {"mean": mean, "std": std, "scores": scores}
            null = null_by_width[key]
            z = ((m["nec"] - null["mean"]) / null["std"]
                 if null["std"] > 1e-9 else np.inf)
            keep = (z >= null_z and m["nec"] > 0)    # nec>0 in region's own direction

            rec = dict(m)
            rec["nec"] = sgn * m["nec"]                       # restore true sign
            rec["nec_per_bp"] = sgn * m.get("nec_per_bp", 0.0)
            rec["sign"] = "repressor" if sgn < 0 else "activator"
            rec.update({
                "opt_start": m["start"], "opt_end": m["end"],
                "stage1_start": m["start"], "stage1_end": m["end"],
                "null_nec": sgn * null["mean"], "null_std": null["std"],
                "z": float(z),
            })
            (motifs if keep else pruned).append(rec)

        motifs.sort(key=lambda r: r["start"])
        pruned.sort(key=lambda r: r["start"])
        if verbose:
            n_rep = sum(r["sign"] == "repressor" for r in motifs)
            print(f"  abs null-gate (per-region |effect|, keep z>={null_z}): "
                  f"{len(motifs)} kept ({n_rep} repressor), {len(pruned)} pruned")
        return {"naive": naive, "motifs": motifs, "pruned": pruned,
                "null_by_width": null_by_width}

    return stage1_necessity
