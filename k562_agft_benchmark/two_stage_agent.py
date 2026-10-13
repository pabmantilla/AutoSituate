"""Two-stage motif optimization: necessity boundaries, then sufficiency.

Vendored verbatim from AutoSituate's hackathon agent (agent/two_stage_agent.py);
the only edits are the three imports at the top (retargeted to the TF-free
benchmark modules) and dropping the Keras `__main__` CLI. The algorithm is
unchanged.

STAGE 1 — NECESSITY (what must be there)
    Threshold the attribution map for naive calls, then expand each call's
    boundaries while marginalized necessity strictly improves. Expansion is
    EXPAND-ONLY: the thresholded core is never trimmed, only flanks are added,
    because the flanks are assumed to modulate the core's strength rather than
    replace it. Strongest-first with merge-on-collision (CLAUDE.md protocol).
    Finally, calls that do not beat a LENGTH-MATCHED NULL are dropped — this is
    what removes the spurious annotations the naive cutoff produced.

STAGE 2 — SUFFICIENCY (is it enough)
    Knock the whole stage-1 set into dinucleotide-shuffled backgrounds and ask
    whether it reproduces WT activity (recovery >= target).

    2a  boundary check — maybe stage 1 simply stopped too early. Extend the set
        1bp at a time, scored by JOINT knock-in recovery, accepting only strict
        improvements. If that closes the gap, the answer was boundaries.

    2b  buffered-feature search — if extension is not enough, the missing
        regulatory logic sits in subsequences that are BUFFERED by the strong
        motifs: knocking them out changes nothing (necessity and ISM are blind
        to them) because the main motifs carry the output on their own. Only a
        knock-in can see them. So knock in the motifs plus one HALF of the
        sequence at a time — left half, right half, middle half. If no half is
        enough the feature is split across halves, so the winning half is
        committed and the search continues in what is left, knocking in ever
        more until sufficiency is met. Then each committed half is bisected
        back down (same three-halves split) and trimmed 1bp at a time to the
        smallest span that still holds the target.

All marginalization uses N=100 dinucleotide shuffles.
"""

import numpy as np

from metrics import dinuc_shuffle
from two_stage_core import optimize_all, naive_motif_detection


# ----------------------------------------------------------------------
# Region arithmetic
# ----------------------------------------------------------------------

def merge_regions(regions):
    """Sorted, disjoint regions — overlapping and touching spans merged."""
    out = []
    for s, e in sorted((int(s), int(e)) for s, e in regions if e > s):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def total_bp(regions):
    """Total bp covered, double-counting nothing."""
    return sum(e - s for s, e in merge_regions(regions))


def _subtract(intervals, hole):
    """Remove `hole` from a list of intervals."""
    hs, he = hole
    out = []
    for s, e in intervals:
        if he <= s or hs >= e:
            out.append((s, e))
            continue
        if s < hs:
            out.append((s, hs))
        if he < e:
            out.append((he, e))
    return [(s, e) for s, e in out if e > s]


def _halves(s, e, min_width):
    """The three overlapping 'halves' of [s, e): left, right, and middle.

    The middle half exists so that a feature straddling the midpoint is not
    cut in two by every split — a clean bisection alone would never see it.
    """
    w = e - s
    if w <= min_width:
        return []
    half = max(min_width, w // 2)
    q = (w - half) // 2
    out = []
    for a, b in [(s, s + half), (e - half, e), (s + q, s + q + half)]:
        a, b = max(s, a), min(e, b)
        if b - a >= min_width and (a, b) != (s, e) and (a, b) not in out:
            out.append((a, b))
    return out


def _overlaps(region, regions):
    s, e = region
    return any(s < b and e > a for a, b in regions)


# ----------------------------------------------------------------------
# Shared shuffled backgrounds
# ----------------------------------------------------------------------

class ShuffleContext:
    """One fixed set of dinuc-shuffled backgrounds for a (sequence, head).

    Every knock-in and knock-out below is scored against this same background
    set, so scores from anywhere in the pipeline are directly comparable, and
    many candidate region sets can be evaluated in a single forward pass.

    Backgrounds are drawn with the same `np.random.default_rng(seed)` call
    order as virtual_experimentalist's necessity_test / joint_sufficiency_test,
    so for a given seed the scores here match those functions exactly.

    knock_in(regions)  -> WT `regions` implanted into shuffled backgrounds
                          suff     = mean(implanted - shuffled)
                          recovery = suff / (WT - mean shuffled)
    knock_out(regions) -> shuffled background spliced into WT at `regions`
                          nec      = mean(WT - chimeric)
    """

    MAX_ROWS = 4000  # cap on sequences per forward pass

    def __init__(self, model, sequence, head_index=0, num_shuffles=100, seed=42):
        self.model = model
        self.sequence = sequence
        self.head_index = head_index
        self.L = sequence.shape[0]
        self.num_shuffles = num_shuffles
        self.seed = seed

        rng = np.random.default_rng(seed)
        self.bg = np.array([dinuc_shuffle(sequence, rng)
                            for _ in range(num_shuffles)])
        self.shuffled_preds = self._predict(self.bg)
        self.mean_shuffled = float(self.shuffled_preds.mean())
        self.wt_pred = float(self._predict(sequence[np.newaxis])[0])
        self.wt_signal = self.wt_pred - self.mean_shuffled

        self.n_experiments = 0  # distinct region sets actually run
        self._memo = {"ki": {}, "ko": {}}

    def _predict(self, x):
        return self.model.predict(x, verbose=0)[self.head_index].flatten()

    def _build(self, regions, mode):
        if mode == "ki":
            x = self.bg.copy()
            for s, e in regions:
                x[:, s:e, :] = self.sequence[np.newaxis, s:e, :]
        else:
            x = np.tile(self.sequence[np.newaxis], (self.num_shuffles, 1, 1))
            for s, e in regions:
                x[:, s:e, :] = self.bg[:, s:e, :]
        return x

    def _score(self, region_sets, mode):
        """Run every region set in as few forward passes as possible."""
        keys = [tuple(merge_regions(r)) for r in region_sets]
        memo = self._memo[mode]
        todo = [k for k in dict.fromkeys(keys) if k not in memo]

        per_pass = max(1, self.MAX_ROWS // self.num_shuffles)
        for i in range(0, len(todo), per_pass):
            chunk = todo[i:i + per_pass]
            batch = np.concatenate([self._build(k, mode) for k in chunk])
            preds = self._predict(batch).reshape(len(chunk), self.num_shuffles)
            for k, p in zip(chunk, preds):
                bp = sum(e - s for s, e in k)
                if mode == "ki":
                    suff = float(np.mean(p - self.shuffled_preds))
                    memo[k] = {
                        "regions": list(k), "total_bp": bp, "suff": suff,
                        "recovery": suff / self.wt_signal if abs(self.wt_signal) > 1e-9 else 0.0,
                        "mean_pred": float(p.mean()),
                    }
                else:
                    nec = float(np.mean(self.wt_pred - p))
                    memo[k] = {
                        "regions": list(k), "total_bp": bp, "nec": nec,
                        "nec_per_bp": nec / bp if bp else 0.0,
                        "mean_pred": float(p.mean()),
                    }
            self.n_experiments += len(chunk)

        return [memo[k] for k in keys]

    def knock_in(self, regions):
        return self._score([regions], "ki")[0]

    def knock_in_many(self, region_sets):
        return self._score(region_sets, "ki")

    def knock_out(self, regions):
        return self._score([regions], "ko")[0]

    def knock_out_many(self, region_sets):
        return self._score(region_sets, "ko")


# ----------------------------------------------------------------------
# Stage 1 — necessity boundaries + spurious-call pruning
# ----------------------------------------------------------------------

def _null_necessity(ctx, width, occupied, n_null, rng):
    """Necessity of random same-width windows that miss every called motif.

    Gives each call a length-matched noise floor, so "is this region real?"
    is answered against what an arbitrary window of the same size scores.
    """
    free = [s for s in range(0, ctx.L - width + 1)
            if not _overlaps((s, s + width), occupied)]
    if len(free) < n_null:
        free = list(range(0, ctx.L - width + 1))
    starts = rng.choice(free, size=min(n_null, len(free)), replace=False)
    scores = np.array([r["nec"] for r in
                       ctx.knock_out_many([[(int(s), int(s) + width)] for s in starts])])
    return float(scores.mean()), float(scores.std()), scores


def stage1_necessity(model, sequence, attribution_map, ctx, num_std=1.5,
                     min_len=1, max_expand=15, n_null=16, null_z=2.0,
                     verbose=True):
    """Naive cutoff -> expand-only necessity boundaries -> drop spurious calls.

    Returns dict with 'naive', 'motifs' (kept), 'pruned' (dropped, with the
    null they failed against), and 'null_by_width'.
    """
    naive = naive_motif_detection(attribution_map, sequence,
                                  num_std=num_std, min_len=min_len)
    if verbose:
        print(f"  naive cutoff (mean + {num_std} std of CWM): {len(naive)} calls")

    if not naive:
        return {"naive": [], "motifs": [], "pruned": [], "null_by_width": {}}

    opt = optimize_all(model, sequence, naive, head_index=ctx.head_index,
                       num_shuffles=ctx.num_shuffles, seed=ctx.seed,
                       verbose=False, max_expand=max_expand)
    if verbose:
        n_merged = len(naive) - len(opt)
        print(f"  expand-only necessity optimization: {len(opt)} motifs"
              + (f" ({n_merged} absorbed by merge)" if n_merged else ""))

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

    if verbose:
        print(f"  length-matched null (n={n_null} random windows per width, "
              f"keep z >= {null_z}): {len(motifs)} kept, {len(pruned)} pruned "
              f"as spurious")

    return {"naive": naive, "motifs": motifs, "pruned": pruned,
            "null_by_width": null_by_width}


# ----------------------------------------------------------------------
# Stage 2a — is it just that the boundaries stopped too early?
# ----------------------------------------------------------------------

def _finalize_motifs(ctx, bounds, source_motifs):
    """Collapse possibly-overlapping bounds into final motifs, rescored."""
    merged = merge_regions([tuple(b) for b in bounds])
    out = []
    for s, e in merged:
        members = [m for m, b in zip(source_motifs, bounds)
                   if b[0] < e and b[1] > s]
        ko = ctx.knock_out([(s, e)])
        rec = dict(members[0]) if members else {}
        rec.update({
            "start": s, "end": e, "opt_start": s, "opt_end": e,
            "naive_start": min(m["naive_start"] for m in members),
            "naive_end": max(m["naive_end"] for m in members),
            "naive_indices": sorted({i for m in members
                                     for i in m.get("naive_indices", [])}),
            "stage1_start": min(m["stage1_start"] for m in members),
            "stage1_end": max(m["stage1_end"] for m in members),
            "merged": len(members) > 1 or any(m.get("merged") for m in members),
            "absorbed_buffered": any(m.get("absorbed_buffered") for m in members),
            "nec": ko["nec"], "nec_per_bp": ko["nec_per_bp"],
            "suff": ctx.knock_in([(s, e)])["suff"],
        })
        rec["n_extended"] = (e - s) - (rec["stage1_end"] - rec["stage1_start"])
        out.append(rec)
    return out


def _absorb_adjacent(ctx, motifs, buffered, gap=2, verbose=True):
    """Fold buffered regions that landed against a motif into that motif.

    A buffered region touching a motif is not a separate element — it is that
    motif's flank, which stage 1 could not see because knocking it out changes
    nothing. Reporting it as its own feature would be misleading, so it is
    merged in and the motif is rescored on the full span.
    """
    if not motifs or not buffered:
        return motifs, buffered

    bounds = [[m["start"], m["end"]] for m in motifs]
    kept = []
    for b in buffered:
        hit = next((i for i, (s, e) in enumerate(bounds)
                    if b["start"] - gap <= e and s - gap <= b["end"]), None)
        if hit is None:
            kept.append(b)
            continue
        bounds[hit] = [min(bounds[hit][0], b["start"]),
                       max(bounds[hit][1], b["end"])]
        motifs[hit]["absorbed_buffered"] = True
        if verbose:
            print(f"    [{b['start']},{b['end']}) sits against a motif — "
                  f"absorbed as its flank -> "
                  f"[{bounds[hit][0]},{bounds[hit][1]})")

    return _finalize_motifs(ctx, bounds, motifs), kept


EXTEND_JUMPS = (1, 2, 3, 5, 8)


def stage2_extend(ctx, motifs, target=0.90, max_extend=20, verbose=True):
    """Optimize the motif boundaries by SUFFICIENCY.

    Stage 1 stopped each motif where its OWN necessity stopped improving. This
    is the only step that moves those boundaries by sufficiency: every
    candidate step is scored by the joint knock-in of all motifs implanted
    together, and only strict improvements are accepted.

    Steps of 1bp are tried first. A flank can sit behind a dead bp or two,
    though, and a strict 1bp rule stalls in front of that gap and leaves the
    flank to be rediscovered later as a separate 'buffered' region. So when no
    1bp step improves, progressively longer jumps are tried (EXTEND_JUMPS) and
    the one with the best gain per bp is taken — the set only jumps when it is
    otherwise stuck, so boundaries stay tight.
    """
    bounds = [[m["start"], m["end"]] for m in motifs]
    before = ctx.knock_in([tuple(b) for b in bounds])["recovery"]
    cur = before
    trace = [{"step": 0, "motif": None, "side": None, "bp": 0,
              "regions": [tuple(b) for b in bounds], "recovery": cur}]
    added = 0

    while added < max_extend and cur < target:
        stepped = False
        for jump in EXTEND_JUMPS:
            if jump > max_extend - added:
                break
            moves, sets = [], []
            for i, (s, e) in enumerate(bounds):
                for side, (ns, ne) in (("L", (s - jump, e)), ("R", (s, e + jump))):
                    if ns < 0 or ne > ctx.L:
                        continue
                    trial = [tuple(b) for b in bounds]
                    trial[i] = (ns, ne)
                    moves.append((i, side, ns, ne))
                    sets.append(trial)
            if not moves:
                continue

            recs = [r["recovery"] for r in ctx.knock_in_many(sets)]
            best = int(np.argmax([(r - cur) / jump for r in recs]))
            if recs[best] <= cur + 1e-6:
                continue  # this jump size is stuck too — try a longer one

            i, side, ns, ne = moves[best]
            bounds[i] = [ns, ne]
            cur = recs[best]
            added += jump
            stepped = True
            trace.append({"step": added, "motif": i, "side": side, "bp": jump,
                          "regions": [tuple(b) for b in bounds], "recovery": cur})
            break
        if not stepped:
            break

    final = _finalize_motifs(ctx, bounds, motifs) if motifs else []

    if verbose:
        print(f"  recovery before extension: {before:6.2%}")
        print(f"  recovery after  extension: {cur:6.2%}  "
              f"(+{added}bp across the set, budget {max_extend})")

    return {"motifs": final, "recovery_before": before, "recovery_after": cur,
            "n_added": added, "reached_target": cur >= target, "trace": trace}


# ----------------------------------------------------------------------
# Stage 2b — buffered-feature search
# ----------------------------------------------------------------------

def _resolve(ctx, territory, fixed, target, min_width, depth, max_depth,
             split_budget, trace, verbose):
    """Smallest set of regions inside `territory` that, with `fixed` knocked
    in alongside, reaches `target`.

    Precondition: `fixed` + all of `territory` already reaches target — true at
    the top level because knocking in the whole sequence just rebuilds WT.

    Each call knocks in the three overlapping halves of the territory. If one
    half is enough, descend into it and split again. If none is enough the
    feature is split across halves, so hold the best half whole, resolve what
    is left of the territory against it, then come back and minimize the held
    half now that the other piece is known.
    """
    territory = [t for t in territory if t[1] > t[0]]
    if not territory:
        return []
    if ctx.knock_in(fixed)["recovery"] >= target:
        trace.append({"phase": "drop", "depth": depth, "territory": territory})
        return []

    cands = []
    for a, b in territory:
        cands.extend(_halves(a, b, min_width))
    cands = list(dict.fromkeys(cands))
    if not cands or depth >= max_depth:
        return list(territory)

    recs = [r["recovery"] for r in ctx.knock_in_many([fixed + [c] for c in cands])]
    best = int(np.argmax(recs))
    chosen, chosen_rec = cands[best], recs[best]
    met = chosen_rec >= target
    trace.append({"phase": "split", "depth": depth, "territory": list(territory),
                  "tested": list(zip(cands, recs)), "chosen": chosen,
                  "recovery": chosen_rec, "met": met})
    if verbose:
        print(f"    {'  ' * depth}half [{chosen[0]:3d},{chosen[1]:3d}) "
              f"-> {chosen_rec:6.2%}"
              + ("  enough — splitting it again" if met else
                 "  not enough — feature is split, holding this half and searching the rest"))

    if met:
        return _resolve(ctx, [chosen], fixed, target, min_width, depth + 1,
                        max_depth, split_budget, trace, verbose)

    rest = []
    for t in territory:
        rest.extend(_subtract([t], chosen))
    if not rest or split_budget <= 0:
        return list(territory)

    part_b = _resolve(ctx, rest, fixed + [chosen], target, min_width, depth + 1,
                      max_depth, split_budget - 1, trace, verbose)
    part_a = _resolve(ctx, [chosen], fixed + part_b, target, min_width,
                      depth + 1, max_depth, split_budget - 1, trace, verbose)
    return part_a + part_b


def _optimize_edges(ctx, region, fixed, target, budget, min_width, trace):
    """Stop halving and settle the boundaries 1bp at a time.

    Halving lands on a power-of-two grid, so the edges of what it returns are
    an artifact of where the splits happened to fall, not of where the feature
    actually is. This walks each edge:
        grow   — accept only if joint recovery STRICTLY increases, so a real
                 edge the splits cut through is recovered
        shrink — accept if recovery does not drop and the target still holds,
                 so dead bp the splits left on are given back
    Growth is preferred while it pays, shrinking breaks ties, and the target is
    never given up.
    """
    s, e = region
    cur = ctx.knock_in(fixed + [(s, e)])["recovery"]
    for _ in range(budget):
        cands = []
        if s - 1 >= 0:
            cands.append(("grow", (s - 1, e)))
        if e + 1 <= ctx.L:
            cands.append(("grow", (s, e + 1)))
        if e - s > min_width:
            cands.append(("shrink", (s + 1, e)))
            cands.append(("shrink", (s, e - 1)))

        recs = [r["recovery"] for r in
                ctx.knock_in_many([fixed + [c] for _, c in cands])]
        best = None
        for (kind, cand), rec in zip(cands, recs):
            ok = (rec > cur + 1e-6 if kind == "grow"
                  else rec >= cur - 1e-6 and rec >= target)
            if not ok:
                continue
            score = (rec, kind == "shrink")  # best recovery; shrink breaks ties
            if best is None or score > best[0]:
                best = (score, kind, cand, rec)
        if best is None:
            break
        _, kind, (s, e), cur = best
        trace.append({"phase": "edges", "kind": kind, "chosen": (s, e),
                      "recovery": cur})
    return (s, e)


def stage2_buffered_search(ctx, motif_regions, target=0.90, min_width=8,
                           max_depth=8, split_budget=4, edge_budget=20,
                           verbose=True):
    """Find the sequence the motifs were hiding.

    Recursively knocks in halves of the sequence alongside the motifs (see
    `_resolve`) until the target is met and the responsible subsequences are
    pruned down, then stops halving and settles each region's edges 1bp at a
    time with the others held fixed (see `_optimize_edges`), dropping any that
    turn out to be removable.
    """
    motif_regions = merge_regions(motif_regions)
    trace = []
    found = merge_regions(_resolve(ctx, [(0, ctx.L)], motif_regions, target,
                                   min_width, 0, max_depth, split_budget,
                                   trace, verbose))

    buffered = list(found)
    for i in range(len(buffered)):
        others = motif_regions + [r for j, r in enumerate(buffered)
                                  if j != i and r is not None]
        if ctx.knock_in(others)["recovery"] >= target:
            trace.append({"phase": "drop", "region": buffered[i]})
            buffered[i] = None
            continue
        buffered[i] = _optimize_edges(ctx, buffered[i], others, target,
                                      edge_budget, min_width, trace)
    buffered = [b for b in buffered if b is not None]
    buffered = merge_regions(buffered)  # edge growth may have joined two

    final_regions = merge_regions(motif_regions + buffered)
    final_rec = ctx.knock_in(final_regions)["recovery"]

    out = []
    for b in buffered:
        without = merge_regions(motif_regions + [r for r in buffered if r != b])
        ko = ctx.knock_out([b])
        ki = ctx.knock_in([b])
        out.append({
            "start": b[0], "end": b[1], "opt_start": b[0], "opt_end": b[1],
            "buffered": True,
            "contribution": final_rec - ctx.knock_in(without)["recovery"],
            "nec": ko["nec"], "suff": ki["suff"],
            "recovery_alone": ki["recovery"],
        })
    out.sort(key=lambda r: r["start"])

    if verbose and out:
        print(f"    pruned down to {len(out)} buffered region(s): "
              + ", ".join(f"[{r['start']},{r['end']})" for r in out))

    return {"buffered": out, "recovery": final_rec, "trace": trace,
            "reached_target": final_rec >= target}


# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------

def run_two_stage(model, sequence, attribution_map, head_index=0,
                  target_recovery=0.90, num_std=1.5, min_len=1,
                  num_shuffles=100, seed=42, max_expand=15, max_extend=20,
                  min_width=8, max_depth=8, split_budget=4, absorb_gap=2,
                  n_null=16, null_z=2.0, verbose=True):
    """Run stage 1 then stage 2 and return everything the results cell needs.

    Returns a dict with the final 'motifs' and 'buffered' regions, the
    'recovery' after each step, a 'status', the full experiment 'traces', and
    the ShuffleContext ('ctx') so other annotation methods can be scored
    against the very same backgrounds.
    """
    ctx = ShuffleContext(model, sequence, head_index=head_index,
                         num_shuffles=num_shuffles, seed=seed)

    if verbose:
        print("=" * 78)
        print(f"  WT prediction {ctx.wt_pred:.4f} | shuffled baseline "
              f"{ctx.mean_shuffled:.4f} | WT signal {ctx.wt_signal:.4f}")
        print(f"  target recovery {target_recovery:.0%} of WT signal, "
              f"{num_shuffles} dinuc shuffles")
        print("=" * 78)
        print("\nSTAGE 1 — necessity boundaries")

    s1 = stage1_necessity(model, sequence, attribution_map, ctx,
                          num_std=num_std, min_len=min_len,
                          max_expand=max_expand, n_null=n_null,
                          null_z=null_z, verbose=verbose)

    naive_regions = [(m["start"], m["end"]) for m in s1["naive"]]
    rec_naive = ctx.knock_in(naive_regions)["recovery"] if naive_regions else 0.0
    s1_regions = [(m["start"], m["end"]) for m in s1["motifs"]]
    rec_s1 = ctx.knock_in(s1_regions)["recovery"] if s1_regions else 0.0

    if verbose:
        print(f"\nSTAGE 2 — sufficiency  (knock the set into shuffled backgrounds)")
        print(f"  stage-1 set recovers {rec_s1:.2%} of WT signal "
              f"(target {target_recovery:.0%})")

    motifs = s1["motifs"]
    buffered, ext, buf = [], None, None
    rec_ext = rec_s1

    if rec_s1 >= target_recovery:
        status = "captured_at_stage1"
        if verbose:
            print("  target already met — the necessary set is also sufficient.")
    else:
        if verbose:
            print("\n  2a — is it just boundaries? extending for joint sufficiency")
        ext = stage2_extend(ctx, motifs, target=target_recovery,
                            max_extend=max_extend, verbose=verbose)
        motifs = ext["motifs"]
        rec_ext = ext["recovery_after"]

        if ext["reached_target"]:
            status = "captured_by_extension"
            if verbose:
                print("  target met by extension — stage 1 had stopped too early.")
        else:
            if verbose:
                print("\n  2b — buffered features: knocking in halves of the sequence")
                print("       (these regions are invisible to knockout/ISM because")
                print("        the main motifs carry the output without them)")
            buf = stage2_buffered_search(
                ctx, [(m["start"], m["end"]) for m in motifs],
                target=target_recovery, min_width=min_width,
                max_depth=max_depth, split_budget=split_budget,
                verbose=verbose)
            motifs, buffered = _absorb_adjacent(ctx, motifs, buf["buffered"],
                                                gap=absorb_gap, verbose=verbose)
            status = ("captured_by_buffered" if buf["reached_target"]
                      else "incomplete")

    final_regions = merge_regions([(m["start"], m["end"]) for m in motifs]
                                  + [(b["start"], b["end"]) for b in buffered])
    rec_final = ctx.knock_in(final_regions)["recovery"] if final_regions else 0.0

    # Each stage's verdict is on the set IT was holding; the final set is
    # re-scored after absorption and merging, so the number actually reported
    # is the one that decides the status. Without this a run can say
    # "captured by buffered" next to a final recovery under the target.
    if rec_final < target_recovery:
        status = "incomplete"
    elif status == "incomplete":
        status = "captured_by_buffered"

    if verbose:
        print("\n" + "=" * 78)
        print(f"  {status.replace('_', ' ').upper()}")
        print(f"  naive {rec_naive:6.2%} -> stage1 {rec_s1:6.2%} -> "
              f"extended {rec_ext:6.2%} -> final {rec_final:6.2%}   "
              f"({total_bp(final_regions)}bp annotated, "
              f"{ctx.n_experiments} virtual experiments)")
        print("=" * 78)

    return {
        "status": status,
        "head_index": head_index,
        "target_recovery": target_recovery,
        "sequence_length": ctx.L,
        "wt_pred": ctx.wt_pred,
        "mean_shuffled": ctx.mean_shuffled,
        "wt_signal": ctx.wt_signal,
        "naive": s1["naive"],
        "pruned": s1["pruned"],
        "stage1_motifs": s1["motifs"],
        "motifs": motifs,
        "buffered": buffered,
        "regions": final_regions,
        "recovery": {"naive": rec_naive, "stage1": rec_s1,
                     "extended": rec_ext, "final": rec_final},
        "traces": {"extend": ext["trace"] if ext else [],
                   "buffered": buf["trace"] if buf else []},
        "n_experiments": ctx.n_experiments,
        "ctx": ctx,
    }
