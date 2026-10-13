"""Load the SEAM foregrounds + their WT sequences and attach an activity bin.

Each record is one foreground directory:
    id     : seq_id / directory name
    true   : measured MPRA activity (seam_seeds)   -- used for the activity bin
    pred   : model-predicted activity (seam_seeds)
    seq    : 230bp insert string
    left   : (15,4) left adapter one-hot  (held WT in all swaps)
    right  : (15,4) right adapter one-hot (held WT in all swaps)
    var    : (200,4) one-hot of the variable/attribution window [15:215]
    fg     : (200,4) foreground_scaled attribution map
"""

import csv
import os

import numpy as np

import config as C


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load_seeds():
    """seq_id -> {seq, true, pred}, union over the seed CSVs (first wins)."""
    seeds = {}
    for path in C.SEED_CSVS:
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            for row in csv.DictReader(fh):
                sid = row["seq_id"].rstrip(":")
                seeds.setdefault(sid, {
                    "seq": row["seq"],
                    "true": _f(row.get("true")),
                    "pred": _f(row.get("pred")),
                })
    return seeds


def onehot(seq):
    oh = np.zeros((len(seq), 4), dtype=np.float32)
    for i, b in enumerate(seq):
        j = C.ALPHA.get(b)
        if j is not None:
            oh[i, j] = 1.0
    return oh


def load_dataset():
    """Return the list of usable foreground records (84 expected)."""
    seeds = load_seeds()
    recs = []
    for d in sorted(os.listdir(C.FOREGROUNDS)):
        if d in C.SKIP:
            continue
        fdir = C.FOREGROUNDS / d
        if not fdir.is_dir() or not (fdir / C.FOREGROUND_MAP).exists():
            continue
        if d not in seeds:
            continue
        seq = seeds[d]["seq"]
        if len(seq) != C.ENH_LEN:
            continue
        oh = onehot(seq)
        recs.append({
            "id": d,
            "true": seeds[d]["true"],
            "pred": seeds[d]["pred"],
            "seq": seq,
            "left": oh[: C.VAR_START],
            "right": oh[C.VAR_END:],
            "var": oh[C.VAR_START: C.VAR_END],
            "fg": np.load(fdir / C.FOREGROUND_MAP).astype(np.float32),
        })
    return recs


REP_MAX = -0.7   # repressive:  true <= -0.7
ACT_MIN = 0.7    # activating:   true >= 0.7


def activity_bins(recs, key="true"):
    """Label each record repressive / neutral / activating.

    The panel was selected to sit in three well-separated activity clusters, so
    this uses fixed cutoffs that fall in the empty gaps rather than forcing even
    thirds. On `true` the clusters are:
        repressive  -1.34 .. -1.00   (gap)
        neutral     -0.42 .. +0.25   (gap)      <- includes the named neutral_* set
        activating  +1.20 .. +1.93
    so -0.7 / +0.7 separate them cleanly (counts are uneven: ~23 / 21 / 40).
    """
    bins = []
    for r in recs:
        v = r[key]
        if not np.isfinite(v):
            bins.append("unknown")
        elif v <= REP_MAX:
            bins.append("repressive")
        elif v >= ACT_MIN:
            bins.append("activating")
        else:
            bins.append("neutral")
    return bins, (REP_MAX, ACT_MIN)
