"""The six annotation methods, each applied systematically across all sequences.

Every method maps one sequence's foreground to a set of regions (start, end) in
0..200 variable-region coordinates. Seqlet callers and discovery are run POOLED
across all sequences (so their nulls/clustering see the whole panel), then split
back per sequence.

    recursive_seqlets     tangermeme, default threshold=0.01
    tfmodisco_seqlets     tangermeme's TF-MoDISco-style caller
    modiscolite_seqlets   real TF-MoDISco (modiscolite) seqlet caller
    modisco_hits          TF-MoDISco discovery -> CWM hit-calling on the foreground
                          (cosine scan of each pattern CWM vs the contribution map)
    naive                 mean +/- N*std threshold on the projected track
                          (two-sided by default, so repressive motifs are kept)
    jaspar_fimo           FIMO of JASPAR2026 motifs onto the one-hot (default)

Projected track = (foreground * WT one-hot) summed over the 4 channels, i.e. the
attribution of the actual base at each position. The SEAM maps are mean-centred,
so a plain channel-sum would be ~0; weighting by the one-hot recovers the signal.
"""

import warnings

import numpy as np
import torch

import config as C

warnings.filterwarnings("ignore")


# --- shared helpers ---------------------------------------------------------
def projected_tracks(recs):
    """(N, 200) per-position contribution at the WT base."""
    return np.stack([(r["fg"] * r["var"]).sum(axis=1) for r in recs]).astype(np.float32)


def onehot_stack_cl(recs):
    """(N, 4, 200) channels-first one-hot (memelite/FIMO convention)."""
    return np.stack([r["var"].T for r in recs]).astype(np.float32)


def _empty(n):
    return [[] for _ in range(n)]


def _group_df(df, n):
    """DataFrame with example_idx/start/end -> list[n] of (start, end)."""
    out = _empty(n)
    for row in df.itertuples(index=False):
        i, s, e = int(row.example_idx), int(row.start), int(row.end)
        if 0 <= i < n and e > s:
            out[i].append((s, e))
    return out


# --- M1 recursive_seqlets ---------------------------------------------------
def m_recursive(recs, tracks):
    from tangermeme.seqlet import recursive_seqlets
    # recursive_seqlets is positive-only by design (see its docstring); call it on
    # |track| so repressor spans (negative attribution) are caught too. We only use
    # the boundaries downstream, so no signed-sum re-extraction is needed.
    df = recursive_seqlets(np.abs(tracks), threshold=0.01)
    return _group_df(df, len(recs))


# --- M2 tangermeme tfmodisco_seqlets ---------------------------------------
def m_tfmodisco_seqlets(recs, tracks):
    from tangermeme.seqlet import tfmodisco_seqlets
    df = tfmodisco_seqlets(torch.from_numpy(tracks))
    return _group_df(df, len(recs))


# --- M3 modiscolite real seqlet caller --------------------------------------
def m_modiscolite_seqlets(recs, tracks):
    from modiscolite.extract_seqlets import extract_seqlets
    # TF-MoDISco default parameterisation (mirrors modiscolite.tfmodisco.TFMoDISco).
    win, flank = 21, 10
    seqlets, _ = extract_seqlets(
        tracks, window_size=win, flank=flank,
        suppress=int(0.5 * win) + flank,
        target_fdr=0.2, min_passing_windows_frac=0.03,
        max_passing_windows_frac=0.2, weak_threshold_for_counting_sign=0.8,
    )
    out = _empty(len(recs))
    for sq in seqlets:
        i, s, e = int(sq.example_idx), int(sq.start), int(sq.end)
        if 0 <= i < len(recs) and e > s:
            out[i].append((s, e))
    return out


# --- modisco discovery -> hit-calling (pattern member seqlets) --------------
def discover_patterns(recs, min_cluster=100):
    """Pooled TF-MoDISco discovery. Returns the list of discovered patterns
    (pos + neg SeqletSets).

    `min_cluster` is the minimum-seqlets threshold, applied to BOTH
    min_metacluster_size and final_min_cluster_size (on this ~83-seqlet panel the
    metacluster gate alone is inert — 83 clears any value <83 identically — so the
    per-pattern final cluster size is the knob that actually changes the output)."""
    from modiscolite.tfmodisco import TFMoDISco
    one_hot = np.stack([r["var"] for r in recs]).astype(np.float32)          # (N,200,4)
    hyp = np.stack([r["fg"] for r in recs]).astype(np.float32)               # (N,200,4)
    pos, neg = TFMoDISco(one_hot=one_hot, hypothetical_contribs=hyp,
                         min_metacluster_size=min_cluster,
                         final_min_cluster_size=min_cluster)
    return list(pos or []) + list(neg or [])


def m_modisco_hits(recs, patterns):
    """modisco hit-calling: each discovered pattern's member seqlets are its
    instances on the contribution scores (no threshold, default behaviour)."""
    out = _empty(len(recs))
    for pattern in (patterns or []):
        for sq in pattern.seqlets:
            i, s, e = int(sq.example_idx), int(sq.start), int(sq.end)
            if 0 <= i < len(recs) and e > s:
                out[i].append((s, e))
    return out


# --- M5 naive threshold -----------------------------------------------------
def m_naive(recs, tracks, num_std=2.0, two_sided=True):
    """mean +/- N*std threshold on the per-sequence projected track.

    two_sided=True keeps both tails (|t - mean| > N*std), so strong NEGATIVE
    contributions (repressive motifs, where wt_signal is negative) are captured
    too -- matching the |track| convention of the seqlet callers. two_sided=False
    is the original upper-tail-only cut (t > mean + N*std)."""
    out = []
    for t in tracks:
        dev = num_std * t.std()
        mask = np.abs(t - t.mean()) > dev if two_sided else t > (t.mean() + dev)
        out.append(_runs(mask))
    return out


def _runs(mask):
    regs, i, L = [], 0, len(mask)
    while i < L:
        if mask[i]:
            j = i
            while j < L and mask[j]:
                j += 1
            regs.append((i, j))
            i = j
        else:
            i += 1
    return regs


# --- M6 JASPAR FIMO ---------------------------------------------------------
def m_jaspar_fimo(recs):
    return _fimo_regions(str(C.JASPAR_MEME), recs)


def _fimo_regions(motifs, recs):
    """Run memelite FIMO (default settings) and collect hit spans per sequence."""
    from memelite import fimo
    seqs = onehot_stack_cl(recs)                     # (N, 4, 200)
    dfs = fimo(motifs, seqs)                          # list, one DataFrame per motif
    out = _empty(len(recs))
    for df in dfs:
        if len(df) == 0:
            continue
        for row in df.itertuples(index=False):
            i, s, e = int(row.sequence_name), int(row.start), int(row.end)
            if 0 <= i < len(recs) and e > s:
                out[i].append((s, e))
    return out
