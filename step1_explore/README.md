# Stage-1 exploration

Does the two-stage annotator's **first step** (how it seeds candidate motifs)
limit what it finds? The rest of the pipeline only *expands and prunes* seeds —
necessity boundary growth, then sufficiency extension, then the buffered-feature
search — so anything the seed misses must be rescued by the (CWM-blind) buffered
search. Here we hold the whole algorithm fixed and swap **only Stage 1**, on the
**raw ISM maps** (`wt_attribution.npy`), scored on the K562 AlphaGenome encoder.

## The four seeders (`seeders.py`)
| variant | Stage-1 seed |
|---|---|
| `naive_pos` | mean + 1.5·std on the CWM, **positive tail only** — today's behaviour |
| `naive_abs` | mean ± 1.5·std (threshold \|CWM − mean\|) — two-sided, seeds **repressor** cores too |
| `adaptive_nec` | sweep the threshold strict→loose; keep the level with the best **necessity-per-bp** (model knock-out picks the depth) |
| `seqlets` | tangermeme **recursive seqlet** calling on \|CWM\| instead of a global cutoff |

Everything after the seed is byte-for-byte the original `two_stage_agent`
(see `stage1_variants.make_stage1` — only the one seed line differs; the driver
monkeypatches `two_stage_agent.stage1_necessity` and calls the unchanged
`run_two_stage`, so stages 2a/2b are untouched).

## Run
```bash
sbatch slurm/run_explore.sbatch         # gpuq / bio_ai / 1×H100, ~all four variants
# quick check first:
VARIANTS=naive_pos LIMIT=3 uv run python run_explore.py
```
Env knobs: `VARIANTS=naive_pos,seqlets` (subset), `LIMIT=5` (first N seqs).

## Outputs (`results/`)
- `<variant>/per_sequence.csv` — one row per sequence for that variant
- `<variant>/annotations.pkl` — regions + core/flank/context split + status
- `seed_comparison.csv` — **the main artifact**: tidy long table over every
  `(variant, sequence)` with seed count/bp, recovery at each stage
  (naive → stage1 → extended → **final**), final bp, **core vs context bp**,
  status, and #model experiments.
- `explore_summary.csv` — one row per variant: the medians that answer *does a
  better Stage 1 localise more (higher `core_bp`, lower `frac_zero_core`) and
  lean less on the buffered search?*

`analysis.ipynb` reads the two summary CSVs and draws the comparison figures
(run it after the job finishes — no GPU needed).

## What to look for
Baseline finding this explores: with `naive_pos` the seed recovers ~5% of WT and
the final ~96% is produced almost entirely by the buffered search (on repressive
sequences Stage-1 core bp ≈ 0). A good Stage 1 should move recovery into
`rec_stage1`, raise `core_bp`, drop `frac_zero_core`, and ideally cut
`med_n_experiments` (less brute-force searching).

## Notes
- Scored with the pipeline's own `ShuffleContext` (same `SEED`/`NUM_SHUFFLES`
  as the benchmark), so `rec_final` is comparable to `results/two_stage`. The
  heavy window-matched null is **not** recomputed here (not needed to rank seeds).
- `adaptive_nec` chooses depth by necessity-per-bp over the std grid
  `(3.0 … 0.0)`; its threshold-sweep knock-outs are counted in `n_experiments`.
  This stopping rule is a choice — easy to change in `seeders.ADAPTIVE_STD_GRID`
  / the `argmax` criterion.
