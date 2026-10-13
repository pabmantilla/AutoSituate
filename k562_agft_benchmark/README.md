# K562 AGFT annotation benchmark

Applies AutoSituate's **basic** annotation methods, systematically, over the
existing SEAM_validate single-sequence foregrounds, scored on the finetuned
**K562 AlphaGenome encoder-only** model. No custom two-stage algorithm here — just
the standard callers, compared head-to-head by motif knock-out / knock-in.

## Inputs
- **Foregrounds**: `SEAM_validate/results/single_seq/foregrounds/*/foreground_scaled.npy`
  (the SEAM denoised foreground, `(200,4)`; central 200 bp `[15:215]` of the 230 bp
  insert). 84 of 85 dirs used — `rs536864738_WT` has no WT sequence and is skipped.
- **WT sequences**: `seam_seeds.csv` ∪ `seam_seeds_crepool.csv`.
- **Model**: `EncoderMPRAModel` @ `mpra_K562/finetuned_encoder.pt`; each insert is
  padded with the 51 bp construct suffix → 281 bp before scoring (same recipe as
  `SEAM_ism.py`).
- **JASPAR**: `eigen-interactions/motif_db/JASPAR2026_vertebrates.meme` (1019 motifs).

## Annotation methods (`annotate.py`), ordered least → most computationally heavy
| method | notes |
|--------|-------|
| `naive` | mean + 2·std threshold on the CWM (one-sided, positive tail) |
| `recursive_seqlets` | tangermeme, default `threshold=0.01`, pooled |
| `tfmodisco_seqlets` | tangermeme's TF-MoDISco-style caller, pooled |
| `modiscolite_seqlets` | **real** TF-MoDISco (modiscolite) seqlet caller, pooled |
| `jaspar_fimo` | FIMO (memelite) of JASPAR2026 motifs onto the **one-hot**, default settings |
| `modisco_hits` | TF-MoDISco **discovery** (default settings) → each discovered pattern's **member seqlets** are its hits. NB: on this 84-seq panel default TF-MoDISco discovers 0 patterns (needs >100 seqlets), so this row is empty. |

Projected track = `(foreground * WT one-hot).sum(channels)` — the SEAM maps are
mean-centred, so a plain channel-sum is ~0; the one-hot weighting recovers signal.

## Metrics (`metrics.py`) — no ISM
Dinuc-shuffle the whole 200 bp variable region once per replicate (N=100, shared
across all methods and both tests):
- **KO / necessity**: WT with the called positions overwritten by the shuffle →
  `necessity = mean(WT − KO)`; `nec_fraction = necessity / (WT − shuffled_mean)`.
- **KI / sufficiency**: shuffle with the called positions overwritten by WT →
  `sufficiency = mean(KI − shuffled)`; `recovery = sufficiency / (WT − shuffled_mean)`.

## Run
```bash
sbatch slurm/run_benchmark.sbatch     # gpuq / bio_ai / 1×H100
```
Outputs land in `results/`: `per_sequence.csv`, `per_method.csv`, `run_meta.txt`.

## Files
```
config.py          paths, geometry, KO/KI constants
data.py            load foregrounds + WT seqs + activity tertile bins
model.py           K562 EncoderMPRAModel wrapper (predict on the 200 bp window)
metrics.py         dinuc shuffle, ShuffleContext, joint KO / KI
annotate.py        the 6 methods
run_benchmark.py   driver
slurm/             run_benchmark.sbatch + logs/
```

## Notes / choices to flag
- "Whole-sequence shuffle" = the 200 bp variable region; 15 bp adapters + 51 bp
  construct suffix are held at WT (matches how SEAM mutagenised).
- `naive` and `recursive_seqlets` are **positive-tail only**, so they call little on
  repressive sequences by construction.
- Activity bins use fixed cutoffs in the empty gaps of the `true` distribution
  (`-0.7 / +0.7` → ~23 repressive / 21 neutral / 40 activating), not even thirds.
- `jaspar_fimo` scans the **one-hot sequence** (FIMO uses base identity only); the
  model-derived functional judgement comes from the downstream KO/KI. `modisco_hits`
  uses TF-MoDISco at **default settings**, so its hits are the member seqlets of the
  discovered patterns (called on the contribution scores). All methods are run at
  default settings — no tuned thresholds.
