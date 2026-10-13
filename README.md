# AutoSituate

Automated motif annotation for regulatory sequences. Given a sequence and its
attribution map, AutoSituate localizes the functional elements (motif core +
context) and **validates them against the model** by knock-out / knock-in, rather
than trusting a static threshold or a motif database. The two-stage annotator
grows a necessity-defined core, extends it to sufficiency, then searches for
buffered context features.

## Headline results

Benchmarked on 84 K562 single-sequence foregrounds, scored on the finetuned K562
AlphaGenome encoder. Every method run at default settings; metrics are medians
over the panel. **Recovery** = fraction of WT activity restored by knocking the
called positions back in; **footprint** = called bp; **rec/bp** = recovery per
called base (compactness). Details: [`k562_agft_benchmark/`](k562_agft_benchmark/).

| Method | Seqs annotated | Recovery | Footprint (bp) | Recovery / bp |
|---|---|---|---|---|
| **AutoSituate (two-stage)** | **84 / 84** | **96.5%** | **56** | **0.0194** |
| FIMO (JASPAR2026) | 84 / 84 | 90.5% | 159.5 | 0.0055 |
| TF-MoDISco-lite seqlets | 57 / 84 | 69.0% | 67 | 0.0087 |
| TF-MoDISco seqlets | 54 / 84 | 65.9% | 64.5 | 0.0088 |
| MoDISco hits | 30 / 84 | 42.6% | 50 | 0.0067 |
| recursive seqlets | 31 / 84 | 17.5% | 17 | 0.0104 |
| naive (mean+2σ) | 84 / 84 | 4.7% | 10 | 0.0053 |

**AutoSituate annotates every sequence and recovers ~96% of WT activity in ~56 bp**
— matching FIMO's recovery in roughly a third of the footprint (~3.5× more
efficient per base), and far ahead of MoDISco-based callers on both coverage and
recovery.

## Layout

- [`k562_agft_benchmark/`](k562_agft_benchmark/) — head-to-head benchmark of the
  annotation methods (the table above).
- [`step1_explore/`](step1_explore/) — does the two-stage Stage-1 seeder limit
  what the pipeline finds? Swaps only Stage 1, holds the rest fixed.
- `algo_impliment/` — original algorithm implementation.
