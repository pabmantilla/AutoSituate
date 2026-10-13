#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --qos=bio_ai
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=k562_two_stage_ism
#SBATCH --output=/grid/koo/home/pmantill/projects/AutoSituate/k562_agft_benchmark/slurm/logs/%x_%j.out
#SBATCH --error=/grid/koo/home/pmantill/projects/AutoSituate/k562_agft_benchmark/slurm/logs/%x_%j.err
#
# AutoSituate two-stage annotator (OUR method) seeded from the RAW ISM map
# (wt_attribution.npy) instead of the SEAM foreground — a SEPARATE method
# (two_stage_ism) so it compares head-to-head with the foreground run, which is
# left untouched. Writes only results/two_stage_ism/. Then regenerates every
# figure (both variants now present) WITHOUT --refresh so annotations.pkl stays.

DIR="/grid/koo/home/pmantill/projects/AutoSituate/k562_agft_benchmark"
module purge
module load EB5
module load CUDA/12.9.1
source /grid/koo/home/pmantill/projects/Virtual_Experiments/Hippo_axis/Hippo_dependency_mpra/Hippo_agft_venv/bin/activate
export PYTHONUNBUFFERED=1
export TS_ATTR=ism

echo "K562 two-stage on raw ISM (ours) | Job $SLURM_JOB_ID | GPU $CUDA_VISIBLE_DEVICES | $(date)"
cd "$DIR"
uv run python run_two_stage.py
uv run python visualize.py all              # examples + significance-examples + summaries (both variants)
uv run python efficiency_summary.py
echo "Done: $(date)"
