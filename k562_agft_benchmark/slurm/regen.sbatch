#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --qos=bio_ai
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=k562_regen
#SBATCH --output=/grid/koo/home/pmantill/projects/AutoSituate/k562_agft_benchmark/slurm/logs/%x_%j.out
#SBATCH --error=/grid/koo/home/pmantill/projects/AutoSituate/k562_agft_benchmark/slurm/logs/%x_%j.err
#
# Full data + FIGURE regeneration (one job remakes everything, no manual steps):
#   1. run_benchmark.py            -> per_sequence.csv / per_method.csv (incl. null significance)
#   2. visualize.py all --refresh  -> annotations.pkl + all 84 examples + summaries + significance
#   3. swap/run_swap.py            -> swap_scores_by_method.npz (reads the refreshed cache)
#   4. swap/make_swap_figures.py   -> every swap figure
# Forward passes only, same K562 encoder env as run_benchmark.sbatch / run_swap.sbatch.

DIR="/grid/koo/home/pmantill/projects/AutoSituate/k562_agft_benchmark"
module purge
module load EB5
module load CUDA/12.9.1
source /grid/koo/home/pmantill/projects/Virtual_Experiments/Hippo_axis/Hippo_dependency_mpra/Hippo_agft_venv/bin/activate
export PYTHONUNBUFFERED=1

echo "K562 regen | Job $SLURM_JOB_ID | GPU $CUDA_VISIBLE_DEVICES | $(date)"
cd "$DIR"
uv run python run_benchmark.py
uv run python visualize.py all --refresh       # all 84 examples + summaries + significance
uv run python swap/run_swap.py
uv run python swap/make_swap_figures.py        # every swap figure
echo "Done: $(date)"
