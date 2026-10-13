#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --qos=bio_ai
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=k562_agft_bench
#SBATCH --output=/grid/koo/home/pmantill/projects/AutoSituate/k562_agft_benchmark/slurm/logs/%x_%j.out
#SBATCH --error=/grid/koo/home/pmantill/projects/AutoSituate/k562_agft_benchmark/slurm/logs/%x_%j.err
#
# K562 AlphaGenome-encoder annotation benchmark: 6 annotation methods x 84 SEAM
# foregrounds, scored by motif knock-out (necessity) + knock-in (sufficiency).
# Forward passes only (no ISM). Mirrors SEAM_ism.py's model env.

DIR="/grid/koo/home/pmantill/projects/AutoSituate/k562_agft_benchmark"
module purge
module load EB5
module load CUDA/12.9.1
source /grid/koo/home/pmantill/projects/Virtual_Experiments/Hippo_axis/Hippo_dependency_mpra/Hippo_agft_venv/bin/activate
export PYTHONUNBUFFERED=1

echo "K562 AGFT benchmark | Job $SLURM_JOB_ID | GPU $CUDA_VISIBLE_DEVICES | $(date)"
cd "$DIR"
uv run python run_benchmark.py
uv run python visualize.py all --refresh      # all 84 examples + summaries + significance
echo "Done: $(date)"
