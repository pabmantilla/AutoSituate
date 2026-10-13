#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --qos=bio_ai
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=k562_swap
#SBATCH --output=/grid/koo/home/pmantill/projects/AutoSituate/k562_agft_benchmark/swap/slurm/logs/%x_%j.out
#SBATCH --error=/grid/koo/home/pmantill/projects/AutoSituate/k562_agft_benchmark/swap/slurm/logs/%x_%j.err
#
# All-by-all motif-syntax x context swap, per annotation method (84x84 per method).
# Forward passes only, same K562 encoder + dinuc-shuffle as the KO/KI benchmark.

DIR="/grid/koo/home/pmantill/projects/AutoSituate/k562_agft_benchmark"
module purge
module load EB5
module load CUDA/12.9.1
source /grid/koo/home/pmantill/projects/Virtual_Experiments/Hippo_axis/Hippo_dependency_mpra/Hippo_agft_venv/bin/activate
export PYTHONUNBUFFERED=1

echo "K562 swap | Job $SLURM_JOB_ID | GPU $CUDA_VISIBLE_DEVICES | $(date)"
cd "$DIR"
uv run python swap/run_swap.py
uv run python swap/make_swap_figures.py        # repaint every swap figure
echo "Done: $(date)"
