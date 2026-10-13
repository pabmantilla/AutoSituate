#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --qos=bio_ai
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=k562_bench_ism
#SBATCH --output=/grid/koo/home/pmantill/projects/AutoSituate/k562_agft_benchmark/slurm/logs/%x_%j.out
#SBATCH --error=/grid/koo/home/pmantill/projects/AutoSituate/k562_agft_benchmark/slurm/logs/%x_%j.err
#
# The 6 benchmark methods seeded from the RAW ISM map (wt_attribution.npy) instead
# of the SEAM foreground — to test whether they benefit from the foreground like
# our method doesn't. Writes to results/ism/ only; the foreground benchmark is kept.

DIR="/grid/koo/home/pmantill/projects/AutoSituate/k562_agft_benchmark"
module purge
module load EB5
module load CUDA/12.9.1
source /grid/koo/home/pmantill/projects/Virtual_Experiments/Hippo_axis/Hippo_dependency_mpra/Hippo_agft_venv/bin/activate
export PYTHONUNBUFFERED=1
export BENCH_ATTR=ism

echo "K562 6-method benchmark on raw ISM | Job $SLURM_JOB_ID | GPU $CUDA_VISIBLE_DEVICES | $(date)"
cd "$DIR"
uv run python run_benchmark.py
echo "Done: $(date)"
