#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --qos=bio_ai
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --job-name=step1_explore
#SBATCH --output=/grid/koo/home/pmantill/projects/AutoSituate/step1_explore/slurm/logs/%x_%j.out
#SBATCH --error=/grid/koo/home/pmantill/projects/AutoSituate/step1_explore/slurm/logs/%x_%j.err
#
# Stage-1 exploration on the RAW ISM maps: run the two-stage annotator with four
# different initial seeders (naive_pos / naive_abs / adaptive_nec / seqlets),
# everything else in the pipeline unchanged. Writes results/ only.

DIR="/grid/koo/home/pmantill/projects/AutoSituate/step1_explore"
BENCH="/grid/koo/home/pmantill/projects/AutoSituate/k562_agft_benchmark"
module purge
module load EB5
module load CUDA/12.9.1
source /grid/koo/home/pmantill/projects/Virtual_Experiments/Hippo_axis/Hippo_dependency_mpra/Hippo_agft_venv/bin/activate
export PYTHONUNBUFFERED=1
# pin BLAS threads (SEAM/modisco OpenBLAS-overflow guard carried over)
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

echo "step1_explore | Job $SLURM_JOB_ID | GPU $CUDA_VISIBLE_DEVICES | $(date)"
cd "$BENCH"              # imports (config/data/model) resolve relative to the benchmark
uv run python "$DIR/run_explore.py"
echo "Done: $(date)"
