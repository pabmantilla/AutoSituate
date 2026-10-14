#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --qos=bio_ai
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --job-name=step1_rescue
#SBATCH --output=/grid/koo/home/pmantill/projects/AutoSituate/step1_explore/slurm/logs/%x_%j.out
#SBATCH --error=/grid/koo/home/pmantill/projects/AutoSituate/step1_explore/slurm/logs/%x_%j.err
#
# Same four-seeder sweep as run_explore.sbatch, but with the EXPERIMENTAL
# post-sufficiency counter-element (repressor) gate ON: when a finished
# run_two_stage result overshoots (final recovery > 1 + tol), seed candidates
# from the opposite-sign CWM tail and greedily add the one that pulls recovery
# back toward 1.0, keeping all existing annotations fixed. Writes *_rescue
# outputs only; the baseline results/ files are left untouched.

DIR="/grid/koo/home/pmantill/projects/AutoSituate/step1_explore"
BENCH="/grid/koo/home/pmantill/projects/AutoSituate/k562_agft_benchmark"
module purge
module load EB5
module load CUDA/12.9.1
source /grid/koo/home/pmantill/projects/Virtual_Experiments/Hippo_axis/Hippo_dependency_mpra/Hippo_agft_venv/bin/activate
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

# counter-element gate knobs
export RESCUE=1
export RESCUE_TOL=0.05
export RESCUE_MAX_ADD=2

echo "step1_rescue | Job $SLURM_JOB_ID | GPU $CUDA_VISIBLE_DEVICES | $(date)"
cd "$BENCH"              # imports (config/data/model) resolve relative to the benchmark
uv run python "$DIR/run_explore.py"
echo "Done: $(date)"
