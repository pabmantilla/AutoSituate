#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --qos=bio_ai
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=k562_two_stage
#SBATCH --output=/grid/koo/home/pmantill/projects/AutoSituate/k562_agft_benchmark/slurm/logs/%x_%j.out
#SBATCH --error=/grid/koo/home/pmantill/projects/AutoSituate/k562_agft_benchmark/slurm/logs/%x_%j.err
#
# AutoSituate two-stage annotator (OUR method) on the K562 AlphaGenome-encoder.
# Runs the stage-1 necessity / stage-2 sufficiency pipeline per sequence, scores
# the final regions through the SAME metrics.py as the 6-method benchmark, and
# writes ONLY new files under results/two_stage/ (nothing existing is touched).
# Then regenerates the comparison figures WITHOUT --refresh so annotations.pkl
# from the 6-method run is left intact.

DIR="/grid/koo/home/pmantill/projects/AutoSituate/k562_agft_benchmark"
module purge
module load EB5
module load CUDA/12.9.1
source /grid/koo/home/pmantill/projects/Virtual_Experiments/Hippo_axis/Hippo_dependency_mpra/Hippo_agft_venv/bin/activate
export PYTHONUNBUFFERED=1

echo "K562 two-stage (ours) | Job $SLURM_JOB_ID | GPU $CUDA_VISIBLE_DEVICES | $(date)"
cd "$DIR"
uv run python run_two_stage.py
uv run python visualize.py all              # examples + significance-examples + summaries (NO --refresh: annotations.pkl untouched)
uv run python efficiency_summary.py
echo "Done: $(date)"
