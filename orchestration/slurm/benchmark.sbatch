#!/bin/bash
#SBATCH --job-name=mcgpu-bench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/mcgpu-bench-%j.out

# Run the full MonteCarloGPU benchmark on one GPU node of an HPC cluster.
# Submit with: sbatch orchestration/slurm/benchmark.sbatch
#
# Results land in results/ keyed by the Slurm job id so parallel submissions do
# not clobber each other.

set -euo pipefail

module load cuda/12.2 || true          # site specific, harmless if absent
source .venv/bin/activate || true       # optional project virtualenv

echo "Job $SLURM_JOB_ID on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

RESULT_DIR="results/job_${SLURM_JOB_ID:-local}"
mkdir -p "$RESULT_DIR"

python3 python/benchmark.py
python3 python/convergence.py
python3 python/plot_results.py

# Archive this run so the sweep aggregator can pick it up later.
cp results/benchmark.json "$RESULT_DIR/" 2>/dev/null || true
cp results/*.png "$RESULT_DIR/" 2>/dev/null || true
echo "Done. Artifacts in $RESULT_DIR"
