#!/bin/bash
#SBATCH --job-name=mcgpu-sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/mcgpu-sweep-%A_%a.out
#SBATCH --array=0-4

# Job array that sweeps volatility across five GPU tasks in parallel. Each array
# task prices the reference option at a different volatility and writes its own
# result file. This is the pattern for scaling a parameter study across an HPC
# cluster without a single long serial run.
#
# Submit with: sbatch orchestration/slurm/sweep.sbatch

set -euo pipefail
module load cuda/12.2 || true

VOLS=(0.10 0.20 0.30 0.40 0.50)
VOL=${VOLS[$SLURM_ARRAY_TASK_ID]}

echo "Array task $SLURM_ARRAY_TASK_ID pricing vol=$VOL on $(hostname)"
mkdir -p results/sweep

python3 - "$VOL" <<'PY'
import sys, json, os
sys.path.insert(0, "python")
from mcgpu.torch_pricer import TorchPricer
from mcgpu.config import REFERENCE

vol = float(sys.argv[1])
ref = REFERENCE
p = TorchPricer()
res = p.price_european(ref.spot, ref.strike, ref.rate, vol, ref.maturity,
                       10_000_000, ref.n_steps, ref.seed)
out = {"vol": vol, "price": res.price, "std_error": res.std_error,
       "gpu_ms": res.elapsed_ms, "device": res.device}
task = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
with open(f"results/sweep/vol_{task}.json", "w") as f:
    json.dump(out, f, indent=2)
print(out)
PY
