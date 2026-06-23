# Orchestration

Running the benchmark at scale on shared GPU infrastructure. Two schedulers are
covered because they are the two you meet on real clusters. Slurm on HPC systems
and Kubernetes in cloud native setups.

## Slurm

Single node benchmark on one GPU.

```bash
sbatch orchestration/slurm/benchmark.sbatch
```

Parameter sweep as a job array. Five volatility points priced in parallel across
five GPU tasks.

```bash
sbatch orchestration/slurm/sweep.sbatch
squeue --me                 # watch the array tasks
```

Each array task writes `results/sweep/vol_<task>.json`. The pattern generalizes
to any grid study. Set `--array=0-N` and index a parameter list by
`SLURM_ARRAY_TASK_ID`.

## Kubernetes

Build and push the image, then apply the persistent volume claim and the job.

```bash
docker build -t montecarlo-gpu:latest .
kubectl apply -f orchestration/k8s/results-pvc.yaml
kubectl apply -f orchestration/k8s/benchmark-job.yaml

kubectl logs -f job/montecarlo-benchmark
```

The pod requests `nvidia.com/gpu: 1`, so the NVIDIA device plugin schedules it
onto a node with a free GPU. Results persist to the `montecarlo-results` volume
so they survive the pod exiting.

## Why both

An intern on a performance team runs the same workload on whatever the cluster
exposes. Slurm dominates traditional HPC and academic clusters. Kubernetes with
the NVIDIA device plugin dominates cloud GPU fleets. The container image is the
same in both cases, which is the point of packaging the workload the way this
repo does.
