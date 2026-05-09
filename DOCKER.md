# Docker and bserver reproduction

This image uses PyTorch 2.7.1 with CUDA 12.8 for RTX 50-series GPUs. On `bserver`, it is configured to expose host GPU 1 (`NVIDIA GeForce RTX 5090`) as container `cuda:0`.

## Build

```bash
docker build -t pearl-wan:latest .
```

## Prepare models and data in mainland China

```bash
mkdir -p /home/b/models/pearl-wan /home/b/models/huggingface data exp
docker run --rm --gpus device=1 \
  -e HF_ENDPOINT=https://hf-mirror.com \
  -v /home/b/models/pearl-wan:/workspace/models \
  -v /home/b/models/huggingface:/workspace/.cache/huggingface \
  -v "$PWD/data:/workspace/pearl_wan/data" \
  --entrypoint bash pearl-wan:latest \
  scripts/prepare_reproduction_assets.sh
```

The scripts use `hf-mirror.com` first and keep model weights outside the image.

## Smoke test

```bash
docker run --rm --gpus device=1 --entrypoint nvidia-smi pearl-wan:latest --query-gpu=index,name --format=csv,noheader

docker run --rm --gpus device=1 \
  -e HF_ENDPOINT=https://hf-mirror.com \
  -v /home/b/models/pearl-wan:/workspace/models \
  -v /home/b/models/huggingface:/workspace/.cache/huggingface \
  -v "$PWD/data:/workspace/pearl_wan/data" \
  -v "$PWD/exp:/workspace/pearl_wan/exp" \
  pearl-wan:latest \
  --device_edge cuda \
  --device_cloud cuda \
  --num_samples 1 \
  --max_tokens 1 \
  --enable_adaptive_window \
  --enable_compression \
  --enable_fallback
```

## Full repository reproduction

```bash
docker compose run --rm --entrypoint bash pearl-wan scripts/run_reproduction.sh
```

This runs the same benchmark matrix as `run_benchmark_ablation.slurm`: RTTs `20 50 100`, four ablations, HumanEval/GSM8K/MGSM with `limit=5`, and the 1.5B -> 7B speed test with `num_samples=2`.
