ARG BASE_IMAGE=pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_ENDPOINT=https://hf-mirror.com \
    HF_HOME=/workspace/.cache/huggingface \
    PEARL_WAN_MODEL_DIR=/workspace/models \
    PEARL_WAN_DATA_DIR=/workspace/pearl_wan/data

WORKDIR /workspace/pearl_wan

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY . .

RUN mkdir -p /workspace/models /workspace/pearl_wan/data /workspace/pearl_wan/exp /workspace/.cache/huggingface

ENTRYPOINT ["python", "benchmark/eval_wan.py"]
CMD ["--num_samples", "1", "--max_tokens", "32", "--enable_adaptive_window", "--enable_compression", "--enable_fallback"]
