#!/bin/bash -eux

model=${1:-"Qwen/QwQ-32B-AWQ"}
shift || true

podman run --rm --device nvidia.com/gpu=all --security-opt=label=disable --net=host --ipc=host \
-v ~/.cache:/root/.cache \
--env "HUGGING_FACE_HUB_TOKEN=<hf_token>" \
--env "VLLM_ATTENTION_BACKEND=FLASHINFER" \
 vllm-cu128 vllm serve ${model} \
--trust-remote-code \
--gpu-memory-utilization 0.9 \
--enable-prefix-caching \
--enable-chunked-prefill \
--max-model-len 32768 \
--disable-sliding-window \
--generation-config ${model} \
--enable-reasoning --reasoning-parser deepseek_r1 \
--enable-auto-tool-choice --tool-call-parser hermes \
"$@"

