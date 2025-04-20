# Running vLLM on the NVIDIA RTX 5090

This guide shows how to use [vLLM](https://github.com/vllm-project/vllm) with the NVIDIA RTX 5090 GPU. As the RTX 5090 utilizes NVIDIA's newer Blackwell architecture with CUDA 12.8, official upstream support remains limited. We focus on the [QwQ-32B](https://huggingface.co/Qwen/QwQ-32B-AWQ) model with AWQ quantization, which achieves inference speeds of approximately **65 tokens per second**. This guide will be updated as official support improves. Please submit issues for any corrections or suggestions.

---

## Prerequisites

Before you begin, ensure you have:

- **NVIDIA RTX 5090 GPU**: Installed and detected by your system. [Link](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04&target_type=deb_network)
- **Podman or Docker**: A container runtime installed. [Link](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- **Hugging Face Hub Token**: Required for model access, get it from your Hugging Face account.

---

## Key points

- [FlashInfer](https://docs.flashinfer.ai) is used as the flash attention backend to get the optimal performance (vLLM v0).
- **Cuda graph**: speeds up the inference. Don't set --enforce-eager or use 100% GPU memory settings.
- **Chunked prefill**: needed to reduce peak memory usage.
- **Prefix caching**: chat history cached between requests (also relevant for single-user setups).
- **bitsandbytes**: offers inflight quantization but runs slower than AWQ.
- **GGUF**: performs slowly with vLLM; consider using llama.cpp for better results.
- **CCACHE**: use ccache to speed up the build process. Also you can adjust the concurrency levels in the Dockerfile.
- **32k context size**: 32k context size fits in VRAM with below arguments, assuming nothing else is running on the GPU.
- **vLLM v1**: only flash-attention v2 currently works. *VLLM_FLASH_ATTN_VERSION=2* and *VLLM_ATTENTION_BACKEND=FLASH_ATTN*.
---

## Build vLLM Image

The image is based on the [NVIDIA PyTorch image](https://hub.docker.com/r/nvidia/cuda) which gets updated every month. The cuda architecture is 12.0 to save compilation time. 

```bash
mkdir -p ~/vllm/ccache
podman build -v ~/vllm/ccache:/root/.ccache  -t vllm-cu128 -f Dockerfile
```

---

## Run vLLM

```bash
podman run --rm --device nvidia.com/gpu=all --security-opt=label=disable \
    --net=host --ipc=host \
    -v ~/.cache:/root/.cache \
    --env "HUGGING_FACE_HUB_TOKEN=<hf_token>" \
    --env "VLLM_ATTENTION_BACKEND=FLASHINFER" \
    vllm-cu128 vllm serve Qwen/QwQ-32B-AWQ \
    --gpu-memory-utilization 0.9 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-model-len 32768 \
    --enable-reasoning --reasoning-parser deepseek_r1 \
    --enable-auto-tool-choice --tool-call-parser hermes 
```
Note that the reasoning is enabled which means we will have a [reasoning_content](https://docs.vllm.ai/en/latest/features/reasoning_outputs.html) in the assistant message. Also [tool usage](https://qwen.readthedocs.io/en/latest/framework/function_call.html#vllm) is supported and works well with QwQ-32b.

---

## TODO

- Reduze size of image by removing build artefacts
- Proper benchmarking of flash attention backends (flashInfer, FA-2, ...)
- ...
