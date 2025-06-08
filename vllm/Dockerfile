FROM docker.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV MAX_JOBS=16
ENV NVCC_THREADS=4
ENV FLASHINFER_ENABLE_AOT=1
ENV USE_CUDA=1
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST='12.0+PTX'
ENV CCACHE_DIR=/root/.ccache
ENV CMAKE_BUILD_TYPE=Release
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Install required packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    kmod \
    git \
    cmake \
    ninja-build \
    build-essential \
    ccache \
    python3-pip \
    python3-dev \
    python3-venv \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip config set global.break-system-packages true
RUN pip install --no-cache-dir --upgrade --ignore-installed pip setuptools wheel
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

RUN pip install --no-cache-dir git+https://github.com/facebookresearch/xformers.git@main#egg=xformers

RUN git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git /workspace/bitsandbytes
WORKDIR /workspace/bitsandbytes
RUN cmake -DCOMPUTE_BACKEND=cuda -S .
RUN make -j 4
RUN pip install -e .

# Build flashinfer, set FLASHINFER_ENABLE_AOT=0 for JIT compiler
RUN git clone https://github.com/flashinfer-ai/flashinfer.git --recursive /workspace/flashinfer
WORKDIR /workspace/flashinfer
RUN pip install --no-cache-dir ninja build packaging "setuptools>=75.6.0"
RUN python3 -m flashinfer.aot
RUN python3 -m build --no-isolation --wheel
RUN pip install dist/flashinfer*.whl

RUN pip install aiohttp==3.11.18
RUN pip install protobuf==5.29.4
RUN pip install click==8.1.8
RUN pip install rich==13.7.1
RUN pip install starlette==0.46.2

# Build vllm
RUN git clone https://github.com/vllm-project/vllm.git /workspace/vllm
WORKDIR /workspace/vllm
RUN python3 use_existing_torch.py
RUN pip install --no-cache-dir -r requirements/build.txt
RUN pip install --no-cache-dir setuptools_scm
RUN python3 setup.py develop

RUN pip install --no-cache-dir accelerate

# (Optional) If you want to run a shell by default or some other command:
CMD ["bash"]
