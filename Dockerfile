# CUDA devel base so nvcc, cuRAND, and the toolkit are all present.
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake g++ git wget \
    python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first so the layer caches across code changes.
COPY python/requirements.txt python/requirements.txt
RUN pip3 install --no-cache-dir -r python/requirements.txt

COPY . .

# Build the CUDA engine, the CLI, and the test suite.
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release -DMC_BUILD_TESTS=ON \
    && cmake --build build -j"$(nproc)"

# Default action runs the full benchmark and drops results into the mounted
# volume. Override with docker run ... python3 ml/train.py etc.
CMD ["python3", "python/benchmark.py"]
