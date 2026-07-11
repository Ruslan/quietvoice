# Containerized deployment

QuietVoice is two planes, so it ships as **two images**:

| Image | Dockerfile | Runs where | GPU |
|---|---|---|---|
| **control plane** | `Dockerfile.control-plane` | anywhere (Mac Docker, any Linux, a VPS) | none (CPU) |
| **inference node (MI300X)** | `Dockerfile.inferenced-mi300x` | a ROCm GPU host | AMD gfx942 |

This mirrors the architecture: the tiny always-on control plane containerizes and runs
anywhere; the inference plane runs where the accelerator is.

## GPU-in-Docker reality (read first)

- **Mac:** Docker has **no access to Metal** (containers run in a Linux VM). Run the
  control plane in Docker if you like, but run the inference node **natively** on the Mac
  (Metal) and point the control plane at it (`INFERENCE_MODE=remote`,
  `INFERENCE_URL=http://host.docker.internal:9095`).
- **AMD hackathon Jupyter pod (W7900):** it is itself a k8s pod — **no Docker inside**.
  Run the native gfx1100 binaries there.
- **Linux GPU VM (MI300X on DigitalOcean, or any CUDA/ROCm box):** full Docker + GPU
  passthrough — **both** images run as containers. This is where the MI300X image below applies.

## Control plane

```bash
make control-plane-docker            # -> quietvoice-control:latest
docker run --rm -p 8090:8090 --env-file .env quietvoice-control
```

## Inference node on MI300X (prebuilt ROCm binaries — no compile)

The image bundles the prebuilt ROCm engine binaries instead of compiling them, so the build
is fast and needs no GPU. The bundles are already published as GitHub Release assets
(`engine-binaries-v1`), and the Makefile defaults point at them — so on the GPU host it is just
clone + make. (Build your own? Upload with `gh release create` and override `CRISPASR_ROCM_URL`
/ `LLAMACPP_ROCM_URL`.)

```bash
# on the MI300X host:
git clone https://github.com/Ruslan/quietvoice && cd quietvoice
make inferenced-docker-mi300x           # pulls the ROCm bundles from the release, no compile

# run it (GPU + models mounted):
docker run --rm \
  --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render \
  --ipc host --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  -p 9095:9095 -v /workspace/models:/models --env-file .env.node \
  quietvoice-inferenced-mi300x
```

`.env.node` points the model paths at the mount, e.g.:
```ini
TTS_MODEL=/models/qwen3-tts-12hz-1.7b-base-q8_0.gguf
TTS_CODEC_MODEL=/models/qwen3-tts-tokenizer-12hz.gguf
WHISPER_MODEL=/models/ggml-large-v3-turbo.bin
GEMMA_MODEL=/models/gemma-4-12B-it-bf16.gguf
GEMMA_MMPROJ=/models/mmproj-gemma-4-12B-it-f16.gguf
```

### Known gap: `llama-mtmd-cli`

`listen_voice`'s Interpret step cold-spawns `llama-mtmd-cli`, which the current llamacpp
bundle does **not** include (it built only `llama-server` + `llama-cli`). `say` and ASR work
as-is. To enable Interpret, rebuild llama.cpp adding the `llama-mtmd-cli` target
(`cmake --build build --target llama-server llama-cli llama-mtmd-cli`) and repack the bundle —
it lands on `PATH` automatically.

## Base image

`rocm/dev-ubuntu-24.04:7.2.4` — matches the build box (Ubuntu 24.04 / glibc 2.39, ROCm 7.2.4).
If a math lib is reported missing at runtime, switch to `…:7.2.4-complete` (larger, full ROCm libs).
