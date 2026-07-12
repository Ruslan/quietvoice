#!/usr/bin/env bash
# serve-node.sh — bring up the QuietVoice inference plane on a GPU box.
#
# Starts four hot-model servers (all bound to localhost) plus the inferenced
# facade, waiting for each to become healthy:
#
#   crispasr --server  qwen3-tts   :9100   TTS  + /v1/voices registry
#   crispasr --server  voxtral4b   :9101   ASR  (ensemble member)
#   crispasr --server  whisper     :9102   ASR  (ensemble member)
#   llama-server       gemma+mmproj:9200   audio -> intent
#   inferenced         (our facade):9095   the ONE endpoint the Mac talks to
#
# The Mac orchestrator reaches only :9095 (over an SSH tunnel) with
# INFERENCE_MODE=remote. Config comes from node.env (see node.env.example).
#
# Portable across CUDA and ROCm: only the crispasr / llama-server binaries
# differ; this script and the Go facade do not.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${1:-$HERE/node.env}"
[ -f "$ENV_FILE" ] && { set -a; . "$ENV_FILE"; set +a; }

# --- defaults (override in node.env) ---
MODELS_DIR="${MODELS_DIR:-/workspace/models}"
LOG_DIR="${LOG_DIR:-/workspace/logs}"
VOICE_DIR="${VOICE_DIR:-/workspace/voices}"
WORK_DIR="${WORK_DIR:-/workspace/work}"

CRISPASR_BIN="${CRISPASR_BIN:-crispasr}"
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-llama-server}"
INFERENCED_BIN="${INFERENCED_BIN:-$HERE/inferenced}"

TTS_MODEL="${TTS_MODEL:-$MODELS_DIR/qwen3-tts-12hz-0.6b-base-q8_0.gguf}"
TTS_CODEC_MODEL="${TTS_CODEC_MODEL:-$MODELS_DIR/qwen3-tts-tokenizer-12hz.gguf}"
VOXTRAL_MODEL="${VOXTRAL_MODEL:-$MODELS_DIR/voxtral-mini-4b-realtime-q4_k.gguf}"
WHISPER_MODEL="${WHISPER_MODEL:-$MODELS_DIR/ggml-large-v3-turbo.bin}"
GEMMA_MODEL="${GEMMA_MODEL:-$MODELS_DIR/google_gemma-4-E4B-it-Q4_K_M.gguf}"
GEMMA_MMPROJ="${GEMMA_MMPROJ:-$MODELS_DIR/mmproj-google_gemma-4-E4B-it-f16.gguf}"

TTS_PORT="${TTS_PORT:-9100}"
VOXTRAL_PORT="${VOXTRAL_PORT:-9101}"
WHISPER_PORT="${WHISPER_PORT:-9102}"
GEMMA_PORT="${GEMMA_PORT:-9200}"
NODE_PORT="${NODE_PORT:-9095}"

ASR_LANG="${ASR_LANG:-ru}"
GEMMA_CTX="${GEMMA_CTX:-4096}"
GEMMA_NGL="${GEMMA_NGL:-999}"          # GPU layers for Gemma (999 = all)
TTS_VOICE="${TTS_VOICE:-ded}"
# Reference SOURCE that inferenced uploads via POST /v1/voices on first say.
# Must live OUTSIDE VOICE_DIR (the server registry starts empty and is filled
# through the API — voices are never pre-placed on the server filesystem).
TTS_VOICE_REF="${TTS_VOICE_REF:-/workspace/assets/$TTS_VOICE.wav}"
INFERENCE_TOKEN="${INFERENCE_TOKEN:-}" # bearer the Mac must present; set one!

mkdir -p "$LOG_DIR" "$VOICE_DIR" "$WORK_DIR"

# VRAM safety net: on NVIDIA, let CUDA spill to host RAM if VRAM is exhausted
# (four models on a 16 GB card is tight). Harmless on ROCm/other backends.
export GGML_CUDA_ENABLE_UNIFIED_MEMORY="${GGML_CUDA_ENABLE_UNIFIED_MEMORY:-1}"

# REQUIRED for voxtral4b on AMD/ROCm gfx942: without it crispasr aborts inside HIP graph
# capture ("operation not permitted when stream is capturing"). Disabling ggml CUDA graphs
# avoids the illegal mid-capture sync. Negligible perf cost on CUDA; safe to leave on.
export GGML_CUDA_DISABLE_GRAPHS="${GGML_CUDA_DISABLE_GRAPHS:-1}"

PIDS=()
cleanup() { echo "→ stopping node…"; for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM

wait_health() { # name url
  local name="$1" url="$2" i
  for i in $(seq 1 180); do
    if curl -fsS -o /dev/null "$url" 2>/dev/null; then echo "  ✓ $name healthy (${i}s)"; return 0; fi
    sleep 1
  done
  echo "  ✗ $name did not become healthy — see $LOG_DIR"; return 1
}

start() { # name logfile -- cmd...
  local name="$1" log="$2"; shift 3
  echo "→ starting $name  ($*)"
  "$@" >"$LOG_DIR/$log" 2>&1 &
  PIDS+=("$!")
}

echo "== QuietVoice inference node =="
command -v "$CRISPASR_BIN" >/dev/null || { echo "crispasr not on PATH"; exit 1; }
command -v "$LLAMA_SERVER_BIN" >/dev/null || { echo "llama-server not on PATH"; exit 1; }
[ -x "$INFERENCED_BIN" ] || { echo "inferenced binary not found/executable at $INFERENCED_BIN"; exit 1; }

start "TTS(qwen3-tts)" tts.log -- \
  "$CRISPASR_BIN" --server --backend qwen3-tts -m "$TTS_MODEL" --codec-model "$TTS_CODEC_MODEL" \
  --voice-dir "$VOICE_DIR" --host 127.0.0.1 --port "$TTS_PORT"

start "ASR(voxtral4b)" voxtral.log -- \
  "$CRISPASR_BIN" --server --backend voxtral4b -m "$VOXTRAL_MODEL" -l "$ASR_LANG" \
  --host 127.0.0.1 --port "$VOXTRAL_PORT"

start "ASR(whisper)" whisper.log -- \
  "$CRISPASR_BIN" --server --backend whisper -m "$WHISPER_MODEL" -l "$ASR_LANG" \
  --host 127.0.0.1 --port "$WHISPER_PORT"

start "Gemma(llama-server)" gemma.log -- \
  "$LLAMA_SERVER_BIN" -m "$GEMMA_MODEL" --mmproj "$GEMMA_MMPROJ" --jinja \
  -c "$GEMMA_CTX" -ngl "$GEMMA_NGL" --host 127.0.0.1 --port "$GEMMA_PORT"

wait_health "TTS"     "http://127.0.0.1:$TTS_PORT/health"
wait_health "Voxtral" "http://127.0.0.1:$VOXTRAL_PORT/health"
wait_health "Whisper" "http://127.0.0.1:$WHISPER_PORT/health"
wait_health "Gemma"   "http://127.0.0.1:$GEMMA_PORT/health"

echo "→ starting inferenced facade on :$NODE_PORT"
INFERENCE_LISTEN=":$NODE_PORT" \
INFERENCE_TOKEN="$INFERENCE_TOKEN" \
WORK_DIR="$WORK_DIR" \
TTS_SERVER_URL="http://127.0.0.1:$TTS_PORT" \
VOXTRAL_SERVER_URL="http://127.0.0.1:$VOXTRAL_PORT" \
WHISPER_SERVER_URL="http://127.0.0.1:$WHISPER_PORT" \
GEMMA_SERVER_URL="http://127.0.0.1:$GEMMA_PORT" \
TTS_VOICE="$TTS_VOICE" \
TTS_VOICE_REF="$TTS_VOICE_REF" \
  "$INFERENCED_BIN" >"$LOG_DIR/inferenced.log" 2>&1 &
PIDS+=("$!")

wait_health "inferenced" "http://127.0.0.1:$NODE_PORT/healthz"

echo
echo "== node up =="
echo "  facade:   http://127.0.0.1:$NODE_PORT  (tunnel this port to the Mac)"
echo "  logs:     $LOG_DIR"
echo "  Ctrl-C to stop everything."
wait
