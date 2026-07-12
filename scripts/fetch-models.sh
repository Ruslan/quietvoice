#!/usr/bin/env bash
# fetch-models.sh — download every QuietVoice model from Hugging Face into
# MODELS_DIR. All repos are PUBLIC (no HF token needed). Idempotent: skips files
# already present; downloads to <file>.part then renames so a kill never leaves a
# half-file mistaken for complete. Run on the GPU box (fast datacenter link) —
# NOT rsync from a laptop.
set -euo pipefail
MODELS_DIR="${MODELS_DIR:-/workspace/models}"
mkdir -p "$MODELS_DIR"; cd "$MODELS_DIR"

# HF throttles single-stream downloads (~40 MB/s), so the 24 GB Gemma bf16 alone takes ~10 min
# of paid rental. aria2c with 16 connections saturates the box link (~5-20x faster). Install it
# once if missing; if apt is unavailable/fails we fall back to wget below (correctness unchanged).
if ! command -v aria2c >/dev/null 2>&1; then
  echo "→ installing aria2 (multi-connection downloader) ..."
  apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq aria2 >/dev/null 2>&1 || true
fi

# repo  filename   (filename is identical on HF and on our boxes)
MODELS=(
  "bartowski/google_gemma-4-E4B-it-GGUF        google_gemma-4-E4B-it-Q4_K_M.gguf"
  "bartowski/google_gemma-4-E4B-it-GGUF        mmproj-google_gemma-4-E4B-it-f16.gguf"
  "cstr/voxtral-mini-4b-realtime-GGUF          voxtral-mini-4b-realtime-q4_k.gguf"
  "ggerganov/whisper.cpp                       ggml-large-v3-turbo.bin"
  "cstr/qwen3-tts-0.6b-base-GGUF               qwen3-tts-12hz-0.6b-base-q8_0.gguf"
  "cstr/qwen3-tts-1.7b-base-GGUF               qwen3-tts-12hz-1.7b-base-q8_0.gguf"
  "cstr/qwen3-tts-tokenizer-12hz-GGUF          qwen3-tts-tokenizer-12hz.gguf"
)

# Env-driven override: if HF_MODELS is set (space-separated "repo::file" entries),
# fetch exactly that set instead of the built-in one. This is how `make mi300x`
# pulls the PREMIUM model set (Gemma-12B bf16 etc.) declared in deploy/.env.mi300x.
if [ -n "${HF_MODELS:-}" ]; then
  MODELS=()
  for pair in $HF_MODELS; do
    MODELS+=("${pair%%::*} ${pair##*::}")
  done
fi

for entry in "${MODELS[@]}"; do
  # shellcheck disable=SC2086
  set -- $entry; repo="$1"; file="$2"
  if [ -s "$file" ]; then echo "✓ $file (present)"; continue; fi
  echo "→ $file   ($repo)"
  url="https://huggingface.co/$repo/resolve/main/$file"
  if command -v aria2c >/dev/null 2>&1; then
    # 16 connections; download to <file>.part then rename so a kill never leaves a half-file
    # mistaken for complete. -c resumes a prior aria2c .part (via its .aria2 control file).
    aria2c -c -x16 -s16 -k1M --file-allocation=none --console-log-level=warn -o "$file.part" "$url" \
      || { echo "!! download FAILED: $url  (404? wrong repo/filename — check HF)"; exit 1; }
  else
    wget -q --show-progress -O "$file.part" "$url" \
      || { echo "!! download FAILED: $url  (404? wrong repo/filename — check HF)"; rm -f "$file.part"; exit 1; }
  fi
  mv "$file.part" "$file"
done
echo "== all models present in $MODELS_DIR =="
ls -la "$MODELS_DIR"/*.gguf "$MODELS_DIR"/*.bin 2>/dev/null

# Synthetic voice pack (optional): fetch + unpack the rights-clean voices into the TTS
# voice-dir so `say` has a default voice + a pool to rotate over. Release asset; Apache-2.0.
if [ -n "${VOICES_URL:-}" ]; then
  VD="${VOICES_DIR:-/workspace/voices}"; mkdir -p "$VD"
  echo "-> voice pack -> $VD  ($VOICES_URL)"
  curl -fsSL "$VOICES_URL" | tar xz -C "$VD" --strip-components=1 --wildcards '*/voice*.wav' '*/voice*.txt'
  n="$(ls "$VD"/voice*.wav 2>/dev/null | wc -l | tr -d ' ')"
  [ "${n:-0}" -ge 1 ] && echo "✓ $n voices unpacked into $VD" || echo "!! voice pack produced no voiceNN.wav"
fi
