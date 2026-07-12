#!/usr/bin/env bash
# pack.sh — assemble the QuietVoice synthetic voice-pack release archive from the
# generated voices + the durable legal docs + the reproducible generator. The result
# (qwen-tts-synth-voices-<ver>.tgz) is a GitHub Release asset that `make mi300x` fetches
# into the node's TTS_VOICE_DIR. Everything inside is Apache-2.0 (see LICENSE.md).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # scripts/voices-pack
REPO="$(cd "$HERE/../.." && pwd)"
VOICES="${1:-$REPO/build/voices-out}"                         # dir with voiceNN.wav/.txt + manifest.json
VER="${VER:-v1}"
STAGE="$REPO/build/qwen-tts-synth-voices-$VER"
OUT="$REPO/build/qwen-tts-synth-voices-$VER.tgz"

ls "$VOICES"/voice*.wav >/dev/null 2>&1 || { echo "!! no voiceNN.wav in $VOICES — run scripts/generate-voices.py first"; exit 1; }

rm -rf "$STAGE"; mkdir -p "$STAGE"
cp "$VOICES"/voice*.wav "$VOICES"/voice*.txt "$VOICES"/manifest.json "$STAGE"/
cp "$HERE/README.md" "$HERE/LICENSE.md" "$STAGE"/
cp "$REPO/scripts/generate-voices.py" "$STAGE"/
( cd "$REPO/build" && tar czf "$OUT" "qwen-tts-synth-voices-$VER" )
echo "== packed: $OUT =="
shasum -a 256 "$OUT"
echo "-- contents --"; tar tzf "$OUT"
echo "Next: upload as a GitHub Release asset (gh release upload / piccy), point the fetch at its URL."
