#!/usr/bin/env bash
# smoke.sh — exercise each inferenced endpoint end-to-end: (register voice ->)
# speech -> transcriptions -> interpret. Run on the box, or over an SSH tunnel to :9095.
#
# The node is deployed VOICE-FREE (the MCP registers the voice at first `say`), so
# this smoke self-registers a voice to exercise TTS, mirroring what the MCP does:
#   VOICE_WAV=<ref.wav>  (+ VOICE_TXT=<transcript-file> OR VOICE_TRANSCRIPT="literal")
# It POSTs /v1/voices, then synthesizes with VOICE (default ded) and round-trips
# that WAV through ASR + interpret. Without VOICE_WAV the TTS leg is SKIPPED (with
# a note); ASR/interpret then need SMOKE_WAV=<any-speech.wav>. Idempotent: the
# voice upload is a no-op if already registered on the node.
set -euo pipefail

BASE="${BASE:-http://127.0.0.1:9095}"
TOKEN="${INFERENCE_TOKEN:-}"
VOICE="${VOICE:-ded}"
VOICE_WAV="${VOICE_WAV:-}"
WORK="${WORK:-/tmp}"
TEXT="${TEXT:-Hello! This is the QuietVoice smoke test. All models are running at once on one card.}"
OUT="$WORK/qv_smoke_tts.wav"

# Resolve the transcript: a VOICE_TXT file's contents, else a literal VOICE_TRANSCRIPT.
TRANSCRIPT="${VOICE_TRANSCRIPT:-}"
if [ -n "${VOICE_TXT:-}" ] && [ -f "$VOICE_TXT" ]; then
  TRANSCRIPT="$(cat "$VOICE_TXT")"
elif [ -n "${VOICE_TXT:-}" ]; then
  TRANSCRIPT="$VOICE_TXT"   # VOICE_TXT was a literal, not a file
fi

AUTH=(); [ -n "$TOKEN" ] && AUTH=(-H "Authorization: Bearer $TOKEN")
pass=0; fail=0
ok()   { echo "  PASS: $1"; pass=$((pass+1)); }
bad()  { echo "  FAIL: $1"; fail=$((fail+1)); }

echo "== QuietVoice smoke @ $BASE =="

# 0. replicas listing (also confirms the admin API / auth).
echo "-> GET /admin/replicas"
LIST="$(curl -sS "${AUTH[@]}" "$BASE/admin/replicas" || true)"
printf '%s\n' "$LIST" | tr ',' '\n' | grep -E '"role"|"healthy"|"port"|"url"' || printf '%s\n' "$LIST"

SMOKE_WAV="${SMOKE_WAV:-}"
if [ -n "$VOICE_WAV" ] && [ -f "$VOICE_WAV" ]; then
  # 1a. Register the voice (what the MCP's EnsureVoice does). transcript is REQUIRED.
  if [ -z "$TRANSCRIPT" ]; then
    bad "voice register -> no transcript (set VOICE_TXT=<file> or VOICE_TRANSCRIPT=\"...\"); /v1/voices 400s without it"
  else
    echo "-> POST /v1/voices (name=$VOICE, file=$VOICE_WAV)"
    RESP="$(curl -sS "${AUTH[@]}" -X POST "$BASE/v1/voices" \
      -F "name=$VOICE" -F "transcript=$TRANSCRIPT" -F "voice=@$VOICE_WAV" || true)"
    # 201 registered, or already-present — both are fine for the round-trip.
    printf '%s' "$RESP" | grep -qE '"results"|already' && ok "voice register -> $RESP" || bad "voice register -> $RESP"
  fi

  # 1b. TTS: POST /v1/audio/speech (JSON {input, voice}) -> WAV.
  # TEXT must not contain a double-quote (kept simple; escape it yourself if needed).
  echo "-> POST /v1/audio/speech (voice=$VOICE)"
  code="$(curl -sS -o "$OUT" -w '%{http_code}' "${AUTH[@]}" -H "Content-Type: application/json" \
    -X POST "$BASE/v1/audio/speech" \
    -d "$(printf '{"input":"%s","voice":"%s"}' "$TEXT" "$VOICE")" || echo 000)"
  if [ "$code" = "200" ] && [ -s "$OUT" ]; then
    bytes="$(wc -c < "$OUT" | tr -d ' ')"
    ok "TTS -> HTTP 200, $bytes bytes at $OUT"
    SMOKE_WAV="$OUT"
  else
    bad "TTS -> HTTP $code (voice registered? see /v1/voices above)"
  fi
else
  echo "-> TTS leg SKIPPED: no VOICE_WAV given (the node is voice-free; the MCP"
  echo "   registers the voice on first say). Set VOICE_WAV + VOICE_TXT to test TTS here."
fi

# ASR/interpret need a WAV: the one TTS just made, or a caller-provided SMOKE_WAV.
if [ -z "$SMOKE_WAV" ] || [ ! -s "$SMOKE_WAV" ]; then
  echo "== ASR/interpret skipped (no WAV; set SMOKE_WAV=<file.wav>). smoke: $pass passed, $fail failed =="
  [ "$fail" = 0 ] || exit 1
  exit 0
fi

# 2. ASR: POST /v1/audio/transcriptions (multipart file) -> {text}.
echo "-> POST /v1/audio/transcriptions"
RESP="$(curl -sS "${AUTH[@]}" -X POST "$BASE/v1/audio/transcriptions" \
  -F "file=@$SMOKE_WAV" -F "language=ru" || true)"
if printf '%s' "$RESP" | grep -q '"text"'; then
  ok "ASR -> $RESP"
else
  bad "ASR -> $RESP"
fi

# 3. Interpret: POST /v1/interpret (multipart audio + meta) -> intent JSON.
echo "-> POST /v1/interpret (mode=assisted)"
RESP="$(curl -sS "${AUTH[@]}" -X POST "$BASE/v1/interpret" \
  -F "audio=@$SMOKE_WAV" -F 'meta={"mode":"assisted"}' || true)"
if printf '%s' "$RESP" | grep -qE '"intent"|"type"'; then
  ok "interpret -> $RESP"
else
  bad "interpret -> $RESP"
fi

echo "== smoke: $pass passed, $fail failed =="
[ "$fail" = 0 ] || exit 1
