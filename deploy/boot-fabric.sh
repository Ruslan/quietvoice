#!/usr/bin/env bash
# boot-fabric.sh — the one-curl fabric boot: POST /admin/replicas to bring up the
# whole QuietVoice inference plane on one GPU, then poll each replica to healthy.
# Run on the box, or over an SSH tunnel to :9095.
#
# Target (premium): 3x qwen3-tts-1.7b (tts) + whisper-large-v3-turbo (asr) + Gemma-4 12B
# bf16 (gemma). inferenced's supervisor launches the servers on consecutive ports from
# TTS_BASE_PORT (9100). The POST is SYNCHRONOUS: it health-waits each replica
# (up to 3 min each) before returning, so a slow cold model load blocks the call.
#
# NOTE on ports: adminSet iterates a JSON map, whose Go iteration order is random,
# so which role lands on 9100/9101/9102 is NOT deterministic in a single mixed
# curl. Routing is by ROLE (not port), so this is harmless. Set FABRIC_SEQUENTIAL=1
# to boot role-by-role (tts -> asr -> gemma) for deterministic port assignment.
set -euo pipefail

BASE="${BASE:-http://127.0.0.1:9095}"
TOKEN="${INFERENCE_TOKEN:-}"
N_TTS="${N_TTS:-2}"
N_ASR="${N_ASR:-1}"
N_GEMMA="${N_GEMMA:-1}"
# N_VOXTRAL: extra asr replica(s) keyed to the "voxtral" model, so the assisted
# ensemble runs a GENUINE hot Voxtral+Whisper -> Gemma (whisper on the default asr
# pool, voxtral on its own). Default 0 = single hot whisper (today's behavior).
# Requires VOXTRAL_MODEL set in inferenced's env (creates the "voxtral" recognizer).
N_VOXTRAL="${N_VOXTRAL:-0}"
POLL_SECS="${POLL_SECS:-240}"

AUTH=(); [ -n "$TOKEN" ] && AUTH=(-H "Authorization: Bearer $TOKEN")

post_replicas() { # json-body
  curl -sS "${AUTH[@]}" -H "Content-Type: application/json" \
    -X POST "$BASE/admin/replicas" -d "$1"
}

post_replicas_model() { # model json-body — launch a model-keyed replica pool
  curl -sS "${AUTH[@]}" -H "Content-Type: application/json" \
    -X POST "$BASE/admin/replicas?model=$1" -d "$2"
}

echo "== booting fabric on $BASE  (tts:$N_TTS asr:$N_ASR gemma:$N_GEMMA voxtral:$N_VOXTRAL) =="
if [ "${FABRIC_SEQUENTIAL:-0}" = "1" ]; then
  # Deterministic ports: tts first (9100..), then asr, then gemma.
  echo "-> sequential boot"
  post_replicas "{\"tts\":$N_TTS}"   >/dev/null && echo "  tts:$N_TTS requested"
  post_replicas "{\"asr\":$N_ASR}"   >/dev/null && echo "  asr:$N_ASR requested"
  RESP="$(post_replicas "{\"gemma\":$N_GEMMA}")"
else
  # One curl (the signature move). Port order across roles is nondeterministic.
  RESP="$(post_replicas "{\"tts\":$N_TTS,\"asr\":$N_ASR,\"gemma\":$N_GEMMA}")"
fi

# The handler returns HTTP 400 with an `error` field if a replica failed to come
# up (e.g. OOM / not healthy within 3m). Surface it loudly.
if printf '%s' "$RESP" | grep -q '"error"'; then
  echo "!! fabric boot reported an error:"
  printf '%s\n' "$RESP"
  echo "   (VRAM tight? drop to N_TTS=1, or use 0.6b tts / Gemma E2B — see SKILL.md)"
  exit 1
fi

# Model-keyed voxtral asr pool (the genuine hot ensemble). Separate POST because it
# needs ?model=voxtral so the supervisor loads the voxtral gguf + voxtral4b backend
# (from the recognizer registry) and keys its own pool.
if [ "$N_VOXTRAL" -ge 1 ]; then
  echo "-> voxtral asr (model-keyed hot pool): $N_VOXTRAL requested"
  VRESP="$(post_replicas_model voxtral "{\"asr\":$N_VOXTRAL}")"
  if printf '%s' "$VRESP" | grep -q '"error"'; then
    echo "!! voxtral asr replica failed to come up:"
    printf '%s\n' "$VRESP"
    echo "   (is VOXTRAL_MODEL set in inferenced's env? is there VRAM headroom?)"
    exit 1
  fi
fi
echo "-> boot POST returned; polling health for up to ${POLL_SECS}s"

# Poll GET /admin/replicas until every listed replica is healthy.
want=$((N_TTS + N_ASR + N_GEMMA + N_VOXTRAL))
for i in $(seq 1 "$POLL_SECS"); do
  LIST="$(curl -sS "${AUTH[@]}" "$BASE/admin/replicas" || true)"
  total="$(printf '%s' "$LIST" | grep -o '"url"' | wc -l | tr -d ' ')" || total=0
  healthy="$(printf '%s' "$LIST" | grep -o '"healthy":true' | wc -l | tr -d ' ')" || healthy=0
  if [ "${healthy:-0}" -ge "$want" ] && [ "${total:-0}" -ge "$want" ]; then
    echo "== fabric healthy: $healthy/$want replicas up (${i}s) =="
    printf '%s\n' "$LIST"
    # Register the synthetic voice pack (idempotent). crispasr --server may not auto-load
    # voice-dir wavs as named voices, so POST each explicitly — makes TTS_VOICE + the
    # VOICE_ROTATE pool deterministic. Voices are mounted read-only at /voices.
    for w in /voices/voice*.wav; do
      [ -e "$w" ] || break
      n="$(basename "$w" .wav)"; t="/voices/$n.txt"; [ -f "$t" ] || continue
      c="$(curl -s -o /dev/null -w '%{http_code}' "${AUTH[@]}" -F "name=$n" -F "transcript=<$t" -F "voice=@$w" "$BASE/v1/voices" || echo 000)"
      case "$c" in
        2*|409) echo "  registered voice: $n";;                       # 409 = already there (idempotent)
        *)      echo "  !! voice $n register FAILED (HTTP $c) — say/rotation may 400 on this voice";;
      esac
    done
    echo "  next: ./smoke.sh"
    exit 0
  fi
  sleep 1
done
echo "!! only $healthy/$want replicas healthy after ${POLL_SECS}s"
curl -sS "${AUTH[@]}" "$BASE/admin/replicas" || true
echo
echo "   check the box: rocm-smi ; docker compose -f deploy/docker-compose.mi300x.yml logs inferenced-mi300x"
exit 1
