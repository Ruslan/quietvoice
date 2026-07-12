#!/usr/bin/env bash
# bootstrap-local.sh — turnkey LOCAL setup for a QuietVoice judge/user.
#
# You already ran `make mi300x` on the AMD MI300X box (the inference plane). This wires
# your LOCAL machine to it: SSH tunnel → the QuietVoice MCP control plane in Docker →
# registered into Claude Code and/or Codex. ~5 minutes; then your agent can say & listen.
#
# You provide TWO things, interactively:
#   1) the SSH command to the box   (e.g.  ssh -p 22 root@1.2.3.4)
#   2) your Telegram bot token      (from @BotFather)
# It fetches INFERENCE_TOKEN off the box over SSH and discovers your chat id via the bot.
# PRIVACY: the control plane (your Telegram token + your voice) runs LOCALLY in Docker on
# THIS machine; nothing of yours is sent to the rented box.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="quietvoice-control:latest"
ENVFILE="$REPO/quietvoice-mcp.env"          # gitignored (holds tokens)
MCP_URL="http://localhost:8090/rpc"
die(){ echo "!! $*" >&2; exit 1; }
yesno(){ read -r -p "$1 [y/N] " a; [ "$a" = y ] || [ "$a" = Y ]; }
for b in docker curl ssh; do command -v "$b" >/dev/null || die "need '$b' on PATH"; done

# --- 1. box SSH + fetch INFERENCE_TOKEN over it (</dev/null so remote reads don't eat our stdin) ---
read -r -p "SSH command to the MI300X box (e.g. 'ssh -p 22 root@host'): " SSH_CMD
[ "${SSH_CMD%% *}" = ssh ] || die "that doesn't start with 'ssh'"
echo "== locating deploy/.env.mi300x on the box ..."
BOX_ENV="$($SSH_CMD 'find /workspace /root /home -name .env.mi300x 2>/dev/null | head -1' </dev/null | tr -d '\r' || true)"
[ -n "$BOX_ENV" ] || die "no deploy/.env.mi300x on the box — did 'make mi300x' run there?"
TOKEN="$($SSH_CMD "grep '^INFERENCE_TOKEN=' '$BOX_ENV'" </dev/null | head -1 | cut -d= -f2- | tr -d '\r' || true)"
[ -n "$TOKEN" ] || die "INFERENCE_TOKEN not found in $BOX_ENV on the box"
echo "   got token (${#TOKEN} chars) from $BOX_ENV"

# Clear any prior MCP container BEFORE telegram polling (else it 409s our getUpdates).
docker rm -f quietvoice-mcp >/dev/null 2>&1 || true

# --- 2. SSH tunnel :9095 (bind 0.0.0.0 so the Docker MCP reaches it via host.docker.internal) ---
pkill -f '0.0.0.0:9095:127.0.0.1:9095' 2>/dev/null || true    # drop a stale tunnel to another box
echo "== opening SSH tunnel  localhost:9095 -> box:9095 (background) ..."
$SSH_CMD -fN -o ExitOnForwardFailure=yes -L 0.0.0.0:9095:127.0.0.1:9095 </dev/null \
  || die "tunnel failed (port 9095 already bound, or box unreachable)"
sleep 1
curl -fsS --max-time 8 http://127.0.0.1:9095/healthz >/dev/null \
  || die "node /healthz unreachable through the tunnel"
code="$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $TOKEN" http://127.0.0.1:9095/v1/voices || echo 000)"
case "$code" in
  200) echo "   node reachable + token OK. voices: $(curl -s -H "Authorization: Bearer $TOKEN" http://127.0.0.1:9095/v1/voices)";;
  401|403) die "token rejected ($code) — must equal INFERENCE_TOKEN in deploy/.env.mi300x on the box";;
  5*) die "node not ready ($code) — the fleet is still booting; watch 'make mi300x-logs' on the box, then re-run";;
  *) die "unexpected /v1/voices status: $code";;
esac

# --- 3. Telegram: validate bot, discover chat id via /start ---
read -r -p "Telegram bot token (from @BotFather): " BOT
me="$(curl -s "https://api.telegram.org/bot$BOT/getMe")"
echo "$me" | grep -q '"ok":true' || die "Telegram rejected that token"
UN="$(echo "$me" | sed -E 's/.*"username":"([^"]+)".*/\1/')"
echo "== bot @$UN verified.  OPEN https://t.me/$UN , SEND /start, then press Enter here."
read -r _
CHAT=""
for _ in $(seq 1 20); do
  U="$(curl -s "https://api.telegram.org/bot$BOT/getUpdates" || true)"
  # scrape the CHAT id (not message/user id) of the most recent update; tolerate no-match under set -e
  CHAT="$(printf '%s' "$U" | grep -o '"chat":{"id":-\?[0-9]*' | tail -1 | grep -o '\-\?[0-9]\+' || true)"
  [ -n "$CHAT" ] && break
  sleep 1
done
[ -n "$CHAT" ] || die "didn't see any message from you — open the bot, send /start, then re-run"
echo "   chat id: $CHAT"

# --- 4. control-plane env + run MCP in Docker ---
cat > "$ENVFILE" <<EOF
INFERENCE_MODE=remote
INFERENCE_URL=http://host.docker.internal:9095
INFERENCE_TOKEN=$TOKEN
QUIET_VOICE_BOT_TOKEN=$BOT
QUIET_VOICE_CHAT_ID=$CHAT
VOICE_ROTATE=1
MCP_LISTEN=:8090
EOF
echo "== building the control-plane image (first build pulls golang + compiles — a few minutes) ..."
docker build -q -f "$REPO/deploy/Dockerfile.control-plane" -t "$IMAGE" "$REPO" >/dev/null
echo "== starting the MCP control plane in Docker (:8090) ..."
docker run -d --name quietvoice-mcp -p 8090:8090 \
  --add-host host.docker.internal:host-gateway \
  --env-file "$ENVFILE" "$IMAGE" >/dev/null
sleep 3
[ "$(docker inspect -f '{{.State.Running}}' quietvoice-mcp 2>/dev/null)" = true ] \
  || { docker logs --tail 20 quietvoice-mcp 2>&1 || true; die "control plane container is not running (see logs above)"; }
echo "   control plane up: $(docker ps --filter name=quietvoice-mcp --format '{{.Status}}')"

# --- 5. register into agents (with consent; never clobber silently) ---
if command -v claude >/dev/null && yesno "Register QuietVoice MCP in Claude Code (user scope)?"; then
  claude mcp add -s user --transport http quietvoice "$MCP_URL" 2>/dev/null \
    && echo "   ✓ added to Claude Code" || echo "   (already registered in Claude Code)"
fi
CODEX_CFG="$HOME/.codex/config.toml"
if [ -f "$CODEX_CFG" ] && yesno "Register QuietVoice MCP in Codex (~/.codex/config.toml)?"; then
  cp "$CODEX_CFG" "$CODEX_CFG.bak" 2>/dev/null || true
  grep -q '^\[mcp_servers.quietvoice\]' "$CODEX_CFG" \
    || printf '\n[mcp_servers.quietvoice]\nurl = "%s"\n' "$MCP_URL" >> "$CODEX_CFG"
  echo "   ✓ added to Codex (backup: $CODEX_CFG.bak)"
fi
if ! command -v claude >/dev/null && [ ! -f "$CODEX_CFG" ]; then
  printf '{\n  "mcpServers": {\n    "quietvoice": { "transport": "http", "url": "%s" }\n  }\n}\n' "$MCP_URL" > ./.mcp.json
  echo "   wrote ./.mcp.json (project-local) — point your agent at it, or add manually: $MCP_URL"
fi

echo
echo "== DONE. QuietVoice MCP live at $MCP_URL (Telegram bot @$UN → chat $CHAT)."
echo "   Ask your agent to speak (say) or listen (listen_voice)."
echo "   Stop:  docker rm -f quietvoice-mcp ; pkill -f '0.0.0.0:9095:127.0.0.1:9095'"
