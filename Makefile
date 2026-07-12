ifneq (,$(wildcard .env))
include .env
export
endif

.PHONY: run inferenced build test tidy control-plane-docker inferenced-docker-mi300x \
        mi300x mi300x-demo mi300x-env mi300x-models mi300x-logs mi300x-down generate-voices \
        tunnel use-node

# Run the QuietVoice orchestrator (control plane: MCP + Telegram transport).
run:
	@echo "=== QuietVoice orchestrator ==="
	@echo "MCP JSON-RPC:  POST http://localhost$${MCP_LISTEN:-:8090}/rpc"
	@echo "Inference:     mode=$${INFERENCE_MODE:-local}"
	@echo "Ctrl+C to stop"
	@echo ""
	go run ./cmd/quietvoice

# Run the inference node (inference plane: crispasr TTS + llama.cpp Gemma over HTTP).
inferenced:
	@echo "=== QuietVoice inference node ==="
	@echo "Listening on $${INFERENCE_LISTEN:-:9095}"
	go run ./cmd/inferenced

test:
	go test ./...

tidy:
	go mod tidy

build:
	@echo "=== Building QuietVoice binaries ==="
	mkdir -p bin
	go build -o bin/quietvoice ./cmd/quietvoice
	go build -o bin/inferenced ./cmd/inferenced

# ============================================================================
# Container images (see deploy/). Control plane runs anywhere (CPU); the MI300X
# inference image bundles PREBUILT ROCm binaries — no compile, no GPU to build.
# ============================================================================
CONTROL_IMAGE            ?= quietvoice-control:latest
INFERENCED_MI300X_IMAGE  ?= quietvoice-inferenced-mi300x:latest
# Prebuilt ROCm engine bundles — published as GitHub Release assets, so
# `make inferenced-docker-mi300x` works out of the box. Override to use your own.
GH_REL ?= https://github.com/Ruslan/quietvoice/releases/download/engine-binaries-v1
CRISPASR_ROCM_URL ?= $(GH_REL)/rocm-mi300x-crispasr.tgz
LLAMACPP_ROCM_URL ?= $(GH_REL)/rocm-mi300x-llamacpp.tgz

# Control plane image (CPU, portable). Builds anywhere with Docker.
control-plane-docker:
	docker build -f deploy/Dockerfile.control-plane -t $(CONTROL_IMAGE) .

# Inference node image for AMD MI300X. Bundles prebuilt ROCm crispasr + llama.cpp.
inferenced-docker-mi300x:
	@[ -n "$(CRISPASR_ROCM_URL)" ] && [ -n "$(LLAMACPP_ROCM_URL)" ] || { \
	  echo "ERROR: set CRISPASR_ROCM_URL and LLAMACPP_ROCM_URL."; \
	  echo "  piccy dist/rocm-mi300x-crispasr.tgz   # -> CRISPASR_ROCM_URL"; \
	  echo "  piccy dist/rocm-mi300x-llamacpp.tgz   # -> LLAMACPP_ROCM_URL"; \
	  echo "  then: make inferenced-docker-mi300x CRISPASR_ROCM_URL=... LLAMACPP_ROCM_URL=..."; \
	  exit 1; }
	docker build -f deploy/Dockerfile.inferenced-mi300x \
	  --build-arg CRISPASR_ROCM_URL="$(CRISPASR_ROCM_URL)" \
	  --build-arg LLAMACPP_ROCM_URL="$(LLAMACPP_ROCM_URL)" \
	  -t $(INFERENCED_MI300X_IMAGE) .

# ============================================================================
# MI300X premium demo environment (docker compose). One command brings up the
# inference plane + the whole HOT fleet: 3x qwen3-tts-1.7b + whisper-large-v3-turbo
# + Gemma-4 12B bf16 (no voxtral — it aborts on ROCm/gfx942).
# Prereqs: deploy/.env.mi300x (auto-copied from .env.demo-example by mi300x-env; set INFERENCE_TOKEN +
# model paths) and the ROCm bundle URLs (CRISPASR_ROCM_URL / LLAMACPP_ROCM_URL).
# ============================================================================
MI300X_COMPOSE ?= deploy/docker-compose.mi300x.yml
MI300X_ENV     ?= deploy/.env.mi300x

# Rights-clean synthetic voice pack: generate 10 voices with Qwen3-TTS VoiceDesign
# (Apache-2.0, "not real people" — see scripts/voices-pack/LICENSE.md) and package them
# into a release archive `make mi300x` can fetch into TTS_VOICE_DIR. Needs a python env
# with qwen-tts: `pip install -U qwen-tts soundfile` (~4 GB model download on first run).
PYTHON ?= python3
generate-voices:
	$(PYTHON) scripts/generate-voices.py --out build/voices-out
	bash scripts/voices-pack/pack.sh build/voices-out

# Ensure the runtime env exists — copy the demo example if missing. The demo token
# is DANGEROUS_CHANGE_ME (accepted on a throwaway box; set a real one to keep it).
mi300x-env:
	@if [ ! -f $(MI300X_ENV) ]; then \
	  TOKEN=$$(openssl rand -hex 24 2>/dev/null || head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n'); \
	  sed "s|INFERENCE_TOKEN=DANGEROUS_CHANGE_ME|INFERENCE_TOKEN=$$TOKEN|" deploy/.env.demo-example > $(MI300X_ENV); \
	  echo "======================================================================"; \
	  echo "  created $(MI300X_ENV) with a FRESH random INFERENCE_TOKEN."; \
	  echo "  SAVE THIS — put the SAME value in the Mac control-plane .env so MCP can reach the node:"; \
	  echo "      INFERENCE_TOKEN=$$TOKEN"; \
	  echo "======================================================================"; \
	else echo "== $(MI300X_ENV) exists — keeping its INFERENCE_TOKEN =="; fi

# Fetch the HF_MODELS declared in the env into MODELS_DIR (idempotent; public repos,
# no HF token). Runs on the GPU box (fast link) — the premium set is ~30 GB.
mi300x-models: mi300x-env
	@set -a; . ./$(MI300X_ENV); set +a; bash scripts/fetch-models.sh

# compose reads env at INTERPOLATION time from --env-file (not the `env_file:` service key),
# so the ROCm bundle URLs + mount dirs must come from .env.mi300x for a fresh box (no root .env).
COMPOSE = docker compose --env-file $(MI300X_ENV) -f $(MI300X_COMPOSE)

# Bring up the premium node + fleet (detached). Ensures env + models first, then the
# compose `boot` sidecar POSTs the fleet and polls it healthy. Rehearses: git pull -> up.
# (Assumes Docker + ROCm drivers already present — recon that on the real box.)
mi300x: mi300x-models
	$(COMPOSE) up -d --build
	@echo "== node up. fleet boots via the 'boot' sidecar; follow: make mi300x-logs =="
	@echo "== Mac side — point your local MCP at this box =="; \
	  echo "   INFERENCE_TOKEN: $$(grep '^INFERENCE_TOKEN=' $(MI300X_ENV) | cut -d= -f2-)"; \
	  echo "   easiest: run deploy/bootstrap-local.sh on your Mac (ssh + telegram token -> MCP in Docker + Claude/Codex)"; \
	  echo "   manual:  make tunnel BOX=<user>@<this-box>  then  make use-node TOKEN=<token>  then  make run"

# Full demo: up, wait healthy, then round-trip end-to-end (say->listen->interpret) via
# deploy/smoke.sh using a pack voice. NO MCP. Override VOICE/VOICE_WAV/VOICE_TXT if needed.
mi300x-demo: mi300x
	@echo "== waiting for the boot sidecar to bring the fleet healthy ..."
	$(COMPOSE) wait boot
	@set -a; . ./$(MI300X_ENV); set +a; VD="$${VOICES_DIR:-/workspace/voices}"; \
	  VOICE="$${VOICE:-voice01}" VOICE_WAV="$${VOICE_WAV:-$$VD/voice01.wav}" VOICE_TXT="$${VOICE_TXT:-$$VD/voice01.txt}" \
	  bash deploy/smoke.sh

mi300x-logs:
	$(COMPOSE) logs -f

mi300x-down:
	$(COMPOSE) down

# --- Local side: consume a remote node from the Mac control plane (two-plane) ---
# For judges the turnkey path is deploy/bootstrap-local.sh (interactive: ssh + telegram
# token -> tunnel + MCP-in-Docker + Claude/Codex). These two targets are the manual path.
tunnel:               ## make tunnel BOX=user@host [RPORT=9095 LPORT=9095]
	@[ -n "$(BOX)" ] || { echo "usage: make tunnel BOX=user@host [RPORT=9095 LPORT=9095]"; exit 1; }
	@echo "== localhost:$${LPORT:-9095} -> $(BOX):$${RPORT:-9095}  (keep open; Ctrl-C stops) =="
	ssh -N -L $${LPORT:-9095}:127.0.0.1:$${RPORT:-9095} $(BOX)

use-node:             ## make use-node TOKEN=... [URL=http://127.0.0.1:9095] — verify node + write .env
	@URL="$${URL:-http://127.0.0.1:9095}"; \
	  curl -fsS --max-time 5 "$$URL/healthz" >/dev/null \
	    || { echo "!! $$URL/healthz unreachable — is the tunnel up? (make tunnel BOX=user@box)"; exit 1; }; \
	  code=$$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $(TOKEN)" "$$URL/v1/voices"); \
	  [ "$$code" = "200" ] || { echo "!! token rejected (HTTP $$code) — must equal INFERENCE_TOKEN in deploy/.env.mi300x on the box"; exit 1; }; \
	  for kv in "INFERENCE_MODE=remote" "INFERENCE_URL=$$URL" "INFERENCE_TOKEN=$(TOKEN)"; do \
	    k=$${kv%%=*}; if grep -q "^$$k=" .env 2>/dev/null; then sed -i.bak "s|^$$k=.*|$$kv|" .env; else echo "$$kv" >> .env; fi; done; rm -f .env.bak; \
	  echo "== node OK. voices: $$(curl -s -H "Authorization: Bearer $(TOKEN)" $$URL/v1/voices)"; echo "   next: make run"
