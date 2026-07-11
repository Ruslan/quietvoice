ifneq (,$(wildcard .env))
include .env
export
endif

.PHONY: run inferenced build test tidy control-plane-docker inferenced-docker-mi300x

# Orchestrator (control plane: MCP + Telegram transport).
run:
	@echo "QuietVoice orchestrator — MCP on $${MCP_LISTEN:-:8090}, inference=$${INFERENCE_MODE:-local}"
	go run ./cmd/quietvoice

# Inference node (inference plane: crispasr TTS + llama.cpp Gemma over HTTP).
inferenced:
	@echo "QuietVoice inference node on $${INFERENCE_LISTEN:-:9095}"
	go run ./cmd/inferenced

build:
	mkdir -p bin
	go build -o bin/quietvoice ./cmd/quietvoice
	go build -o bin/inferenced ./cmd/inferenced

test:
	go test ./...

tidy:
	go mod tidy

# --- Container images (see deploy/). Control plane runs anywhere (CPU); the ---
# --- MI300X image bundles PREBUILT ROCm binaries from GitHub Releases (no compile). ---
CONTROL_IMAGE            ?= quietvoice-control:latest
INFERENCED_MI300X_IMAGE  ?= quietvoice-inferenced-mi300x:latest
GH_REL ?= https://github.com/Ruslan/quietvoice/releases/download/engine-binaries-v1
CRISPASR_ROCM_URL ?= $(GH_REL)/rocm-mi300x-crispasr.tgz
LLAMACPP_ROCM_URL ?= $(GH_REL)/rocm-mi300x-llamacpp.tgz

control-plane-docker:
	docker build -f deploy/Dockerfile.control-plane -t $(CONTROL_IMAGE) .

inferenced-docker-mi300x:
	docker build -f deploy/Dockerfile.inferenced-mi300x \
	  --build-arg CRISPASR_ROCM_URL="$(CRISPASR_ROCM_URL)" \
	  --build-arg LLAMACPP_ROCM_URL="$(LLAMACPP_ROCM_URL)" \
	  -t $(INFERENCED_MI300X_IMAGE) .
