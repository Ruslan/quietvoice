ifneq (,$(wildcard .env))
include .env
export
endif

.PHONY: run inferenced build test tidy

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
