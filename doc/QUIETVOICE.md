# QuietVoice

Remote spoken I/O for AI coding agents. An existing Claude Code / Codex / MCP
session sends concise spoken updates to Telegram (`say`) and understands the
user's contextual voice replies (`listen_voice`) — no local microphone,
speakers, or macOS dependency. Runs on macOS and Linux.

## Two planes, one contract

QuietVoice is split into a **control plane** and an **inference plane** joined by
a single Go interface, `internal/inference.Engine` (`Synthesize`, `Interpret`).
This is deliberate: the orchestrator is a small always-on web server, while the
GPU work is portable and swappable.

```
Claude / Codex (any host)
        │ MCP over network (POST /rpc)
        ▼
┌─────────────────────────────┐        control plane (cmd/quietvoice)
│ QuietVoice orchestrator      │        - MCP JSON-RPC (say, listen_voice)
│  mcp → voice → store         │        - Telegram transport (long polling)
│  telegram transport          │        - sessions / pending voice / routing
└───────────────┬─────────────┘        no GPU, lightweight, always on
                │ inference.Engine
      ┌─────────┴──────────┐
      ▼                    ▼
 local adapter        remote adapter ──HTTP──► cmd/inferenced (GPU node)
 (in-process exec)                              crispasr TTS + llama.cpp Gemma
 crispasr + llama.cpp                           portable: Metal / CUDA / CPU
```

- **`INFERENCE_MODE=local`** — monolith. Inference runs in the orchestrator
  process on this host's GPU (Metal on a Mac, CUDA on a GPU box, CPU as a slow
  fallback). One binary, simplest to run.
- **`INFERENCE_MODE=remote`** — the orchestrator runs anywhere and calls a
  separate `inferenced` node over HTTP. The GPU machine (a rented droplet or a
  box under the desk) is a thin wrapper around the same local engine. The
  orchestrator's business logic is identical in both modes.

### One contract, one URL

`inferenced` speaks a single **OpenAI-compatible** surface inbound and outbound:

```
POST /v1/audio/speech          # TTS   {model?, input, voice, response_format} -> audio
POST /v1/audio/transcriptions  # ASR   (multipart file[, model, language]) -> {text}
GET  /v1/voices                # list registered voices (aggregate of the pool)
POST /v1/voices                # register a voice (multipart name/transcript/voice), fan-out
POST /v1/interpret             # QuietVoice-native rich intent (transcripts + tone + urgency)
POST /admin/replicas           # scale hot replicas live: {"tts":N,"asr":M,"gemma":K}
```

The client keeps hitting **one URL** while `inferenced` owns the ports, pools, and
fan-out. `POST /admin/replicas` launches or scales hot replicas on the fly: a
role-generic supervisor runs **crispasr** for TTS, **crispasr-whisper** for ASR,
and **llama-server** for Gemma on one GPU. Note that raw replication of a GGUF
engine caps around ~×2 (HIP/CUDA time-slicing); real throughput comes from
batching (e.g. a vLLM backend), which the same contract lets you drop in as
another adapter.

### Voice is a control-plane property

The reference voice is configured on the **control plane** (`VOICE_NAME`,
`VOICE_WAV`, `VOICE_TXT` — the transcript being a file path *or* literal text),
not baked into the node. An inference node deploys with **no voice** (no
`--voice-dir`, no `TTS_VOICE` on the node). Before every `say`, the remote
adapter runs `EnsureVoice`: `GET /v1/voices`, and only if the voice is missing,
`POST /v1/voices` (multipart name/transcript/voice). So pointing the MCP at a
fresh or restarted node transparently re-registers the voice, with no manual
step. (`TTS_VOICE` is now just a legacy alias for `VOICE_NAME`.)

## Tools

| Tool | Meaning |
|------|---------|
| `say(text)` | Synthesize `text` (crispasr) and deliver it as a Telegram voice note. Long text is split into sentence-ish chunks (a single synth request caps at a few hundred chars), synthesized in parallel across the replica pool, and stitched into one voice note. An intentional spoken side channel — not a screen reader. |
| `listen_voice(prompt_text?)` | Return the user's next spoken intent: consume a pre-recorded pending voice, or notify the user and wait for one, then interpret the audio with Gemma into concise agent-ready text. |

## Listen modes (how audio becomes text)

`listen_voice` supports four modes, selected by `LISTEN_MODE`. The default is
**`assisted`**, chosen because the heavily-quantized Gemma 4 E4B (Q4_K_M) can
mis-hear technical Russian and then *paraphrase* the error into a plausible but
wrong term, while the dedicated Voxtral ASR transcribes the words faithfully.

| Mode | Pipeline | Trade-off |
|------|----------|-----------|
| `assisted` (default) | **Every** configured recognizer (Voxtral, Whisper large, …) transcribes the audio in parallel; all transcripts + the audio go to Gemma, told to reconcile them and trust their exact wording | Faithful words **and** concise intent (multi-hypothesis correction). One inference per recognizer + Gemma. |
| `intent` | Gemma interprets the audio directly | Uses prosody/context, but can mis-hear and paraphrase technical terms. |
| `literal` | First recognizer only | Verbatim transcript, zero paraphrase, no shaping. |
| `clean_text` | Gemma polishes dictation | Messy speech → written message. |

The ASR ensemble is configured by `VOXTRAL_MODEL`, `WHISPER_MODEL`, and/or the
general `ASR_RECOGNIZERS` list (`name=backend:/path;…`) — any subset of one or
many recognizers. In `assisted` mode all of them run and Gemma cross-checks the
candidates (`buildUserPrompt` in `internal/inference/local`).

**Why `assisted` is the default — a real, measured example.** The user asked, in
Russian, whether Llama and Crisp *«стартуют»* (spin up) on every request or stay
resident in memory:

- `intent` (Gemma audio-only) returned *«Lama и Crisp **конфликтуют** каждый
  раунд…»* — it mis-heard *«стартуют»* as *«**конфликтуют**»* (conflict), changing
  the meaning.
- `assisted` returned *«Лама и Crisp **стартуют** … или они постоянно находятся в
  памяти для холодного старта?»* — the ASR ensemble heard *«стартуют»* correctly and
  Gemma kept the exact term while tightening the phrasing.

(The example is Russian because that is what was actually measured; the effect —
a small model paraphrasing a mis-heard technical term — is language-independent.)

The reference transcript is presented to Gemma as authoritative for technical
terms, tool names, and identifiers (`buildUserPrompt` in
`internal/inference/local`); the audio is kept only for tone/disambiguation. If
Voxtral is unavailable, `assisted` degrades gracefully to audio-only `intent`.

Set `LISTEN_MODE=literal` when you want an exact transcript, or `intent` to skip
the Voxtral pass (faster, less accurate). Set `VOXTRAL_MODEL` for any mode that
uses Voxtral.

## Structured intent (tone & urgency)

`listen_voice` doesn't just return words — Gemma also judges, from the **audio**,
the utterance `type` (command/question/decision/suggestion/hypothesis), the
speaker's `tone`, and `urgency`. When the tone is notable or urgency is high, the
agent-facing result is annotated, e.g.:

```
Patch only the new client, leave the old one, add three reconnection attempts.
[urgency: high · tone: frustrated]
```

Calm/neutral replies are returned plain (no noise). All fields are also written to
`eval.jsonl`. This is why the audio is kept alongside the transcripts in
`assisted` mode — the words come from the ASR ensemble, the *emotion* from Gemma
hearing you.

## Onboarding

Send the Telegram bot `/start` and it replies with your numeric chat id, the
ready-to-paste `.env` lines (`QUIET_VOICE_CHAT_ID` / `QUIET_VOICE_USER_ID`), and
the MCP endpoint (`MCP_PUBLIC_URL` if set) — zero-config setup for a new agent.

## Evaluation corpus (`eval.jsonl`)

Every interpreted voice appends one JSON line to `EVAL_LOG_PATH`
(`voice_sessions/eval.jsonl`): the audio path, mode, agent prompt, **each
recognizer's transcript**, and Gemma's final intent. This is a re-evaluation
corpus — you can compare recognizers, re-run just the LLM step over stored
transcripts with a different prompt/model, or build a model-selection dataset,
all **without re-running inference**. It's plain JSONL (append-only, `jq`- and
script-friendly); there is intentionally no SQLite dependency (see the roadmap
for when SQL might be worth adding behind the `store.Store` interface).

## Pending voice

The user can record a voice message *before* any agent asks. It is stored as
`pending`; the next `listen_voice` consumes it immediately (handoff §7). Routing
rules (handoff §10): exactly one waiting request → bind; none → keep pending;
several → bind to the oldest (MVP).

Telegram system messages (recognition feedback and the like) are in English.

## Components

```
cmd/quietvoice/     orchestrator (control plane)
cmd/inferenced/     standalone inference node (inference plane)
internal/
  mcp/              JSON-RPC 2.0 MCP server → voice.Service
  voice/            say / listen_voice, pending queue, routing, say-context
  telegram/         Bot API long polling, sendVoice, getFile, allowlist
  store/            Store interface + zero-dep JSON persistence
  inference/        Engine contract + wire types
    local/          crispasr TTS + llama-mtmd-cli Gemma (+ Voxtral literal)
    remote/         HTTP client to inferenced
  engine/           builds local|remote Engine from config
  audio/            WAV → OGG/Opus for Telegram (ffmpeg)
  config/           env/.env loader
```

Zero external Go dependencies (stdlib only) — no cgo, portable across macOS and
Linux. State persists to `voice_sessions/state.json`; a SQLite store can replace
it behind the same `store.Store` interface later.

## Prerequisites

- `crispasr` (TTS + Voxtral/Whisper ASR) on PATH
- `llama-mtmd-cli` (llama.cpp multimodal) on PATH, plus Gemma 4 model + mmproj
- `ffmpeg` on PATH
- A Telegram bot token (via @BotFather) and your numeric chat id

## Run

Monolith (Mac dev):

```bash
cp .env.example .env
# set TELEGRAM_BOT_TOKEN, TELEGRAM_DEFAULT_CHAT_ID, TELEGRAM_ALLOWED_USER_IDS,
# and the TTS_MODEL / GEMMA_MODEL paths
make run
```

Split (orchestrator + remote GPU node):

```bash
# on the GPU machine — no reference voice needed here
INFERENCE_LISTEN=:9095 INFERENCE_TOKEN=<token> make inferenced

# on the orchestrator — the voice travels with the control plane
INFERENCE_MODE=remote INFERENCE_URL=http://<your-gpu-node>:9095 INFERENCE_TOKEN=<token> \
  VOICE_NAME=ded VOICE_WAV=./voices/ded.wav VOICE_TXT=./voices/ded.txt make run
```

Point your MCP client at `POST http://<host>:8090/rpc`.

## Agent policy (CLAUDE.md / AGENTS.md)

- Use `say` after a meaningful work phase (root cause, plan change, blocker,
  risk, decision needed, milestone) — not after every tool call. Keep it concise
  and free of raw paths / hashes / JSON.
- When the user's entire message is exactly `.`, call `listen_voice` and treat
  the returned text as their actual message.

## Security (MVP)

- Telegram allowlist by **numeric** user/chat id; unknown senders ignored.
- Optional `MCP_API_TOKEN` bearer on `/rpc`; `INFERENCE_TOKEN` on the node.
- Never log bot tokens, auth headers, or raw audio contents.
