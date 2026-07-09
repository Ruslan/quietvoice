# QuietVoice

**A portable spoken side-channel for AI coding agents.** Give an existing Claude
Code / Codex / any MCP-capable session a voice — it sends you concise spoken
updates and understands your contextual voice replies — over Telegram, with **no
local microphone, speakers, or custom UI**. Runs on macOS and Linux.

> QuietVoice is an asynchronous spoken control plane for headless and distributed
> AI agents.

Built for the **AMD Developer Hackathon: ACT II** (project *QuietVoice Fabric*).

---

## Why

Your agent runs long tasks while you're away from the terminal. Normal text
output stays the detailed technical record; QuietVoice adds an **intentional
spoken channel**: `say()` tells you what matters out loud, and `listen_voice()`
turns your spoken reply into agent-ready intent — not a blind transcript.

## Two planes, one contract

QuietVoice separates a **control plane** from an **inference plane**, joined by a
single Go interface (`internal/inference.Engine`). The orchestrator is a small
always-on server; GPU work is portable and swappable (local monolith today, a
remote GPU node tomorrow — e.g. an AMD MI300X on AMD Developer Cloud — with only
a config change).

```
Claude / Codex (any host)
        │ MCP over network (POST /rpc)
        ▼
┌──────────────────────────────┐   control plane (cmd/quietvoice)
│ QuietVoice orchestrator       │   MCP say/listen_voice · Telegram transport
│  mcp → voice → store          │   sessions · pending-voice queue · routing
└───────────────┬──────────────┘   no GPU, always on
                │ inference.Engine
      ┌─────────┴───────────┐
      ▼                     ▼
 local adapter         remote adapter ──HTTP──► cmd/inferenced (GPU node)
 (in-process exec)                               crispasr TTS + llama.cpp Gemma
 crispasr + llama.cpp                            portable: Metal / CUDA / ROCm
```

The inference node (`cmd/inferenced`) speaks **one OpenAI-compatible contract**
inbound and outbound — `POST /v1/audio/speech`, `POST /v1/audio/transcriptions`,
`GET|POST /v1/voices`, plus a QuietVoice-native `POST /v1/interpret` for rich
intent — so it's a **shared inference service** any client can use, not only
QuietVoice. It also scales itself: `POST /admin/replicas {"tts":N,"asr":M,
"gemma":K}` launches hot replicas live (a role-generic supervisor runs crispasr
for TTS, crispasr-whisper for ASR, and llama-server for Gemma on one GPU) while
clients keep hitting the **same URL**.

## Tools

| Tool | Meaning |
|------|---------|
| `say(text)` | Synthesize `text` (crispasr / Qwen3-TTS) and deliver it as a Telegram voice note. Long text is split into sentence-ish chunks, synthesized in parallel across the replica pool, and stitched into one voice note. An intentional spoken side channel — not a screen reader. |
| `listen_voice(prompt_text?)` | Return the user's next spoken intent: consume a pre-recorded pending voice or wait for one, then interpret the audio into concise agent-ready text. |

## Listen modes — ASR ensemble + LLM reconciliation

`listen_voice` defaults to **`assisted`**: every configured recognizer (Voxtral,
Whisper large, …) transcribes the audio **in parallel**, and all transcripts +
the audio go to **Gemma 4** to reconcile into a concise intent — told to trust
the transcripts' exact wording. This multi-hypothesis correction fixes a real,
measured failure: quantized Gemma alone mis-heard the Russian technical term
«стартуют» (*spin up*) and paraphrased it to «конфликтуют» (*conflict*); with the
ensemble it keeps the correct term.

Modes: `assisted` (default) · `intent` (Gemma audio-only) · `literal` (verbatim
ASR) · `clean_text`. See [doc/QUIETVOICE.md](doc/QUIETVOICE.md).

## Evaluation corpus

Every interpreted voice appends one JSON line to `voice_sessions/eval.jsonl`:
audio, mode, prompt, **each recognizer's transcript**, and the final intent — a
re-evaluation corpus to compare models or re-run the LLM step **without
re-inference**. Zero external dependencies (stdlib only, cgo-free); state is
plain JSON/JSONL for portability.

## Quickstart (monolith on a Mac)

Prerequisites on `PATH`: `crispasr` (TTS + Voxtral/Whisper ASR),
`llama-mtmd-cli` (llama.cpp multimodal) + Gemma 4 model & mmproj, `ffmpeg`.
Plus a Telegram bot token (@BotFather) and your numeric chat id.

```bash
cp .env.example .env
# set QUIET_VOICE_BOT_TOKEN, QUIET_VOICE_CHAT_ID, and the model paths
make run                      # MCP on :8090 (POST /rpc) + Telegram long polling
```

Then point your MCP client at `POST http://<host>:8090/rpc`. First send your bot
any message in Telegram (a bot can't message you until you do).

### Split (orchestrator + remote GPU node)

The **voice is a control-plane property**, not something baked into the node: the
GPU node deploys with no reference voice, and the orchestrator carries it
(`VOICE_NAME` / `VOICE_WAV` / `VOICE_TXT`, where `VOICE_TXT` is a transcript file
path or literal text). Before every `say` the remote adapter runs `EnsureVoice`
— `GET /v1/voices`, and only if missing, `POST /v1/voices` — so pointing the MCP
at a fresh or restarted node transparently re-registers the voice. (`TTS_VOICE`
is now just a legacy alias for `VOICE_NAME`.)

```bash
# on the GPU machine — no voice needed here
INFERENCE_LISTEN=:9095 INFERENCE_TOKEN=<token> make inferenced
# on the orchestrator — the voice travels with it
INFERENCE_MODE=remote INFERENCE_URL=http://<your-gpu-node>:9095 INFERENCE_TOKEN=<token> \
  VOICE_NAME=ded VOICE_WAV=./voices/ded.wav VOICE_TXT=./voices/ded.txt make run
```

## Agent policy (CLAUDE.md / AGENTS.md)

- Use `say` after a meaningful work phase (root cause, plan change, blocker,
  risk, decision needed, milestone) — not after every tool call. Keep it free of
  raw paths / hashes / JSON.
- When the user's entire message is exactly `.`, call `listen_voice` and treat
  the result as their actual message.

## Docs

- [doc/QUIETVOICE.md](doc/QUIETVOICE.md) — architecture & configuration

## Security

Telegram allowlist by **numeric** user/chat id; unknown senders ignored. Optional
`MCP_API_TOKEN` on `/rpc` and `INFERENCE_TOKEN` on the node. Never logs bot
tokens, auth headers, or raw audio.

## License

MIT — see [LICENSE](LICENSE).
