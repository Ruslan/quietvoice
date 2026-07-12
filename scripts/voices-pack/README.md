# QuietVoice synthetic voice pack (`qwen-tts-synth-voices`)

Ten named, rights-clean synthetic voices (`voice01` … `voice10`) for QuietVoice's `say`.
Generated with **Qwen3-TTS VoiceDesign** from text descriptions — no real person's voice is
used. See `LICENSE.md` for the full rights provenance and the Hugging Face licence proof.

## Contents
- `voice01.wav … voice10.wav` — the synthesized voices (24 kHz mono).
- `voice01.txt … voice10.txt` — the reference transcript each voice speaks (identical text;
  used as the `ref-text` when a downstream engine treats the WAV as a cloning reference).
- `manifest.json` — model id, licence, method, torch version, per-voice description + ref-text.
- `generate-voices.py` — the exact, re-runnable generator (provenance is reproducible).
- `LICENSE.md` — Apache-2.0 + likeness-rights analysis + HF card citations.

## How they were created (full path, for auditors)
1. Model: `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` (Apache-2.0, public on Hugging Face).
2. Method: `Qwen3TTSModel.generate_voice_design(text=REF_TEXT, language="English", instruct=<description>)`
   — VoiceDesign turns a **text description** into a novel synthetic voice. No audio input.
3. Inputs: 10 generic voice **descriptions** (varied age/gender/tone, no real person named) and
   one neutral reference sentence they each speak. Both are literals in `generate-voices.py`.
4. Output: one WAV per description → `voiceNN.wav`, plus the spoken sentence → `voiceNN.txt`.

Reproduce:
```bash
pip install -U qwen-tts soundfile
python generate-voices.py --out ./voices-out
```

## How QuietVoice uses this pack
- The archive is unpacked into the inference node's `TTS_VOICE_DIR` (`/voices`). Each voice is a
  named reference the TTS server can synthesize with.
- `TTS_VOICE=voice01` (or any) sets the **default** used when `say` specifies no voice. The
  existing resolver already does `request.Voice → else cfg.TTSVoice`, so a bundled default +
  named overrides work with no code change.
- **Stateless per-agent voices (planned):** the `say` contract stays `Say(sessionID, text)`;
  the server maps `hash(sessionID) → voiceNN`, so each agent in a swarm speaks in its own voice
  with no state and no contract change. (Deterministic hash → voice index.)

## Why not use crispasr's bundled `qwen3-tts-voice-default.gguf`?
That pack ships a single voice built from an *unattributed reference clip* (no licence metadata,
provenance unknown). This pack replaces it with voices whose provenance is fully documented and
reproducible, and which are provably not any real person's voice.
