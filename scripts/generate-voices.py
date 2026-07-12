#!/usr/bin/env python3
"""
generate-voices.py — produce QuietVoice's rights-clean synthetic voice pack.

WHAT: generates N named voices with Qwen3-TTS **VoiceDesign**. VoiceDesign creates a
voice from a natural-language *description* — the result is a synthetic voice that is
NOT a recording of, and not cloned from, any real person (see the model card, quoted in
the emitted LICENSE.md). Every output here is therefore an Apache-2.0 model output with
no voice-likeness / portrait-rights encumbrance.

WHY reproducible: a legal auditor can re-run this exact script against the same public
model (`Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`, Apache-2.0) and obtain equivalent voices,
confirming provenance end-to-end. No real audio is ever an input — only text.

USAGE:
    pip install -U qwen-tts soundfile
    python generate-voices.py --out ./voices-out
    # then package ./voices-out into qwen-tts-synth-voices-v<N>.tgz (README + LICENSE included)

Device: uses CUDA (bf16 + flash-attn) if present, else Apple MPS, else CPU (fp32).
"""
import argparse, json, os, sys

# Fixed reference sentence every voice speaks. Chosen for varied prosody (good as a
# cloning reference downstream) and to carry no third-party text. English is neutral.
REF_TEXT = ("Hello, this is QuietVoice. Here is what I found, what I think, "
            "and the one thing I would change next.")

# 10 generic voice *descriptions* — no real person is named or referenced. VoiceDesign
# synthesises a novel voice matching each description.
VOICES = [
    ("voice01", "A warm, upbeat young woman with a bright, expressive, welcoming radio-host tone."),
    ("voice02", "A calm, deep-voiced older man, measured and reassuring."),
    ("voice03", "A bright, energetic young man, upbeat and quick."),
    ("voice04", "A gentle, soft-spoken woman with a soothing, even delivery."),
    ("voice05", "A confident professional woman, crisp and articulate."),
    ("voice06", "A laid-back, casual man with an easy-going, relaxed tone."),
    ("voice07", "A cheerful mature woman, expressive and lively."),
    ("voice08", "A serious, authoritative man in a clear news-anchor style."),
    ("voice09", "A friendly young man with a crisp, clear, upbeat tech-presenter energy."),
    ("voice10", "A mellow, neutral narrator with an even, unhurried pace."),
]

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"


def pick_device():
    import torch
    if torch.cuda.is_available():
        return "cuda:0", torch.bfloat16, "flash_attention_2"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", torch.float32, "sdpa"
    return "cpu", torch.float32, "sdpa"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./voices-out")
    ap.add_argument("--language", default="English")
    ap.add_argument("--only", default="", help="comma-separated voice names to (re)generate; others are kept")
    args = ap.parse_args()
    only = {x.strip() for x in args.only.split(",") if x.strip()}
    os.makedirs(args.out, exist_ok=True)

    import torch  # noqa
    import soundfile as sf
    from qwen_tts import Qwen3TTSModel

    device, dtype, attn = pick_device()
    print(f"device={device} dtype={dtype} attn={attn} model={MODEL_ID}", file=sys.stderr)
    model = Qwen3TTSModel.from_pretrained(
        MODEL_ID, device_map=device, dtype=dtype, attn_implementation=attn,
    )

    manifest = {
        "model": MODEL_ID,
        "model_license": "Apache-2.0",
        "method": "VoiceDesign (text description -> synthetic voice; no real audio input)",
        "ref_text": REF_TEXT,
        "torch": torch.__version__,
        "voices": [],
    }
    for name, instruct in VOICES:
        manifest["voices"].append({"name": name, "instruct": instruct,
                                   "wav": f"{name}.wav", "ref_text_file": f"{name}.txt"})
        if only and name not in only:
            print(f"-- keep {name} (not in --only)", file=sys.stderr)
            continue
        print(f"-> {name}: {instruct}", file=sys.stderr)
        wavs, sr = model.generate_voice_design(text=REF_TEXT, language=args.language, instruct=instruct)
        sf.write(os.path.join(args.out, f"{name}.wav"), wavs[0], sr)
        with open(os.path.join(args.out, f"{name}.txt"), "w") as f:
            f.write(REF_TEXT + "\n")

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"== wrote {len(VOICES)} voices + manifest.json to {args.out} ==", file=sys.stderr)


if __name__ == "__main__":
    main()
