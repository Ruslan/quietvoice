// Package inference defines the sole contract between the control plane
// (the orchestrator: MCP, Telegram transport, state) and the inference plane
// (TTS + audio understanding, which may run in-process on a laptop GPU or on a
// remote GPU node). Everything the orchestrator needs from a GPU lives behind
// the Engine interface, so the same business logic runs as a monolith today and
// as a split control/inference deployment tomorrow with only a config change.
package inference

import "context"

// ListenMode selects how user audio is turned into text.
type ListenMode string

const (
	// ModeAssisted (default): ASR-assisted intent. A dedicated speech recognizer
	// (Voxtral) produces an accurate literal transcript, which is attached to the
	// Gemma prompt alongside the audio so Gemma shapes a concise intent WITHOUT
	// re-inventing mis-heard technical words. Combines Voxtral's word fidelity
	// with Gemma's intent shaping. See [[quietvoice-listen-modes]].
	ModeAssisted ListenMode = "assisted"
	// ModeIntent: Gemma interprets the audio directly (no ASR assist). Uses
	// prosody/context but can mis-hear and paraphrase technical terms.
	ModeIntent ListenMode = "intent"
	// ModeLiteral: closest possible verbatim transcript (Voxtral only), no intent
	// shaping.
	ModeLiteral ListenMode = "literal"
	// ModeCleanText: messy dictation → polished written message.
	ModeCleanText ListenMode = "clean_text"
)

// ParseListenMode maps a config string to a ListenMode, defaulting to
// ModeAssisted for empty or unrecognized values.
func ParseListenMode(s string) ListenMode {
	switch ListenMode(s) {
	case ModeIntent:
		return ModeIntent
	case ModeLiteral:
		return ModeLiteral
	case ModeCleanText:
		return ModeCleanText
	default:
		return ModeAssisted
	}
}

// SynthesizeRequest asks the engine to turn text into speech audio.
type SynthesizeRequest struct {
	Text  string `json:"text"`
	Voice string `json:"voice,omitempty"` // engine-specific voice/ref id; empty = default
	// Model selects a model-specific hot pool: empty routes to the default/random pool, a
	// set value routes to that model's pool, and an unknown model is an error.
	Model string `json:"model,omitempty"`
}

// SynthesizeResult points at synthesized audio held on disk by the engine.
// The caller owns the file and is responsible for removing it.
type SynthesizeResult struct {
	// WavPath is a local WAV file (crispasr emits 24 kHz mono).
	WavPath string `json:"wav_path"`
}

// AudioInput is the user's recorded speech to interpret. Path must reference a
// file readable by the engine (wav/ogg/opus/mp3/…).
type AudioInput struct {
	Path string `json:"path"`
}

// InterpretRequest carries the compact context around a piece of user audio.
// Deliberately small: the current agent question plus the last few spoken
// updates, never the agent's full context.
type InterpretRequest struct {
	Mode          ListenMode `json:"mode"`
	AgentQuestion string     `json:"agent_question,omitempty"`
	PreviousSay   []string   `json:"previous_say,omitempty"` // newest last; may be empty
}

// TranscriptRef is one recognizer's ASR output, retained for logging and
// offline re-evaluation (compare recognizers, or re-run the LLM over stored
// transcripts without re-running inference).
type TranscriptRef struct {
	Name string `json:"name"`
	Text string `json:"text"`
}

// InterpretResult is agent-ready text plus optional structure.
type InterpretResult struct {
	// Type classifies the utterance: command|suggestion|question|hypothesis|
	// uncertain|transcript. Best-effort; empty when the model gives no structure.
	Type string `json:"type,omitempty"`
	// Intent is the concise agent-ready text — the primary payload returned to
	// the coding agent.
	Intent string `json:"intent"`
	// Uncertainties lists things the user was unsure about (identifiers,
	// locations) that the agent must not treat as facts.
	Uncertainties []string `json:"uncertainties,omitempty"`
	// NeedsLiteral is set when the model judges a literal transcript is required.
	NeedsLiteral bool `json:"needs_literal,omitempty"`
	// Transcripts are the per-recognizer ASR outputs used (assisted/literal mode),
	// for logging and offline re-evaluation.
	Transcripts []TranscriptRef `json:"transcripts,omitempty"`
	// Raw is the unparsed model output, kept for debugging.
	Raw string `json:"raw,omitempty"`
}

// Engine is the ONLY boundary between the control plane and the inference plane.
// Implementations:
//   - inference/local:  execs crispasr (TTS) and llama.cpp Gemma in-process.
//   - inference/remote: an HTTP client to a standalone inference node.
type Engine interface {
	// Synthesize turns text into a speech audio file.
	Synthesize(ctx context.Context, req SynthesizeRequest) (*SynthesizeResult, error)
	// Interpret turns user audio (+ context) into agent-ready text.
	Interpret(ctx context.Context, audio AudioInput, req InterpretRequest) (*InterpretResult, error)
	// Health reports whether the inference backend is reachable and ready.
	Health(ctx context.Context) error
}

// VoiceRegistrar is optionally implemented by an Engine that can (re)register a
// reference voice on its inference backend before synthesis. The remote adapter
// implements it to upload the MCP-configured voice (VOICE_NAME/VOICE_WAV/
// VOICE_TXT) to whatever node it is currently pointed at, so the voice is a
// property of the control-plane config, not of the deployment: pointing the MCP
// at a fresh or restarted node transparently re-provisions the voice. The local
// (in-process) engine relies on its own voice-dir and does not implement it.
type VoiceRegistrar interface {
	// EnsureVoice registers the configured voice on the backend if it is missing.
	// It must be idempotent (safe to call before every Say) and a no-op when no
	// voice is configured.
	EnsureVoice(ctx context.Context) error
}
