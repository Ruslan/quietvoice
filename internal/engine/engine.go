// Package engine builds an inference.Engine from configuration, selecting the
// local (in-process) or remote (HTTP) adapter. It is the single place that
// wires the inference-plane choice, shared by the orchestrator (cmd/quietvoice)
// and the standalone inference node (cmd/inferenced).
package engine

import (
	"fmt"

	"quietvoice/internal/config"
	"quietvoice/internal/inference"
	"quietvoice/internal/inference/local"
	"quietvoice/internal/inference/remote"
)

// BuildLocal constructs the in-process engine from config, assembling the ASR
// ensemble from whichever recognizer models are configured.
func BuildLocal(cfg config.Config) *local.Engine {
	var recognizers []local.Recognizer
	if cfg.VoxtralModel != "" {
		recognizers = append(recognizers, local.Recognizer{Name: "voxtral", Model: cfg.VoxtralModel, Backend: "voxtral4b"})
	}
	if cfg.WhisperModel != "" {
		recognizers = append(recognizers, local.Recognizer{Name: "whisper-large", Model: cfg.WhisperModel, Backend: "whisper"})
	}
	for _, r := range cfg.Recognizers { // arbitrary additional recognizers
		recognizers = append(recognizers, local.Recognizer{Name: r.Name, Model: r.Model, Backend: r.Backend})
	}
	return local.New(local.Config{
		CrispasrBin:   cfg.CrispasrBin,
		LlamaMtmdBin:  cfg.LlamaMtmdBin,
		TTSModel:      cfg.TTSModel,
		TTSCodecModel: cfg.TTSCodecModel,
		TTSVoiceRef:   cfg.TTSVoiceRef,
		TTSRefText:    cfg.TTSRefText,
		GemmaModel:    cfg.GemmaModel,
		GemmaMMProj:   cfg.GemmaMMProj,
		Recognizers:   recognizers,
		WorkDir:       cfg.WorkDir,
		Lang:          cfg.Lang,

		// Hot-server TTS pool + supervisor.
		TTSServerURLs:  cfg.TTSServerURLs,
		TTSServerToken: cfg.TTSServerToken,
		TTSVoice:       cfg.TTSVoice,
		TTSServerBin:   cfg.TTSServerBin,
		TTSVoiceDir:    cfg.TTSVoiceDir,
		TTSBackend:     cfg.TTSBackend,
		TTSBasePort:    cfg.TTSBasePort,
		TTSMaxReplicas: cfg.TTSMaxReplicas,
		TTSModelPaths:  cfg.TTSModelPaths,
		TTSLibDir:      cfg.TTSLibDir,

		// Heterogeneous managed replicas: asr (crispasr whisper) + gemma (llama-server).
		ASRServerBin:   cfg.ASRServerBin,
		ASRModel:       cfg.ASRModel,
		ASRBackend:     cfg.ASRBackend,
		ASRLibDir:      cfg.ASRLibDir,
		GemmaServerBin: cfg.GemmaServerBin,
		GemmaLibDir:    cfg.GemmaLibDir,
	})
}

// Build selects local or remote inference per cfg.InferenceMode.
func Build(cfg config.Config) (inference.Engine, error) {
	switch cfg.InferenceMode {
	case "", "local":
		return BuildLocal(cfg), nil
	case "remote":
		if cfg.InferenceURL == "" {
			return nil, fmt.Errorf("inference mode=remote requires INFERENCE_URL")
		}
		return remote.New(remote.Config{
			BaseURL:         cfg.InferenceURL,
			Token:           cfg.InferenceToken,
			WorkDir:         cfg.WorkDir,
			VoiceName:       cfg.VoiceName,
			VoiceWav:        cfg.VoiceWav,
			VoiceTranscript: cfg.VoiceTranscript,
		}), nil
	default:
		return nil, fmt.Errorf("unknown INFERENCE_MODE %q (want local or remote)", cfg.InferenceMode)
	}
}
