// Package config loads QuietVoice settings from the environment (and an
// optional .env file). It keeps the control plane and inference plane settings
// together but distinct, so the same binary runs as a monolith (inference.mode
// = local) or as an orchestrator talking to a remote GPU node (mode = remote).
package config

import (
	"bufio"
	"os"
	"strconv"
	"strings"
	"time"
)

// ASRRecognizer declares one recognizer in the ensemble.
type ASRRecognizer struct {
	Name    string // label shown to Gemma
	Backend string // crispasr backend, e.g. "voxtral4b" or "whisper"
	Model   string // path to the model
}

// Config is the fully-resolved runtime configuration.
type Config struct {
	// Control plane: MCP endpoint.
	MCPListen string
	MCPToken  string

	// Telegram transport.
	TelegramToken  string
	AllowedUserIDs []int64
	AllowedChatIDs []int64
	DefaultChatID  int64
	DownloadDir    string

	// Inference plane selection.
	InferenceMode   string // "local" (monolith) or "remote"
	InferenceURL    string // remote node base URL (orchestrator → node)
	InferenceToken  string // shared bearer token for the node
	InferenceListen string // listen address when running as an inference node
	AdminEnabled    bool   // expose /admin/replicas on the inference node

	// Hot-server TTS pool + replica supervision (inference plane, Layer 1/2).
	TTSServerURLs  []string // external crispasr --server URLs (pool without a supervisor)
	TTSServerToken string   // optional bearer for the crispasr servers
	TTSVoice       string   // bare voice name registered in the servers' --voice-dir
	TTSServerBin   string   // crispasr binary for `--server`; default = CrispasrBin
	TTSVoiceDir    string   // --voice-dir shared by launched replicas
	TTSBackend     string   // crispasr --backend for launched replicas (default qwen3-tts)
	TTSBasePort    int      // first port for launched replicas (default 9100)
	TTSMaxReplicas int      // admission cap on managed replicas across all roles (default 8)
	TTSLibDir      string   // optional LD_LIBRARY_PATH for the crispasr tts servers

	// Heterogeneous managed replicas: asr (crispasr whisper) + gemma (llama-server)
	// alongside tts, launched by the same supervisor on one GPU.
	ASRServerBin   string // crispasr binary for the asr `--server`; default = TTSServerBin
	ASRModel       string // whisper model the asr replica loads; default = WhisperModel
	ASRBackend     string // crispasr --backend for asr (default whisper)
	ASRLibDir      string // optional LD_LIBRARY_PATH for the asr server; default = TTSLibDir
	GemmaServerBin string // llama-server binary for the gemma replica (default llama-server)
	GemmaLibDir    string // LD_LIBRARY_PATH for llama-server (its own lib dir, != crispasr's)
	// TTSModelPaths maps model name -> gguf path for model-aware routing
	// (decision #2). Parsed from TTS_MODELS ("name=/path;name2=/path2"). Empty =
	// single-model node (the launcher falls back to TTSModel).
	TTSModelPaths map[string]string

	// Local inference: binaries and models.
	CrispasrBin   string
	LlamaMtmdBin  string
	TTSModel      string
	TTSCodecModel string
	TTSVoiceRef   string
	TTSRefText    string
	GemmaModel    string
	GemmaMMProj   string
	// ASR ensemble for literal/assisted modes: every configured recognizer
	// transcribes and all transcripts go to Gemma to reconcile. VoxtralModel and
	// WhisperModel are shorthands; Recognizers adds any others.
	VoxtralModel string
	WhisperModel string
	Recognizers  []ASRRecognizer
	Lang         string

	// Storage and scratch.
	StorePath   string
	WorkDir     string
	EvalLogPath string

	// Voice behaviour.
	WaitTimeout     time.Duration
	SayContextCount int
	ListenMode      string // assisted (default) | intent | literal | clean_text

	// Reference voice, owned by the control plane (MCP), not the deployment. In
	// remote mode the MCP uploads this voice to the inference node on demand, so
	// a node can be deployed with no voice baked in. VoiceName is also the voice
	// name passed to synthesis.
	VoiceName       string // VOICE_NAME (falls back to legacy TTS_VOICE)
	VoiceWav        string // VOICE_WAV: local path to the reference wav
	VoiceTranscript string // VOICE_TXT resolved: file contents if it names a file, else the literal text
}

// Load reads configuration from .env (if present) then the environment.
func Load() Config {
	loadDotEnv(".env")

	c := Config{
		MCPListen: env("MCP_LISTEN", ":8090"),
		MCPToken:  os.Getenv("MCP_API_TOKEN"),

		TelegramToken:  firstEnv("QUIET_VOICE_BOT_TOKEN", "TELEGRAM_BOT_TOKEN"),
		AllowedUserIDs: envInts("QUIET_VOICE_USER_ID", "TELEGRAM_ALLOWED_USER_IDS"),
		AllowedChatIDs: envInts("QUIET_VOICE_CHAT_ID", "TELEGRAM_ALLOWED_CHAT_IDS"),
		DefaultChatID:  firstEnvInt(0, "QUIET_VOICE_CHAT_ID", "TELEGRAM_DEFAULT_CHAT_ID"),
		DownloadDir:    env("VOICE_DOWNLOAD_DIR", "voice_sessions/incoming"),

		InferenceMode:   strings.ToLower(env("INFERENCE_MODE", "local")),
		InferenceURL:    os.Getenv("INFERENCE_URL"),
		InferenceToken:  os.Getenv("INFERENCE_TOKEN"),
		InferenceListen: env("INFERENCE_LISTEN", ":9095"),
		AdminEnabled:    envBool("INFERENCE_ADMIN"),

		TTSServerURLs:  envList("TTS_SERVER_URLS", firstEnv("TTS_SERVER_URL", "REMOTE_CRISPARS", "REMOTE_CRISPASR")),
		TTSServerToken: firstEnv("TTS_SERVER_TOKEN", "REMOTE_CRISPASR_TOKEN"),
		TTSVoice:       os.Getenv("TTS_VOICE"),
		TTSServerBin:   os.Getenv("TTS_SERVER_BIN"),
		TTSVoiceDir:    os.Getenv("TTS_VOICE_DIR"),
		TTSBackend:     os.Getenv("TTS_BACKEND"),
		TTSBasePort:    int(firstEnvInt(0, "TTS_BASE_PORT")),
		TTSMaxReplicas: int(firstEnvInt(0, "TTS_MAX_REPLICAS")),
		TTSModelPaths:  parseModelPaths(os.Getenv("TTS_MODELS")),
		TTSLibDir:      os.Getenv("TTS_LIB_DIR"),

		ASRServerBin:   os.Getenv("ASR_SERVER_BIN"),
		ASRModel:       firstEnv("ASR_MODEL", "WHISPER_MODEL"),
		ASRBackend:     os.Getenv("ASR_BACKEND"),
		ASRLibDir:      os.Getenv("ASR_LIB_DIR"),
		GemmaServerBin: os.Getenv("GEMMA_SERVER_BIN"),
		GemmaLibDir:    os.Getenv("GEMMA_LIB_DIR"),

		CrispasrBin:   env("CRISPASR_BIN", "crispasr"),
		LlamaMtmdBin:  env("LLAMA_MTMD_BIN", "llama-mtmd-cli"),
		TTSModel:      os.Getenv("TTS_MODEL"),
		TTSCodecModel: os.Getenv("TTS_CODEC_MODEL"),
		TTSVoiceRef:   os.Getenv("TTS_VOICE_REF"),
		TTSRefText:    os.Getenv("TTS_REF_TEXT"),
		GemmaModel:    os.Getenv("GEMMA_MODEL"),
		GemmaMMProj:   os.Getenv("GEMMA_MMPROJ"),
		VoxtralModel:  os.Getenv("VOXTRAL_MODEL"),
		WhisperModel:  os.Getenv("WHISPER_MODEL"),
		Recognizers:   parseRecognizers(os.Getenv("ASR_RECOGNIZERS")),
		Lang:          env("ASR_LANG", "ru"),

		StorePath:   env("STORE_PATH", "voice_sessions/state.json"),
		WorkDir:     env("WORK_DIR", "voice_sessions/work"),
		EvalLogPath: env("EVAL_LOG_PATH", "voice_sessions/eval.jsonl"),

		WaitTimeout:     envDuration("LISTEN_WAIT_TIMEOUT", 10*time.Minute),
		SayContextCount: int(firstEnvInt(2, "SAY_CONTEXT_COUNT")),
		ListenMode:      firstEnv("LISTEN_MODE"), // parsed/defaulted at the inference layer

		VoiceName:       firstEnv("VOICE_NAME", "TTS_VOICE"),
		VoiceWav:        strings.TrimSpace(os.Getenv("VOICE_WAV")),
		VoiceTranscript: voiceTranscript(os.Getenv("VOICE_TXT")),
	}
	return c
}

// voiceTranscript resolves VOICE_TXT: if the value names a readable file, the
// transcript is that file's contents; otherwise the value is the literal
// transcript. Empty in, empty out.
func voiceTranscript(v string) string {
	v = strings.TrimSpace(v)
	if v == "" {
		return ""
	}
	if b, err := os.ReadFile(v); err == nil {
		return strings.TrimSpace(string(b))
	}
	return v
}

func env(key, def string) string {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		return v
	}
	return def
}

// firstEnv returns the first non-empty value among keys (canonical name first).
func firstEnv(keys ...string) string {
	for _, k := range keys {
		if v := strings.TrimSpace(os.Getenv(k)); v != "" {
			return v
		}
	}
	return ""
}

// firstEnvInt parses the first non-empty int among keys, else def.
func firstEnvInt(def int64, keys ...string) int64 {
	if v := firstEnv(keys...); v != "" {
		if n, err := strconv.ParseInt(v, 10, 64); err == nil {
			return n
		}
	}
	return def
}

// envInts parses a comma-separated int list from the first non-empty key.
func envInts(keys ...string) []int64 {
	raw := firstEnv(keys...)
	if raw == "" {
		return nil
	}
	var out []int64
	for _, part := range strings.Split(raw, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		if n, err := strconv.ParseInt(part, 10, 64); err == nil {
			out = append(out, n)
		}
	}
	return out
}

// envBool reports whether key is set to a truthy value (1/true/yes/on).
func envBool(key string) bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(key))) {
	case "1", "true", "yes", "on":
		return true
	}
	return false
}

// envList parses a comma-separated list from key; when key is empty it falls
// back to fallback as a single-element list (empty fallback => nil).
func envList(key, fallback string) []string {
	raw := strings.TrimSpace(os.Getenv(key))
	if raw == "" {
		if fallback = strings.TrimSpace(fallback); fallback != "" {
			return []string{fallback}
		}
		return nil
	}
	var out []string
	for _, part := range strings.Split(raw, ",") {
		if part = strings.TrimSpace(part); part != "" {
			out = append(out, part)
		}
	}
	return out
}

func envDuration(key string, def time.Duration) time.Duration {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			return d
		}
	}
	return def
}

// parseRecognizers parses ASR_RECOGNIZERS: semicolon-separated entries, each
// "name=backend:/path/to/model". Malformed entries are skipped.
//
//	ASR_RECOGNIZERS=parakeet=parakeet:/m/parakeet.gguf;canary=whisper:/m/canary.bin
func parseRecognizers(raw string) []ASRRecognizer {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil
	}
	var out []ASRRecognizer
	for _, entry := range strings.Split(raw, ";") {
		entry = strings.TrimSpace(entry)
		if entry == "" {
			continue
		}
		name, rest, ok := strings.Cut(entry, "=")
		if !ok {
			continue
		}
		backend, model, ok := strings.Cut(rest, ":")
		if !ok {
			continue
		}
		name, backend, model = strings.TrimSpace(name), strings.TrimSpace(backend), strings.TrimSpace(model)
		if name == "" || backend == "" || model == "" {
			continue
		}
		out = append(out, ASRRecognizer{Name: name, Backend: backend, Model: model})
	}
	return out
}

// parseModelPaths parses TTS_MODELS: semicolon-separated "name=/path" entries
// mapping a model name to the gguf the supervisor launches for it. Malformed
// entries are skipped. Returns nil when unset (single-model node).
//
//	TTS_MODELS=qwen3-tts-0.6b=/m/0.6b.gguf;qwen3-tts-1.7b=/m/1.7b.gguf
func parseModelPaths(raw string) map[string]string {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil
	}
	out := map[string]string{}
	for _, entry := range strings.Split(raw, ";") {
		entry = strings.TrimSpace(entry)
		if entry == "" {
			continue
		}
		name, path, ok := strings.Cut(entry, "=")
		name, path = strings.TrimSpace(name), strings.TrimSpace(path)
		if !ok || name == "" || path == "" {
			continue
		}
		out[name] = path
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// loadDotEnv sets variables from a KEY=VALUE file without overriding those
// already present in the environment. Missing file is not an error.
func loadDotEnv(path string) {
	f, err := os.Open(path)
	if err != nil {
		return
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		key, val, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		key = strings.TrimSpace(key)
		val = strings.Trim(strings.TrimSpace(val), `"'`)
		if key != "" {
			if _, exists := os.LookupEnv(key); !exists {
				_ = os.Setenv(key, val)
			}
		}
	}
}
