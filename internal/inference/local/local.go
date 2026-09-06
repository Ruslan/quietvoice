// Package local implements inference.Engine by executing the inference tools
// in-process on the same host: crispasr for TTS and literal ASR, and llama.cpp
// (llama-mtmd-cli) with Gemma 4 for contextual audio understanding. This is the
// "monolith" deployment — the GPU is whatever this host has (Metal on a Mac,
// CUDA on a GPU box, or CPU as a slow fallback).
package local

import (
	"bytes"
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"quietvoice/internal/audio"
	"quietvoice/internal/inference"
)

// Config holds paths to the local inference binaries and models. Zero values
// fall back to sensible defaults in New.
type Config struct {
	// Binaries (looked up on PATH if not absolute).
	CrispasrBin  string // default "crispasr"
	LlamaMtmdBin string // default "llama-mtmd-cli"

	// TTS (crispasr / qwen3-tts).
	TTSModel      string // required for TTS
	TTSCodecModel string // required for TTS
	TTSVoiceRef   string // optional reference WAV for voice cloning
	TTSRefText    string // transcription of the reference WAV

	// Hot-server TTS pool (Layer 1). When TTSServerURLs is non-empty, or replicas
	// are launched via the admin API, Synthesize routes to a pooled
	// `crispasr --server` (idle-checkout) instead of cold-spawning the CLI per
	// request — this is the replication path measured at ×1.56 on the W7900.
	TTSServerURLs  []string // external crispasr --server base URLs (unmanaged)
	TTSServerToken string   // optional bearer for the crispasr servers
	TTSVoice       string   // bare voice name registered in the servers' --voice-dir

	// Supervisor (Layer 2) launch spec for admin-managed replicas.
	TTSServerBin   string // crispasr binary for `--server`; default = CrispasrBin
	TTSVoiceDir    string // --voice-dir shared by launched servers
	TTSBackend     string // crispasr --backend; default "qwen3-tts"
	TTSBasePort    int    // first port for launched replicas; default 9100
	TTSMaxReplicas int    // admission cap on managed replicas (all roles); default 8
	TTSLibDir      string // optional LD_LIBRARY_PATH for the crispasr tts servers

	// ASR replica launch spec (managed `crispasr --server --backend whisper`).
	// A launched asr replica serves /v1/audio/transcriptions (no codec, no
	// voice-dir); Transcribe routes to it when present, else cold-spawns the CLI.
	ASRServerBin string // crispasr binary for `--server`; default = TTSServerBin
	ASRModel     string // whisper model (bin/gguf) the asr replica loads
	ASRBackend   string // crispasr --backend for asr; default "whisper"
	ASRLibDir    string // optional LD_LIBRARY_PATH for the asr server; default = TTSLibDir

	// Gemma replica launch spec (managed `llama-server -m … --mmproj …`). Serves
	// /v1/chat/completions with audio-in. GemmaModel/GemmaMMProj (below) are reused
	// as the server's -m/--mmproj. NOTE: interpret-routing to this hot server is a
	// follow-up — today the replica is launched, health-tracked, and listed only;
	// Interpret still cold-spawns llama-mtmd-cli. See admin.go / SetReplicas.
	GemmaServerBin string // llama-server binary; default "llama-server"
	GemmaLibDir    string // LD_LIBRARY_PATH for llama-server (its own lib dir, != crispasr's)

	// TTSModelPaths maps a model name (the request's `model` field / a pool key)
	// to the gguf path the supervisor launches for it. When a launched model has
	// no entry here the launcher falls back to TTSModel — so a single-model node
	// works with an empty map, while a multi-model node declares its models. See
	// decision #2 (model-aware routing) in inferenced-unified-api-plan.md.
	TTSModelPaths map[string]string

	// Gemma 4 multimodal (llama.cpp) for intent/clean_text interpretation.
	GemmaModel  string
	GemmaMMProj string

	// Recognizers is the ensemble of dedicated ASRs (crispasr) used for literal
	// and assisted modes. In assisted mode ALL of them transcribe the audio and
	// every transcript is handed to Gemma to reconcile (multi-hypothesis
	// generative error correction), e.g. Voxtral + Whisper large together.
	Recognizers []Recognizer

	// WorkDir is where synthesized/intermediate files are written.
	WorkDir string

	// Lang is the ASR language hint (default "ru").
	Lang string
}

// Recognizer is one dedicated ASR: a crispasr model plus its backend. Name doubles
// as the model-keyed asr pool key: an admin launch of `?model=<Name>` and the
// ensemble's pickASRPool(rec.Name) must agree on it.
type Recognizer struct {
	Name    string // label shown to Gemma AND the asr pool key, e.g. "voxtral", "whisper-large"
	Model   string // path to the gguf/bin model
	Backend string // crispasr backend, e.g. "voxtral4b" or "whisper"
}

// recognizerByName finds a configured recognizer by its logical name (the asr pool
// key the supervisor launches and the ensemble routes to). Returns false when none
// matches.
func recognizerByName(recs []Recognizer, name string) (Recognizer, bool) {
	for _, r := range recs {
		if r.Name == name {
			return r, true
		}
	}
	return Recognizer{}, false
}

// Transcript is one recognizer's output for a piece of audio.
type Transcript struct {
	Name string
	Text string
}

// poolKey identifies a hot-server pool by (role, model). role is tts/asr/gemma;
// model is the model-aware routing key ("" = the default pool for that role).
type poolKey struct {
	role  string
	model string
}

// Engine is the in-process inference implementation.
type Engine struct {
	cfg        Config
	httpClient *http.Client // talks to the crispasr / llama-server pools

	// pools holds one hot-server pool per (role, model). model "" is the default
	// pool that external TTSServerURLs and model-less admin launches land in, and
	// that a request with an empty `model` routes to (decision #2, model-aware
	// routing). No pool for a (role, model) means CLI-fallback mode for that role.
	mu    sync.Mutex
	pools map[poolKey]*pool

	sup        *supervisor // process supervisor for admin-managed replicas
	freeVRAMMB func() int  // VRAM admission probe; overridable in tests

	// voices retains the reference voices uploaded via UploadVoice so a tts replica
	// scaled up LATER can be given the same voices as its neighbors. crispasr voice
	// caches are per-process, so a replica born after the original fan-out would
	// otherwise have none until a manual re-upload.
	voiceMu sync.Mutex
	voices  map[string]storedVoice
}

// storedVoice is a reference voice kept in memory for scale-up replay.
type storedVoice struct {
	name       string
	transcript string
	wav        []byte
	filename   string
}

// New returns a local Engine, applying defaults for unset fields.
func New(cfg Config) *Engine {
	if cfg.CrispasrBin == "" {
		cfg.CrispasrBin = "crispasr"
	}
	if cfg.LlamaMtmdBin == "" {
		cfg.LlamaMtmdBin = "llama-mtmd-cli"
	}
	if cfg.Lang == "" {
		cfg.Lang = "auto" // detect language rather than forcing one; control plane can still pin per request
	}
	if cfg.WorkDir == "" {
		cfg.WorkDir = os.TempDir()
	}
	// crispasr requires a reference transcription when the voice is a WAV. When
	// one isn't configured, fall back to a companion transcript next to the ref
	// (the samples/<voice>/text.md convention), matching the prior tooling.
	if cfg.TTSVoiceRef != "" && cfg.TTSRefText == "" {
		cfg.TTSRefText = loadCompanionRefText(cfg.TTSVoiceRef)
	}
	// Hot-server pool defaults.
	if cfg.TTSServerBin == "" {
		cfg.TTSServerBin = cfg.CrispasrBin
	}
	if cfg.TTSBackend == "" {
		cfg.TTSBackend = "qwen3-tts"
	}
	if cfg.TTSBasePort == 0 {
		cfg.TTSBasePort = 9100
	}
	if cfg.TTSMaxReplicas <= 0 {
		cfg.TTSMaxReplicas = 8
	}
	// ASR replica defaults: it's crispasr too, so it shares the tts binary and lib
	// dir by default; backend is whisper.
	if cfg.ASRServerBin == "" {
		cfg.ASRServerBin = cfg.TTSServerBin
	}
	if cfg.ASRBackend == "" {
		cfg.ASRBackend = "whisper"
	}
	if cfg.ASRLibDir == "" {
		cfg.ASRLibDir = cfg.TTSLibDir
	}
	// Gemma replica default binary (llama-server, distinct from the interpret
	// path's llama-mtmd-cli).
	if cfg.GemmaServerBin == "" {
		cfg.GemmaServerBin = "llama-server"
	}

	e := &Engine{
		cfg:        cfg,
		httpClient: &http.Client{Timeout: 5 * time.Minute},
		pools:      make(map[poolKey]*pool),
		freeVRAMMB: queryFreeVRAMMB,
	}
	// External servers from config land in the default ("") tts pool, so a request
	// with no `model` reaches them.
	if len(cfg.TTSServerURLs) > 0 {
		def := e.ensureTTSPool("")
		for _, u := range cfg.TTSServerURLs {
			if u = strings.TrimSpace(u); u != "" {
				def.add(&worker{url: strings.TrimRight(u, "/"), role: "tts"})
			}
		}
	}
	e.sup = newSupervisor(cfg, &roleLauncher{cfg: cfg})
	return e
}

// newRolePool sizes a pool's idle queue to hold every worker that can be idle at
// once: the admission cap plus any externally configured servers.
func (e *Engine) newRolePool() *pool {
	return newPool(e.cfg.TTSMaxReplicas + len(e.cfg.TTSServerURLs))
}

// ensurePool returns the pool for (role, model), creating it on first use.
// Callers hold no lock; ensurePool guards e.pools itself.
func (e *Engine) ensurePool(role, model string) *pool {
	e.mu.Lock()
	defer e.mu.Unlock()
	k := poolKey{role: role, model: model}
	p, ok := e.pools[k]
	if !ok {
		p = e.newRolePool()
		e.pools[k] = p
	}
	return p
}

// lookupPool returns the pool for (role, model), or nil if none was created.
func (e *Engine) lookupPool(role, model string) *pool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.pools[poolKey{role: role, model: model}]
}

// poolSize reports a (role, model) pool's worker count (0 if never created).
func (e *Engine) poolSize(role, model string) int {
	if p := e.lookupPool(role, model); p != nil {
		return p.size()
	}
	return 0
}

// ensureTTSPool / lookupTTSPool / ttsPoolSize are the tts-role shorthands the
// synthesize path and tests use.
func (e *Engine) ensureTTSPool(model string) *pool { return e.ensurePool("tts", model) }
func (e *Engine) lookupTTSPool(model string) *pool { return e.lookupPool("tts", model) }
func (e *Engine) ttsPoolSize(model string) int     { return e.poolSize("tts", model) }

// anyTTSServers reports whether any tts pool has at least one worker (i.e.
// hot-server mode is engaged and Synthesize should route to the pool).
func (e *Engine) anyTTSServers() bool {
	return len(e.workersForRole("tts")) > 0
}

// workersForRole returns a de-duplicated snapshot of every worker across all
// model pools of a role.
func (e *Engine) workersForRole(role string) []worker {
	e.mu.Lock()
	pools := make([]*pool, 0, len(e.pools))
	for k, p := range e.pools {
		if k.role == role {
			pools = append(pools, p)
		}
	}
	e.mu.Unlock()
	return dedupeWorkers(pools)
}

// allTTSWorkers returns every tts worker (used for voice fan-out and the tts
// health check).
func (e *Engine) allTTSWorkers() []worker { return e.workersForRole("tts") }

// allWorkers returns a de-duplicated snapshot of every worker across all roles
// and models (used for the admin listing).
func (e *Engine) allWorkers() []worker {
	e.mu.Lock()
	pools := make([]*pool, 0, len(e.pools))
	for _, p := range e.pools {
		pools = append(pools, p)
	}
	e.mu.Unlock()
	return dedupeWorkers(pools)
}

// dedupeWorkers flattens pools into a url-deduplicated worker snapshot.
func dedupeWorkers(pools []*pool) []worker {
	seen := make(map[string]bool)
	var out []worker
	for _, p := range pools {
		for _, w := range p.workers() {
			if seen[w.url] {
				continue
			}
			seen[w.url] = true
			out = append(out, w)
		}
	}
	return out
}

// routeTTS selects the pool a synthesize request must use (decision #2):
//   - model == "": the default pool (must have workers);
//   - model set with a pool that has workers: that pool;
//   - model set with no such pool: an error (no on-demand launch).
func (e *Engine) routeTTS(model string) (*pool, error) {
	if model == "" {
		p := e.lookupTTSPool("")
		if p == nil || p.size() == 0 {
			return nil, fmt.Errorf("no default TTS pool available")
		}
		return p, nil
	}
	p := e.lookupTTSPool(model)
	if p == nil || p.size() == 0 {
		return nil, fmt.Errorf("no TTS pool for model %q", model)
	}
	return p, nil
}

// loadCompanionRefText looks for a transcript beside the reference WAV:
// <dir>/text.md, then the WAV basename with .md/.txt. Returns "" if none.
func loadCompanionRefText(voiceRef string) string {
	dir := filepath.Dir(voiceRef)
	base := strings.TrimSuffix(voiceRef, filepath.Ext(voiceRef))
	for _, cand := range []string{filepath.Join(dir, "text.md"), base + ".md", base + ".txt"} {
		if data, err := os.ReadFile(cand); err == nil {
			if txt := strings.TrimSpace(string(data)); txt != "" {
				return txt
			}
		}
	}
	return ""
}

// Synthesize turns text into a 24 kHz mono WAV. When a hot `crispasr --server`
// pool is configured (or replicas were launched via the admin API) it routes to
// an idle server over HTTP; otherwise it cold-spawns the crispasr CLI.
func (e *Engine) Synthesize(ctx context.Context, req inference.SynthesizeRequest) (*inference.SynthesizeResult, error) {
	text := strings.TrimSpace(req.Text)
	if text == "" {
		return nil, fmt.Errorf("synthesize: empty text")
	}
	if e.anyTTSServers() {
		return e.synthesizeServer(ctx, text, req)
	}
	return e.synthesizeCLI(ctx, text)
}

// synthesizeServer routes the request to the pool for its model, checks out an
// idle server, and posts the text to its /v1/audio/speech endpoint, writing the
// returned WAV to the work dir.
func (e *Engine) synthesizeServer(ctx context.Context, text string, req inference.SynthesizeRequest) (*inference.SynthesizeResult, error) {
	p, err := e.routeTTS(req.Model)
	if err != nil {
		return nil, fmt.Errorf("synthesize: %w", err)
	}
	if err := os.MkdirAll(e.cfg.WorkDir, 0o755); err != nil {
		return nil, fmt.Errorf("synthesize: prepare workdir: %w", err)
	}

	voice := req.Voice
	if voice == "" {
		voice = e.cfg.TTSVoice
	}

	// A single OOM'd replica must not fail the request: retry on a DIFFERENT
	// worker. Bound the attempts by the pool size and a small absolute cap so a
	// systemic outage still fails fast rather than looping forever.
	maxAttempts := p.size()
	if maxAttempts > 3 {
		maxAttempts = 3
	}
	if maxAttempts < 1 {
		maxAttempts = 1
	}

	// Workers that answered with an error (alive, but the request failed) are held
	// out of rotation for the rest of this call so a retry checks out a genuinely
	// different worker instead of immediately re-drawing the same one. Dead workers
	// are removed from the pool entirely and need no release.
	var held []*worker
	defer func() {
		for _, w := range held {
			p.release(w)
		}
	}()

	var lastErr error
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		w, err := p.checkout(ctx)
		if err != nil {
			if lastErr != nil {
				return nil, lastErr
			}
			return nil, fmt.Errorf("synthesize: no TTS worker available: %w", err)
		}
		wav, err := ttsSpeak(ctx, e.httpClient, w.url, e.cfg.TTSServerToken, voice, text)
		if err == nil {
			p.release(w)
			out := filepath.Join(e.cfg.WorkDir, fmt.Sprintf("tts_%d.wav", time.Now().UnixNano()))
			if err := os.WriteFile(out, wav, 0o644); err != nil {
				return nil, fmt.Errorf("synthesize: write wav: %w", err)
			}
			return &inference.SynthesizeResult{WavPath: out}, nil
		}
		lastErr = fmt.Errorf("synthesize: server %s: %w", w.url, err)
		if isDeadWorkerErr(err) {
			// Worker process is gone (dial/connection refused). Evict it so it is
			// never handed out again; its idle token was consumed by checkout.
			log.Printf("[tts pool] evicting dead worker %s (%v); retrying on another worker", w.url, err)
			p.remove(w.url)
		} else {
			// Worker is alive but the request failed (e.g. missing voice, 5xx).
			// Keep it in the pool, just out of this call's rotation.
			log.Printf("[tts pool] worker %s returned an error (%v); retrying on another worker", w.url, err)
			held = append(held, w)
		}
	}
	if lastErr == nil {
		lastErr = fmt.Errorf("synthesize: no TTS worker available")
	}
	return nil, lastErr
}

// synthesizeCLI cold-spawns the crispasr CLI (the monolith fallback when no hot
// server pool is configured).
func (e *Engine) synthesizeCLI(ctx context.Context, text string) (*inference.SynthesizeResult, error) {
	if e.cfg.TTSModel == "" || e.cfg.TTSCodecModel == "" {
		return nil, fmt.Errorf("synthesize: TTS model/codec not configured")
	}

	if err := os.MkdirAll(e.cfg.WorkDir, 0o755); err != nil {
		return nil, fmt.Errorf("synthesize: prepare workdir: %w", err)
	}
	out := filepath.Join(e.cfg.WorkDir, fmt.Sprintf("tts_%d.wav", time.Now().UnixNano()))

	args := []string{
		"-m", e.cfg.TTSModel,
		"--codec-model", e.cfg.TTSCodecModel,
		"--tts", text,
		"--tts-output", out,
		"--i-have-rights",
		"--no-spoken-disclaimer",
	}
	if e.cfg.TTSVoiceRef != "" {
		args = append(args, "--voice", e.cfg.TTSVoiceRef)
		if e.cfg.TTSRefText != "" {
			args = append(args, "--ref-text", e.cfg.TTSRefText)
		}
	}

	cmd := exec.CommandContext(ctx, e.cfg.CrispasrBin, args...)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("synthesize: crispasr failed: %w: %s", err, tail(stderr.String()))
	}
	if fi, err := os.Stat(out); err != nil || fi.Size() == 0 {
		return nil, fmt.Errorf("synthesize: no audio produced at %s", out)
	}
	return &inference.SynthesizeResult{WavPath: out}, nil
}

// Interpret turns user audio into agent-ready text. The audio is transcoded to
// 16 kHz mono WAV once (both crispasr/Voxtral and llama-mtmd-cli decode it
// reliably, unlike Telegram's OGG/Opus), then dispatched by mode:
//
//   - assisted (default): Voxtral literal transcript is attached to the Gemma
//     prompt (with the audio) so Gemma preserves exact wording while shaping intent.
//   - literal: Voxtral transcript only, no shaping.
//   - intent/clean_text: Gemma interprets the audio directly.
func (e *Engine) Interpret(ctx context.Context, in inference.AudioInput, req inference.InterpretRequest) (*inference.InterpretResult, error) {
	if in.Path == "" {
		return nil, fmt.Errorf("interpret: empty audio path")
	}
	if _, err := os.Stat(in.Path); err != nil {
		return nil, fmt.Errorf("interpret: audio not readable: %w", err)
	}
	if err := os.MkdirAll(e.cfg.WorkDir, 0o755); err != nil {
		return nil, err
	}
	wav := filepath.Join(e.cfg.WorkDir, fmt.Sprintf("asr_%d.wav", time.Now().UnixNano()))
	if err := audio.ToMono16kWav(ctx, in.Path, wav); err != nil {
		return nil, fmt.Errorf("interpret: %w", err)
	}
	defer os.Remove(wav)

	switch req.Mode {
	case inference.ModeLiteral:
		transcripts := e.runRecognizers(ctx, wav, e.langFor(req))
		if len(transcripts) == 0 {
			return nil, fmt.Errorf("interpret(literal): no ASR produced a transcript")
		}
		return &inference.InterpretResult{Type: "transcript", Intent: transcripts[0].Text, Transcripts: toRefs(transcripts)}, nil
	case inference.ModeAssisted:
		return e.interpretAssisted(ctx, wav, req)
	default: // intent, clean_text
		return e.interpretGemma(ctx, wav, req, nil)
	}
}

// Transcribe is the raw OpenAI-style ASR path (POST /v1/audio/transcriptions):
// it returns a single recognizer's faithful transcript of the audio, with no
// Gemma intent shaping (that richer path stays on Interpret / /v1/interpret).
// The model arg selects a recognizer by name; empty picks the first configured
// one. lang overrides the configured language hint when non-empty.
//
// When a hot `crispasr --server --backend whisper` asr replica is running (via
// the admin API), Transcribe routes to it over HTTP (same OpenAI contract);
// otherwise it cold-spawns the crispasr CLI recognizer, as before.
func (e *Engine) Transcribe(ctx context.Context, audioPath, model, lang string) (string, error) {
	if audioPath == "" {
		return "", fmt.Errorf("transcribe: empty audio path")
	}
	if _, err := os.Stat(audioPath); err != nil {
		return "", fmt.Errorf("transcribe: audio not readable: %w", err)
	}
	if lang == "" {
		lang = e.cfg.Lang
	}
	// Hot path: a managed asr server decodes the container itself, so send the
	// original file (no ffmpeg transcode needed).
	if p := e.pickASRPool(model); p != nil {
		return e.transcribeServer(ctx, p, audioPath, model, lang)
	}
	// CLI fallback: the recognizer ensemble cold-spawns crispasr on a 16k mono WAV.
	rec, err := e.pickRecognizer(model)
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(e.cfg.WorkDir, 0o755); err != nil {
		return "", err
	}
	wav := filepath.Join(e.cfg.WorkDir, fmt.Sprintf("asr_%d.wav", time.Now().UnixNano()))
	if err := audio.ToMono16kWav(ctx, audioPath, wav); err != nil {
		return "", fmt.Errorf("transcribe: %w", err)
	}
	defer os.Remove(wav)

	return e.asrTranscriptLang(ctx, wav, rec, lang)
}

// pickASRPool returns the hot asr pool to route a transcription to, or nil to
// fall back to the CLI. A named model prefers its own pool, then the default
// ("") asr pool; an empty model uses the default pool. nil when no asr pool has
// a live worker.
func (e *Engine) pickASRPool(model string) *pool {
	if p := e.lookupPool("asr", model); p != nil && p.size() > 0 {
		return p
	}
	if model != "" {
		if p := e.lookupPool("asr", ""); p != nil && p.size() > 0 {
			return p
		}
	}
	return nil
}

// transcribeServer checks out an idle asr worker, transcribes over HTTP, and
// releases it. A dead worker (dial failure) is evicted so it is never handed out
// again; an alive-but-erroring worker is returned to the pool.
func (e *Engine) transcribeServer(ctx context.Context, p *pool, audioPath, model, lang string) (string, error) {
	w, err := p.checkout(ctx)
	if err != nil {
		return "", fmt.Errorf("transcribe: no ASR worker available: %w", err)
	}
	text, err := asrTranscribeServer(ctx, e.httpClient, w.url, e.cfg.TTSServerToken, audioPath, model, lang)
	if err != nil {
		if isDeadWorkerErr(err) {
			log.Printf("[asr pool] evicting dead worker %s (%v)", w.url, err)
			p.remove(w.url)
		} else {
			p.release(w)
		}
		return "", fmt.Errorf("transcribe: ASR server %s: %w", w.url, err)
	}
	p.release(w)
	return text, nil
}

// pickRecognizer selects a recognizer by name (case-insensitive), or the first
// configured one when model is empty. Errors when none are configured or the
// named model is unknown (decision #2: an unknown model is an error).
func (e *Engine) pickRecognizer(model string) (Recognizer, error) {
	if len(e.cfg.Recognizers) == 0 {
		return Recognizer{}, fmt.Errorf("transcribe: no ASR recognizers configured")
	}
	if model == "" {
		return e.cfg.Recognizers[0], nil
	}
	for _, r := range e.cfg.Recognizers {
		if strings.EqualFold(r.Name, model) {
			return r, nil
		}
	}
	return Recognizer{}, fmt.Errorf("transcribe: no ASR recognizer for model %q", model)
}

// langFor picks the ASR language hint for a request: the per-request Language
// the control plane forwarded (when it states one), else the node's configured
// default. This is where "language lives in the control plane" is enforced.
//
// 🔑 "auto" is the control plane's DEFAULT, not a decision — it means "no
// preference, node decides". Treating it as a decision makes ASR_LANG on the
// node dead config: an operator who pinned a language is silently overridden by
// a caller that never chose anything.
//
// 🪤 Cost of getting this wrong, measured 2026-09-06: the node had ASR_LANG=ru,
// the control plane forwarded "auto", and crispasr ran whisper language-detect
// ahead of the qwen3-asr backend. That combination crashes the worker — exit
// 0xC0000409 from the CLI, "internal error: bad conversion" from the server — on
// EVERY request. The pool dutifully failed over to the CPU whisper at 0.4x
// realtime, so nothing looked broken except that voice replies arrived ~20 s
// late. Pinning the language skips language-detect entirely: same worker, same
// file, 4.5x realtime.
func (e *Engine) langFor(req inference.InterpretRequest) string {
	if l := strings.TrimSpace(req.Language); l != "" && !strings.EqualFold(l, "auto") {
		return l
	}
	return e.cfg.Lang
}

// asrTranscript runs one recognizer for a faithful transcript with the given
// language hint (empty => the recognizer's own default / auto-detect).
func (e *Engine) asrTranscript(ctx context.Context, wav string, rec Recognizer, lang string) (string, error) {
	return e.asrTranscriptLang(ctx, wav, rec, lang)
}

// asrTranscriptLang runs one recognizer with an explicit language hint. When a
// hot `crispasr --server --backend whisper` asr pool has a live worker it routes
// the recognizer there over HTTP (POST /v1/audio/transcriptions) — preferring a
// pool keyed to the recognizer's model, then the default ("") asr pool (where our
// single hot whisper lives). With no live asr pool it cold-spawns the crispasr CLI
// recognizer (the monolith fallback — unchanged).
func (e *Engine) asrTranscriptLang(ctx context.Context, wav string, rec Recognizer, lang string) (string, error) {
	// Hot path: route by rec.Name — the model-keyed asr pool key an admin launch
	// uses (e.g. "voxtral", "whisper-large"). A recognizer whose named pool isn't up
	// falls through pickASRPool to the default ("") pool, so a single-whisper node
	// still serves the whole ensemble; when Voxtral and Whisper are each launched as
	// their own hot pool, this routes each leg to the right model.
	if p := e.pickASRPool(rec.Name); p != nil {
		return e.asrTranscriptServer(ctx, p, wav, rec, lang)
	}
	// CLI fallback (unchanged): cold-spawn the crispasr recognizer on the 16k WAV.
	cmd := exec.CommandContext(ctx, e.cfg.CrispasrBin,
		"-m", rec.Model,
		"--backend", rec.Backend,
		"-l", lang,
		"-f", wav,
	)
	out, err := cmd.Output()
	if err != nil {
		return "", fmt.Errorf("ASR %s failed: %w", rec.Name, err)
	}
	return cleanCLIOutput(out), nil
}

// asrTranscriptServer runs one recognizer against a hot asr pool over HTTP,
// failing over between workers exactly like synthesizeServer / interpretGemmaServer:
// a dead worker (dial failure) is evicted and the request retried elsewhere; an
// alive-but-erroring worker is held out of this call's rotation but kept in the
// pool. Safe under runRecognizers' per-recognizer concurrency: each call checks out
// its own worker. The recognizer's model is a file path (the loaded server already
// owns its model), so the OpenAI `model` form field is left empty; only the
// language hint is forwarded.
func (e *Engine) asrTranscriptServer(ctx context.Context, p *pool, wav string, rec Recognizer, lang string) (string, error) {
	maxAttempts := p.size()
	if maxAttempts > 3 {
		maxAttempts = 3
	}
	if maxAttempts < 1 {
		maxAttempts = 1
	}

	var held []*worker
	defer func() {
		for _, w := range held {
			p.release(w)
		}
	}()

	var lastErr error
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		w, err := p.checkout(ctx)
		if err != nil {
			if lastErr != nil {
				return "", lastErr
			}
			return "", fmt.Errorf("ASR %s: no worker available: %w", rec.Name, err)
		}
		text, err := asrTranscribeServer(ctx, e.httpClient, w.url, e.cfg.TTSServerToken, wav, "", lang)
		if err == nil {
			p.release(w)
			return text, nil
		}
		lastErr = fmt.Errorf("ASR %s: server %s: %w", rec.Name, w.url, err)
		if isDeadWorkerErr(err) {
			log.Printf("[asr pool] evicting dead worker %s (%v); retrying on another worker", w.url, err)
			p.remove(w.url)
		} else {
			log.Printf("[asr pool] worker %s returned an error (%v); retrying on another worker", w.url, err)
			held = append(held, w)
		}
	}
	if lastErr == nil {
		lastErr = fmt.Errorf("ASR %s: no worker available", rec.Name)
	}
	return "", lastErr
}

// runRecognizers transcribes the audio with every configured recognizer in
// parallel and returns the non-empty transcripts (order follows cfg.Recognizers).
func (e *Engine) runRecognizers(ctx context.Context, wav, lang string) []Transcript {
	results := make([]Transcript, len(e.cfg.Recognizers))
	var wg sync.WaitGroup
	for i, rec := range e.cfg.Recognizers {
		wg.Add(1)
		go func(i int, rec Recognizer) {
			defer wg.Done()
			text, err := e.asrTranscript(ctx, wav, rec, lang)
			if err != nil {
				log.Printf("[assisted] %s failed: %v", rec.Name, err)
				return
			}
			log.Printf("[assisted] %s transcript: %d chars", rec.Name, len(text))
			results[i] = Transcript{Name: rec.Name, Text: text}
		}(i, rec)
	}
	wg.Wait()

	out := make([]Transcript, 0, len(results))
	for _, t := range results {
		if strings.TrimSpace(t.Text) != "" {
			out = append(out, t)
		}
	}
	return out
}

// interpretAssisted transcribes with the recognizer ensemble, then hands every
// transcript to Gemma to reconcile. Falls back to audio-only Gemma if no
// recognizer produced a transcript.
func (e *Engine) interpretAssisted(ctx context.Context, wav string, req inference.InterpretRequest) (*inference.InterpretResult, error) {
	transcripts := e.runRecognizers(ctx, wav, e.langFor(req))
	if len(transcripts) == 0 {
		log.Printf("[assisted] no transcripts; falling back to Gemma audio-only")
	}
	return e.interpretGemma(ctx, wav, req, transcripts)
}

// interpretGemma runs Gemma 4 on the WAV, optionally anchored by reference
// transcripts from the ASR ensemble (assisted mode). When a hot `gemma`
// llama-server replica is in the pool it routes to it over the OpenAI chat API
// (audio inline); otherwise it cold-spawns llama-mtmd-cli (the monolith
// fallback — unchanged).
func (e *Engine) interpretGemma(ctx context.Context, wav string, req inference.InterpretRequest, refs []Transcript) (*inference.InterpretResult, error) {
	userPrompt := buildUserPrompt(req, refs)

	// Hot path: a live gemma llama-server keeps the Gemma-4 model + mmproj loaded
	// and understands audio through /v1/chat/completions (verified live). Route to
	// it instead of cold-spawning a CLI that may not even be on PATH.
	if p := e.pickGemmaPool(""); p != nil {
		return e.interpretGemmaServer(ctx, p, wav, userPrompt, refs)
	}

	// CLI fallback: cold-spawn llama-mtmd-cli on the local model/mmproj.
	if e.cfg.GemmaModel == "" || e.cfg.GemmaMMProj == "" {
		return nil, fmt.Errorf("interpret: Gemma model/mmproj not configured")
	}
	cmd := exec.CommandContext(ctx, e.cfg.LlamaMtmdBin,
		"-m", e.cfg.GemmaModel,
		"--mmproj", e.cfg.GemmaMMProj,
		"--audio", wav,
		"--jinja", // Gemma's chat template requires the jinja template engine
		"-sys", systemPrompt,
		"-p", userPrompt,
		"-n", "3072", // room for Gemma's reasoning block PLUS a COMPLETE (non-summarized) answer
		"--temp", "0", // determinism (fidelity mode); matches gemmaChatServer
	)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("interpret: llama-mtmd-cli failed: %w: %s", err, tail(stderr.String()))
	}
	result := finalizeInterpret(string(out), refs)
	log.Printf("[gemma] interpreted %d bytes of output into a %d-char intent (transcripts=%d)", len(out), len(result.Intent), len(refs))
	if result.Intent == "" {
		return nil, fmt.Errorf("interpret: Gemma returned empty output")
	}
	return result, nil
}

// pickGemmaPool returns the hot gemma pool to route an interpret request to, or
// nil to fall back to the CLI. Mirrors pickASRPool: a named model prefers its
// own pool then the default ("") pool; an empty model uses the default pool.
// nil when no gemma pool has a live worker.
func (e *Engine) pickGemmaPool(model string) *pool {
	if p := e.lookupPool("gemma", model); p != nil && p.size() > 0 {
		return p
	}
	if model != "" {
		if p := e.lookupPool("gemma", ""); p != nil && p.size() > 0 {
			return p
		}
	}
	return nil
}

// interpretGemmaServer runs the interpret chat request against the hot gemma
// pool, failing over between workers exactly like synthesizeServer: a dead
// worker (dial failure) is evicted and the request retried elsewhere; an
// alive-but-erroring worker is held out of this call's rotation but kept in the
// pool. The returned assistant content is run through the SAME extractGemmaFinal
// as the CLI path (the hot server may also emit Gemma's `<thought>` block).
func (e *Engine) interpretGemmaServer(ctx context.Context, p *pool, wav, userPrompt string, refs []Transcript) (*inference.InterpretResult, error) {
	maxAttempts := p.size()
	if maxAttempts > 3 {
		maxAttempts = 3
	}
	if maxAttempts < 1 {
		maxAttempts = 1
	}

	var held []*worker
	defer func() {
		for _, w := range held {
			p.release(w)
		}
	}()

	var lastErr error
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		w, err := p.checkout(ctx)
		if err != nil {
			if lastErr != nil {
				return nil, lastErr
			}
			return nil, fmt.Errorf("interpret: no Gemma worker available: %w", err)
		}
		raw, err := gemmaChatServer(ctx, e.httpClient, w.url, e.cfg.TTSServerToken, wav, systemPrompt, userPrompt)
		if err == nil {
			p.release(w)
			return interpretResultFromContent(raw, refs), nil
		}
		lastErr = fmt.Errorf("interpret: gemma server %s: %w", w.url, err)
		if isDeadWorkerErr(err) {
			log.Printf("[gemma pool] evicting dead worker %s (%v); retrying on another worker", w.url, err)
			p.remove(w.url)
		} else {
			log.Printf("[gemma pool] worker %s returned an error (%v); retrying on another worker", w.url, err)
			held = append(held, w)
		}
	}
	if lastErr == nil {
		lastErr = fmt.Errorf("interpret: no Gemma worker available")
	}
	return nil, lastErr
}

// interpretResultFromContent shapes a hot-server assistant message into an
// InterpretResult. It runs the same extractGemmaFinal as the CLI path; if that
// yields nothing (e.g. a bare answer with no channel markers that is somehow all
// whitespace) it falls back to the raw content. Raw carries the exact content
// returned; the Type/assisted-intent logic is identical to the CLI path.
func interpretResultFromContent(raw string, refs []Transcript) *inference.InterpretResult {
	result := finalizeInterpret(raw, refs)
	log.Printf("[gemma pool] interpreted %d bytes of content into a %d-char intent (transcripts=%d)", len(raw), len(result.Intent), len(refs))
	return result
}

// finalizeInterpret builds the InterpretResult from Gemma's RAW output. It first
// pulls the best-effort trailing type/tone/urgency metadata line out (stripMetaLine),
// THEN runs extractGemmaFinal on the remainder — this order matters: extractGemmaFinal's
// no-marker fallback returns the last content line, so a trailing meta line must be
// removed before it, or the tags would masquerade as the answer. Type is the utterance
// classification when the model gave one; empty otherwise (the mode of computation is
// already recorded by the eval log's Mode + the presence of Transcripts).
func finalizeInterpret(raw string, refs []Transcript) *inference.InterpretResult {
	cleaned, typ, tone, urgency := stripMetaLine(raw)
	text := extractGemmaFinal([]byte(cleaned))
	if text == "" {
		text = strings.TrimSpace(cleaned)
	}
	return &inference.InterpretResult{
		Type:        typ,
		Intent:      text,
		Tone:        tone,
		Urgency:     urgency,
		Raw:         raw,
		Transcripts: toRefs(refs),
	}
}

// stripMetaLine removes Gemma's trailing "[[type: … | tone: … | urgency: …]]" metadata
// line from the full model output and returns the cleaned text plus the parsed fields.
// It is DELIBERATELY LENIENT: a small, heavily-quantized Gemma routinely mangles the
// format — wrong brackets, ':' vs '=', '|' vs '·' vs ';', a missing field, or the line
// dropped entirely. We look only at the last couple of non-empty lines, accept a line
// as metadata only if it yields a tone or urgency value (the strong signals — `type`
// alone is too easily confused with prose like "what type of file"), grab `type`
// opportunistically from that same line, and strip just that line. If no meta line is
// found the text is returned unchanged with empty fields.
func stripMetaLine(raw string) (cleaned, typ, tone, urgency string) {
	trimmed := strings.TrimRight(raw, " \t\r\n")
	lines := strings.Split(trimmed, "\n")
	for i := len(lines) - 1; i >= 0 && i >= len(lines)-2; i-- {
		low := strings.ToLower(lines[i])
		if !strings.Contains(low, "tone") && !strings.Contains(low, "urgency") {
			continue
		}
		t := metaTagValue(lines[i], "tone")
		u := metaTagValue(lines[i], "urgency")
		if t == "" && u == "" {
			continue
		}
		ty := normalizeType(metaTagValue(lines[i], "type"))
		rest := strings.Join(append(append([]string{}, lines[:i]...), lines[i+1:]...), "\n")
		return rest, ty, t, u
	}
	return raw, "", "", ""
}

// normalizeType lowercases the parsed utterance type and keeps it only if it is one
// of the documented classes; anything else (a hallucinated or mangled value) is
// dropped to "" rather than surfaced.
func normalizeType(v string) string {
	v = strings.ToLower(strings.TrimSpace(v))
	switch v {
	case "command", "question", "decision", "suggestion", "hypothesis", "uncertain":
		return v
	default:
		return ""
	}
}

// metaTagValue pulls the value following `key` on a loose metadata line, tolerating
// ':' or '=' separators and stopping at the next field/closing delimiter. The key
// must appear on a WORD BOUNDARY so ordinary prose ("ringtone", "tonely") does not
// masquerade as a "tone" tag. Returns "" if no boundary occurrence is found.
func metaTagValue(line, key string) string {
	low := strings.ToLower(line)
	for start := 0; ; {
		rel := strings.Index(low[start:], key)
		if rel < 0 {
			return ""
		}
		idx := start + rel
		if idx == 0 || !isAlnumByte(low[idx-1]) {
			rest := strings.TrimLeft(line[idx+len(key):], " \t:=")
			for i, r := range rest {
				if r == '|' || r == '·' || r == ';' || r == ',' || r == ']' || r == '\n' || r == '⟧' {
					rest = rest[:i]
					break
				}
			}
			return strings.TrimSpace(strings.Trim(strings.TrimSpace(rest), "\"'`[]"))
		}
		start = idx + len(key)
	}
}

func isAlnumByte(b byte) bool {
	return (b >= 'a' && b <= 'z') || (b >= '0' && b <= '9')
}

// toRefs converts internal transcripts to the wire/logging shape.
func toRefs(ts []Transcript) []inference.TranscriptRef {
	if len(ts) == 0 {
		return nil
	}
	out := make([]inference.TranscriptRef, len(ts))
	for i, t := range ts {
		out[i] = inference.TranscriptRef{Name: t.Name, Text: t.Text}
	}
	return out
}

// Health verifies the engine can serve. In hot-server mode it requires at least
// one TTS server in the pool to answer; in CLI mode it checks the binary and
// configured model files are present.
func (e *Engine) Health(ctx context.Context) error {
	if e.anyTTSServers() {
		ws := e.allTTSWorkers()
		for _, w := range ws {
			if ttsHealth(ctx, e.httpClient, w.url, e.cfg.TTSServerToken) == nil {
				return nil
			}
		}
		return fmt.Errorf("no healthy TTS server in pool (%d configured)", len(ws))
	}
	if _, err := exec.LookPath(e.cfg.CrispasrBin); err != nil {
		return fmt.Errorf("crispasr not found: %w", err)
	}
	for _, p := range []string{e.cfg.TTSModel, e.cfg.TTSCodecModel} {
		if p != "" {
			if _, err := os.Stat(p); err != nil {
				return fmt.Errorf("TTS model missing: %w", err)
			}
		}
	}
	return nil
}

// buildUserPrompt assembles the compact context block passed to Gemma. When
// reference transcripts are provided (assisted mode), they are presented as
// candidate outputs from independent speech recognizers so Gemma reconciles them
// for exact wording instead of re-inventing technical terms it may have
// mis-heard from the audio directly.
func buildUserPrompt(req inference.InterpretRequest, refs []Transcript) string {
	var b strings.Builder
	if req.AgentQuestion != "" {
		b.WriteString("AGENT QUESTION:\n")
		b.WriteString(req.AgentQuestion)
		b.WriteString("\n\n")
	}
	if len(req.PreviousSay) > 0 {
		b.WriteString("PREVIOUS SPOKEN UPDATE:\n")
		b.WriteString(strings.Join(req.PreviousSay, "\n"))
		b.WriteString("\n\n")
	}
	if len(refs) > 0 {
		b.WriteString("CANDIDATE TRANSCRIPTS from independent speech recognizers. TRUST them for exact wording — especially technical terms, tool names, and identifiers. Where they disagree, reconcile using the audio; do NOT invent a different word. The audio is mainly for tone and disambiguation.\n")
		for _, t := range refs {
			b.WriteString("- [")
			b.WriteString(t.Name)
			b.WriteString("] ")
			b.WriteString(t.Text)
			b.WriteString("\n")
		}
		b.WriteString("\n")
	}
	b.WriteString("USER AUDIO follows. Return COMPLETE, faithful agent-ready text — preserve everything the user says, do not summarize.")
	return b.String()
}

// systemPrompt is the speech-to-intent interpreter instruction (handoff §14).
// FIDELITY-FIRST (2026-07-11): the earlier "concise, ~100 tokens, do not transcribe
// verbatim" wording made Gemma summarize and DROP content (whole asides, one side of a
// comparison) — see doc/eval-listen-gemma-drops-meaning.md. The rule is now: preserve
// everything the user said; clean recognition noise only; never compress.
// EMOTION (2026-07-11): fold audible tone into the text as inline cues (CAPS for anger/
// swearing, "(laughs)" for laughter, "(emphatic)" for stressed repetition). This is what
// makes voice-in beat plain ASR: users of text-only voice tools fake emotion by hand
// (swearing, typing laughter, repeating themselves) because tone is discarded
// (doc/validation-voice-emotion-need.md). Cues augment the words, never replace them. The
// model also emits a trailing "[[tone: … | urgency: …]]" line, parsed leniently by
// splitEmotionMeta into InterpretResult.Tone/Urgency for the agent-facing marker.
const systemPrompt = `You are a faithful speech-to-intent interpreter for an AI coding agent.
Turn the user's spoken message into clean, COMPLETE, agent-ready text.
FIDELITY IS THE PRIORITY: never summarize, shorten, or drop anything the user actually said.
Rules:
- Preserve EVERY distinct point, question, aside, and observation — including remarks that
  seem minor or off-topic. Do not omit content to be concise.
- Keep the user's actual intent and decisions exactly; do not improve, escalate, or soften them.
- Do not convert suggestions into commands or questions into statements.
- Preserve uncertainty, alternatives, and uncertain recollections.
- When REFERENCE TRANSCRIPTS are provided, treat their wording as authoritative for technical
  terms, tool names, identifiers, and numbers; never replace a term from them with a different
  word. If a technical identifier is unclear, mark it uncertain rather than guessing.
- Do not invent file names, class names, identifiers, paths, or numbers.
- Clean up only recognition noise: filler words, repetitions, and false starts; keep meaningful
  self-corrections. You may reorder into readable sentences, but change no meaning and lose nothing.
- Reply in the same language the user spoke.
- Convey emotional delivery from the AUDIO — hearing the voice is the whole point, not just reading
  a transcript. Fold tone into the text as light cues; a cue must NEVER replace or drop a word the
  user actually said:
  - Anger, frustration, or swearing: RENDER THE HEATED WORDS IN CAPS (keep the words themselves).
  - Laughter: mark it where it happens, e.g. "(laughs)" / "(смеётся)".
  - Emphasis the user makes by repeating a point or raising their voice: keep every repetition and
    note the emphasis, e.g. "(emphatic)".
  - Preserve audible sarcasm, hesitation, and excitement as short parenthetical cues.
  - When delivery is calm and neutral, add no cues — do not invent emotion that is not there.
- Length follows the input: a long spoken message yields a long, complete result. Do not compress.
- AFTER the interpreted text, output ONE final line, by itself, in exactly this form:
  [[type: <command|question|decision|suggestion|hypothesis|uncertain> | tone: <one or two words> | urgency: <low|normal|high>]]
  Choose type by what the utterance DOES; judge tone and urgency from the AUDIO delivery (calm speech
  → tone: neutral | urgency: low). Emit this line exactly once, at the very end, and write nothing
  after it. This line is metadata, not part of the message — do not let it change the interpreted
  text above it.`

// extractGemmaFinal pulls the final answer out of Gemma E4B's output. The model
// emits a "<|channel>thought …" reasoning block and then the final answer after
// a "<channel|>" marker; we return the text after the last such marker. When no
// marker is present (e.g. the answer was cut off) we fall back to the last
// non-empty line.
func extractGemmaFinal(stdout []byte) string {
	s := strings.TrimSpace(string(stdout))
	if idx := strings.LastIndex(s, "<channel|>"); idx >= 0 {
		return strings.TrimSpace(s[idx+len("<channel|>"):])
	}
	s = strings.ReplaceAll(s, "<|channel>", "")
	lines := strings.Split(s, "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		if t := strings.TrimSpace(lines[i]); t != "" {
			return t
		}
	}
	return ""
}

// cleanCLIOutput strips llama.cpp / crispasr log noise and joins the rest.
func cleanCLIOutput(stdout []byte) string {
	lines := strings.Split(string(stdout), "\n")
	var kept []string
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		switch {
		case strings.HasPrefix(line, "ggml_"),
			strings.HasPrefix(line, "llama_"),
			strings.HasPrefix(line, "main:"),
			strings.HasPrefix(line, "mtmd_"),
			strings.HasPrefix(line, "clip_"),
			strings.HasPrefix(line, "build:"),
			strings.HasPrefix(line, "voxtral"),
			strings.HasPrefix(line, "whisper"),
			strings.HasPrefix(line, "crispasr"),
			strings.HasPrefix(line, "Created At"),
			strings.HasPrefix(line, "encoding "):
			continue
		}
		kept = append(kept, line)
	}
	return strings.TrimSpace(strings.Join(kept, " "))
}

// tail returns the last part of s, for compact error messages.
func tail(s string) string {
	s = strings.TrimSpace(s)
	const max = 300
	if len(s) > max {
		return "…" + s[len(s)-max:]
	}
	return s
}
