package local

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
)

// replica is one supervisor-launched inference server process.
type replica struct {
	role       string
	model      string // model name this replica serves ("" = default pool)
	url        string
	port       int
	pid        int
	cmd        *exec.Cmd
	healthPath string // readiness probe path for this role (e.g. /v1/voices, /health)
	startedAt  time.Time
}

// kill terminates the replica's process (SIGTERM, then SIGKILL after a grace
// period). Best-effort: a nil/exited process is a no-op.
func (r *replica) kill() {
	if r.cmd == nil || r.cmd.Process == nil {
		return
	}
	_ = r.cmd.Process.Signal(syscall.SIGTERM)
	done := make(chan struct{})
	go func() { _ = r.cmd.Wait(); close(done) }()
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		_ = r.cmd.Process.Kill()
		<-done
	}
}

// launcher starts one server process for a (role, model) and returns its handle.
// It is an interface so tests can substitute a fake that points at an httptest
// server, exercising the supervisor without a real crispasr binary or a GPU.
type launcher interface {
	launch(ctx context.Context, role, model string, port int) (*replica, error)
}

// supervisor owns the lifecycle of managed replicas: it allocates ports, starts
// processes via the launcher, waits until each is healthy, and can stop them.
// It is the process-management half of the admin API; the pool is the routing
// half. Kept minimal (launch / stop-last / stop-all), not an autoscaler.
type supervisor struct {
	mu       sync.Mutex
	l        launcher
	replicas []*replica // in launch order; stopLast pops the newest
	nextPort int

	client        *http.Client
	token         string
	healthTimeout time.Duration
}

func newSupervisor(cfg Config, l launcher) *supervisor {
	return &supervisor{
		l:             l,
		nextPort:      cfg.TTSBasePort,
		client:        &http.Client{Timeout: 5 * time.Second},
		token:         cfg.TTSServerToken,
		healthTimeout: 3 * time.Minute, // a cold crispasr server can take a while to load the model
	}
}

// total reports the number of currently managed replicas across all models.
func (s *supervisor) total() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.replicas)
}

// vramMB sums the VRAM cost of every currently managed replica (across all roles
// and models), using cost() to price each replica by role. Used for admission so
// a scale-up is rejected when the running fleet plus the new replica would not
// fit in free VRAM (see SetReplicas).
func (s *supervisor) vramMB(cost func(role string) int) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	total := 0
	for _, r := range s.replicas {
		total += cost(r.role)
	}
	return total
}

// count reports how many managed replicas serve the given (role, model).
func (s *supervisor) count(role, model string) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	n := 0
	for _, r := range s.replicas {
		if r.role == role && r.model == model {
			n++
		}
	}
	return n
}

// launch starts one replica for (role, model), waits until it is healthy,
// records it, and returns it. On health failure the process is killed and not
// recorded.
func (s *supervisor) launch(ctx context.Context, role, model string) (*replica, error) {
	s.mu.Lock()
	port := s.nextPort
	s.nextPort++
	s.mu.Unlock()

	r, err := s.l.launch(ctx, role, model, port)
	if err != nil {
		return nil, err
	}
	if err := s.waitHealth(ctx, r.url, r.healthPath); err != nil {
		r.kill()
		return nil, fmt.Errorf("replica on port %d: %w", port, err)
	}
	s.mu.Lock()
	s.replicas = append(s.replicas, r)
	s.mu.Unlock()
	return r, nil
}

// stopLast kills and forgets the most recently launched replica for
// (role, model), returning it (nil if none remain).
func (s *supervisor) stopLast(role, model string) *replica {
	s.mu.Lock()
	idx := -1
	for i := len(s.replicas) - 1; i >= 0; i-- {
		if s.replicas[i].role == role && s.replicas[i].model == model {
			idx = i
			break
		}
	}
	if idx < 0 {
		s.mu.Unlock()
		return nil
	}
	r := s.replicas[idx]
	s.replicas = append(s.replicas[:idx], s.replicas[idx+1:]...)
	s.mu.Unlock()
	r.kill()
	return r
}

// stopAll kills every managed replica (used on shutdown).
func (s *supervisor) stopAll() {
	s.mu.Lock()
	rs := s.replicas
	s.replicas = nil
	s.mu.Unlock()
	for _, r := range rs {
		r.kill()
	}
}

// waitHealth polls the replica's role-appropriate readiness probe (path) until it
// passes or the timeout (or ctx) elapses. path is the replica's healthPath
// (/v1/voices for tts, /health for asr/gemma).
func (s *supervisor) waitHealth(ctx context.Context, base, path string) error {
	if path == "" {
		path = "/v1/voices"
	}
	deadline := time.Now().Add(s.healthTimeout)
	for {
		if healthGet(ctx, s.client, joinURL(base, path), s.token) == nil {
			return nil
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("not healthy within %s", s.healthTimeout)
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(1 * time.Second):
		}
	}
}

// launchPlan is the resolved recipe for starting one replica of a given role:
// which binary, with which args and environment, and how to probe it for
// readiness. Built by launchSpec, executed by roleLauncher.
type launchPlan struct {
	bin        string
	args       []string
	env        []string // nil = inherit the inferenced process environment
	healthPath string
}

// roleLauncher is the production launcher: given a role it spawns the right
// server binary with the right args/env (this now owns the consecutive-port
// launching that the old scripts/tts-pool.sh used to do client-side). One
// launcher serves a heterogeneous fleet — tts + asr (crispasr) and gemma
// (llama-server) — on one GPU.
type roleLauncher struct {
	cfg Config
}

// launchSpec resolves a (role, model) to its launch recipe. Roles:
//   - tts:   `crispasr --server --backend qwen3-tts -m MODEL --codec-model CODEC
//     [--voice-dir DIR]`; probes /v1/voices. (Unchanged from the old tts-only path.)
//   - asr:   `crispasr --server --backend whisper -m MODEL`; probes /health.
//     No codec, no voice-dir — it serves /v1/audio/transcriptions only.
//   - gemma: `llama-server -m MODEL --mmproj MMPROJ`; probes /health. Serves
//     /v1/chat/completions with audio-in. llama-server needs its OWN
//     LD_LIBRARY_PATH (GemmaLibDir), distinct from crispasr's, so cfg.env is set.
func launchSpec(cfg Config, role, model string, port int) (launchPlan, error) {
	ps := strconv.Itoa(port)
	switch role {
	case "tts":
		// Resolve the model name to a gguf path: a declared TTSModelPaths entry wins,
		// otherwise fall back to the single configured TTSModel (single-model node).
		modelPath := cfg.TTSModel
		if p, ok := cfg.TTSModelPaths[model]; ok && p != "" {
			modelPath = p
		}
		if modelPath == "" || cfg.TTSCodecModel == "" {
			return launchPlan{}, fmt.Errorf("cannot launch: TTS model/codec not configured for model %q", model)
		}
		args := []string{
			"--server",
			"--backend", cfg.TTSBackend,
			"-m", modelPath,
			"--codec-model", cfg.TTSCodecModel,
			"--host", "127.0.0.1",
			"--port", ps,
		}
		if cfg.TTSVoiceDir != "" {
			// crispasr needs the dir to exist to store uploaded voices; create it so a
			// fresh node's first POST /v1/voices doesn't 400 on a missing directory.
			if err := os.MkdirAll(cfg.TTSVoiceDir, 0o755); err != nil {
				return launchPlan{}, fmt.Errorf("cannot create TTS voice-dir %q: %w", cfg.TTSVoiceDir, err)
			}
			args = append(args, "--voice-dir", cfg.TTSVoiceDir)
		} else {
			log.Printf("[supervisor] WARNING: launching tts with no --voice-dir (TTS_VOICE_DIR unset and WorkDir empty); POST /v1/voices will 400 and `say` will break end-to-end")
		}
		return launchPlan{bin: cfg.TTSServerBin, args: args, env: envWithLibDir(cfg.TTSLibDir), healthPath: ttsHealthPath}, nil
	case "asr":
		// A model-keyed asr replica (model != "") resolves its gguf path AND crispasr
		// backend from the recognizer registry by name, so Voxtral (voxtral4b) and
		// Whisper (whisper) can run as SEPARATE hot pools — the genuine assisted
		// ensemble on a hot fabric. The default ("") pool keeps the single global
		// ASR model/backend (backward-compatible single-whisper node).
		modelPath := cfg.ASRModel
		backend := cfg.ASRBackend
		if model != "" {
			rec, ok := recognizerByName(cfg.Recognizers, model)
			if !ok {
				return launchPlan{}, fmt.Errorf("cannot launch asr replica %q: no recognizer with that name (set VOXTRAL_MODEL/WHISPER_MODEL or ASR_RECOGNIZERS)", model)
			}
			modelPath = rec.Model
			if rec.Backend != "" {
				backend = rec.Backend
			}
		}
		if modelPath == "" {
			return launchPlan{}, fmt.Errorf("cannot launch: ASR model not configured for %q", model)
		}
		args := []string{
			"--server",
			"--backend", backend,
			"-m", modelPath,
			"--host", "127.0.0.1",
			"--port", ps,
		}
		return launchPlan{bin: cfg.ASRServerBin, args: args, env: envWithLibDir(cfg.ASRLibDir), healthPath: asrHealthPath}, nil
	case "gemma":
		if cfg.GemmaModel == "" || cfg.GemmaMMProj == "" {
			return launchPlan{}, fmt.Errorf("cannot launch: Gemma model/mmproj not configured")
		}
		args := []string{
			"-m", cfg.GemmaModel,
			"--mmproj", cfg.GemmaMMProj,
			"-ngl", "999", // offload all layers to the GPU (llama.cpp caps at what fits);
			// without this llama.cpp defaults to CPU-only and a 12B bf16 brain runs on CPU.
			"--ctx-size", envOr("GEMMA_CTX_SIZE", "8192"), // room for system prompt + say-context + audio +
			// Gemma's reasoning block. On a big-VRAM node bump GEMMA_CTX_SIZE (e.g. 32768): the
			// assisted ensemble with 2 long ASR refs can make Gemma reason past a small ctx and
			// return empty (verified on MI300X 2026-07-12). KV is cheap at 192 GB.
			// Cap the thinking budget so reasoning ALWAYS terminates and an answer is emitted.
			// -1 = unrestricted (default; fine for 1 ASR ref). With 2 refs (voxtral+whisper)
			// Gemma spiralled to 58k chars of reasoning_content, hit max_tokens, returned EMPTY
			// (verified on MI300X 2026-07-12). A positive budget (e.g. 2048) keeps thinking but
			// forces a wrap-up. See GEMMA_REASONING_BUDGET in deploy/.env.demo-example.
			"--reasoning-budget", envOr("GEMMA_REASONING_BUDGET", "-1"),
			"--host", "127.0.0.1",
			"--port", ps,
		}
		return launchPlan{bin: cfg.GemmaServerBin, args: args, env: envWithLibDir(cfg.GemmaLibDir), healthPath: gemmaHealthPath}, nil
	default:
		return launchPlan{}, fmt.Errorf("unknown replica role %q (want tts, asr, or gemma)", role)
	}
}

// envOr returns the environment value for key, or def when unset/empty.
func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func (c *roleLauncher) launch(_ context.Context, role, model string, port int) (*replica, error) {
	plan, err := launchSpec(c.cfg, role, model, port)
	if err != nil {
		return nil, err
	}
	// Deliberately NOT exec.CommandContext: the server must outlive the admin
	// request that started it. Lifecycle is owned by the supervisor via kill().
	cmd := exec.Command(plan.bin, plan.args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	// A non-nil env carries this role's own LD_LIBRARY_PATH (e.g. gemma's
	// llama-server dir vs crispasr's dir). Nil means inherit inferenced's env.
	if plan.env != nil {
		cmd.Env = plan.env
	}
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("start %s (role %s): %w", plan.bin, role, err)
	}
	return &replica{
		role:       role,
		model:      model,
		url:        fmt.Sprintf("http://127.0.0.1:%d", port),
		port:       port,
		pid:        cmd.Process.Pid,
		cmd:        cmd,
		healthPath: plan.healthPath,
		startedAt:  time.Now(),
	}, nil
}

// envWithLibDir returns the process environment with LD_LIBRARY_PATH set to dir
// (any inherited LD_LIBRARY_PATH is replaced, not appended). Returns nil when dir
// is empty, so the caller inherits inferenced's environment unchanged — which
// keeps the tts launch byte-for-byte identical to the previous behavior.
func envWithLibDir(dir string) []string {
	if dir == "" {
		return nil
	}
	base := os.Environ()
	out := make([]string, 0, len(base)+1)
	for _, kv := range base {
		if strings.HasPrefix(kv, "LD_LIBRARY_PATH=") {
			continue
		}
		out = append(out, kv)
	}
	return append(out, "LD_LIBRARY_PATH="+dir)
}
