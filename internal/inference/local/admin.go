package local

import (
	"context"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
)

// Per-replica VRAM cost by role (MiB), measured targets used for admission so a
// scale-up is rejected instead of OOM-ing the GPU:
//   - tts:   qwen3-tts 1.7b ≈ 3304 MiB (the old 2100 undershot the real cost).
//   - asr:   whisper-large ≈ 2000 MiB.
//   - gemma: Gemma E4B Q4 ≈ 7000 MiB.
// VRAM estimation for admission check.
const (
	ttsVRAMCostMB   = 3300
	asrVRAMCostMB   = 2000
	gemmaVRAMCostMB = 7000
)

// roleVRAMCostMB prices one replica of a role for VRAM admission.
func roleVRAMCostMB(role string) int {
	switch role {
	case "asr":
		return asrVRAMCostMB
	case "gemma":
		return gemmaVRAMCostMB
	default:
		return ttsVRAMCostMB
	}
}

// ReplicaStatus is one entry in the admin `GET /admin/replicas` listing.
type ReplicaStatus struct {
	Role    string `json:"role"`
	Model   string `json:"model,omitempty"` // "" = default pool
	URL     string `json:"url"`
	PID     int    `json:"pid,omitempty"`
	Managed bool   `json:"managed"` // false = external URL from config (cannot be scaled down)
	Healthy bool   `json:"healthy"`
}

// ManagedReplicas reports how many replicas the supervisor is running in total
// (across all models).
func (e *Engine) ManagedReplicas() int {
	if e.sup == nil {
		return 0
	}
	return e.sup.total()
}

// ManagedReplicasFor reports how many replicas of a (role, model) are managed.
func (e *Engine) ManagedReplicasFor(role, model string) int {
	if e.sup == nil {
		return 0
	}
	canon, err := normalizeRole(role)
	if err != nil {
		return 0
	}
	return e.sup.count(canon, model)
}

// normalizeRole maps accepted role aliases to the canonical managed roles:
// tts (crispasr qwen3-tts), asr (crispasr whisper), gemma (llama-server).
func normalizeRole(role string) (string, error) {
	switch strings.ToLower(strings.TrimSpace(role)) {
	case "tts", "qwen-tts", "qwen_tts":
		return "tts", nil
	case "asr", "stt", "whisper":
		return "asr", nil
	case "gemma", "llm", "interpret":
		return "gemma", nil
	default:
		return "", fmt.Errorf("unknown replica role %q (want tts, asr, or gemma)", role)
	}
}

// SetReplicas scales the managed replica count for a (role, model) to exactly n,
// launching or stopping servers as needed and (de)registering them in that
// (role, model) pool (an empty model = the default pool). role is tts, asr, or
// gemma. External URLs from config are never touched. Newly launched replicas
// are admitted only if the GPU has room: each new replica's VRAM cost must fit
// in currently-free VRAM (best-effort; running replicas already count as used).
func (e *Engine) SetReplicas(ctx context.Context, role, model string, n int) ([]ReplicaStatus, error) {
	canon, err := normalizeRole(role)
	if err != nil {
		return nil, err
	}
	if e.sup == nil {
		return nil, fmt.Errorf("replica supervision is not enabled")
	}
	if n < 0 {
		n = 0
	}
	cur := e.sup.count(canon, model)
	// The cap is on the total managed replicas across all roles/models.
	if delta := n - cur; delta > 0 && e.sup.total()+delta > e.cfg.TTSMaxReplicas {
		return nil, fmt.Errorf("requested %d %s replicas of model %q would exceed MAX_REPLICAS %d (running %d)", n, canon, model, e.cfg.TTSMaxReplicas, e.sup.total())
	}

	p := e.ensurePool(canon, model)
	cost := roleVRAMCostMB(canon)
	for i := cur; i < n; i++ { // scale up
		// Admission checks the NEW replica's VRAM cost against what is actually
		// free. nvidia-smi "free" already accounts for the running replicas' usage,
		// so we must NOT re-add the running fleet's charge here (that double-counts
		// and falsely rejects). Best-effort: free < 0 (unknown) ⇒ admit.
		if free := e.freeVRAMMB(); free >= 0 && cost > free {
			return e.Replicas(ctx), fmt.Errorf("insufficient VRAM for another %s replica: need %d MB, only %d MB free", canon, cost, free)
		}
		r, err := e.sup.launch(ctx, canon, model)
		if err != nil {
			return e.Replicas(ctx), fmt.Errorf("launch %s replica: %w", canon, err)
		}
		p.add(&worker{url: r.url, role: canon, model: model, managed: true, pid: r.pid})
	}
	for i := cur; i > n; i-- { // scale down
		r := e.sup.stopLast(canon, model)
		if r == nil {
			break
		}
		p.remove(r.url)
	}
	return e.Replicas(ctx), nil
}

// Replicas returns the current pool membership across all roles and models (both
// managed and external workers) with a live, role-appropriate health probe for
// each (tts -> /v1/voices, asr/gemma -> /health).
func (e *Engine) Replicas(ctx context.Context) []ReplicaStatus {
	ws := e.allWorkers()
	out := make([]ReplicaStatus, 0, len(ws))
	for _, w := range ws {
		role := firstNonEmpty(w.role, "tts")
		healthy := healthGet(ctx, e.httpClient, joinURL(w.url, roleHealthPath(role)), e.cfg.TTSServerToken) == nil
		out = append(out, ReplicaStatus{
			Role:    role,
			Model:   w.model,
			URL:     w.url,
			PID:     w.pid,
			Managed: w.managed,
			Healthy: healthy,
		})
	}
	return out
}

// Shutdown stops all managed replicas. Call on process exit so we don't leak
// crispasr servers.
func (e *Engine) Shutdown() {
	if e.sup != nil {
		e.sup.stopAll()
	}
}

// queryFreeVRAMMB returns free GPU memory in MB, or -1 if it can't be determined
// (in which case admission is skipped — best-effort). Tries NVIDIA first, then
// ROCm/AMD.
func queryFreeVRAMMB() int {
	if out, err := exec.Command("nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits").Output(); err == nil {
		if v := firstIntField(string(out)); v >= 0 {
			return v
		}
	}
	// ROCm: `rocm-smi --showmeminfo vram --csv` reports total+used VRAM in bytes.
	if out, err := exec.Command("rocm-smi", "--showmeminfo", "vram", "--csv").Output(); err == nil {
		if free := rocmFreeVRAMMB(string(out)); free >= 0 {
			return free
		}
	}
	return -1
}

// firstIntField parses the first integer token from text (e.g. nvidia-smi's
// per-GPU "MiB free" line). Returns -1 if none.
func firstIntField(text string) int {
	for _, line := range strings.Split(text, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		if n, err := strconv.Atoi(strings.Fields(line)[0]); err == nil {
			return n
		}
	}
	return -1
}

// rocmFreeVRAMMB parses `rocm-smi --showmeminfo vram --csv`, which has a header
// row plus one row per GPU with "VRAM Total Memory (B)" and "VRAM Total Used
// Memory (B)" columns. Returns free MB for the first GPU, or -1 if unparseable.
func rocmFreeVRAMMB(csv string) int {
	lines := strings.Split(strings.TrimSpace(csv), "\n")
	if len(lines) < 2 {
		return -1
	}
	header := strings.Split(lines[0], ",")
	totalCol, usedCol := -1, -1
	for i, h := range header {
		h = strings.ToLower(strings.TrimSpace(h))
		switch {
		case strings.Contains(h, "total") && strings.Contains(h, "used"):
			usedCol = i
		case strings.Contains(h, "total") && strings.Contains(h, "memory"):
			totalCol = i
		}
	}
	if totalCol < 0 || usedCol < 0 {
		return -1
	}
	fields := strings.Split(lines[1], ",")
	if totalCol >= len(fields) || usedCol >= len(fields) {
		return -1
	}
	total, err1 := strconv.ParseInt(strings.TrimSpace(fields[totalCol]), 10, 64)
	used, err2 := strconv.ParseInt(strings.TrimSpace(fields[usedCol]), 10, 64)
	if err1 != nil || err2 != nil || total <= 0 {
		return -1
	}
	return int((total - used) / (1024 * 1024))
}

func firstNonEmpty(a, b string) string {
	if a != "" {
		return a
	}
	return b
}
