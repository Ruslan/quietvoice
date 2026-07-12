package local

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// fakeLauncher launches fake crispasr servers (httptest) instead of real
// processes, so the supervisor + admin logic can be tested without a GPU.
type fakeLauncher struct {
	t       *testing.T
	servers []*httptest.Server
	failNth int // if >0, the launch at this 1-based index returns an error
	n       int
}

func (f *fakeLauncher) launch(_ context.Context, role, model string, port int) (*replica, error) {
	f.n++
	if f.failNth > 0 && f.n == f.failNth {
		return nil, errFakeLaunch
	}
	srv := newFakeReplicaServer(f.t, []byte("RIFFx"))
	f.servers = append(f.servers, srv)
	// cmd is nil: replica.kill() is a guarded no-op; httptest servers are closed
	// by their own t.Cleanup. healthPath mirrors what the real launcher sets so
	// waitHealth probes the same path it would in production.
	return &replica{role: role, model: model, url: srv.URL, port: port, healthPath: roleHealthPath(role)}, nil
}

// newFakeReplicaServer stands in for any managed replica (tts/asr/gemma): it
// answers every role's readiness probe (GET /v1/voices AND GET /health) plus the
// tts speech and asr transcription endpoints, so the supervisor + admin logic
// can be tested for a heterogeneous fleet without a GPU.
func newFakeReplicaServer(t *testing.T, wav []byte) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/voices", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"voices": []string{"ded"}})
	})
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	mux.HandleFunc("POST /v1/audio/speech", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "audio/wav")
		_, _ = w.Write(wav)
	})
	mux.HandleFunc("POST /v1/audio/transcriptions", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"text": "стартуют"})
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv
}

var errFakeLaunch = errFake("launch failed")

type errFake string

func (e errFake) Error() string { return string(e) }

// newAdminEngine builds an Engine wired to a fake launcher and a fake VRAM probe.
func newAdminEngine(t *testing.T, freeVRAM int, l *fakeLauncher) *Engine {
	t.Helper()
	cfg := Config{WorkDir: t.TempDir(), TTSMaxReplicas: 4}
	e := New(cfg)
	// Replace the real supervisor with one driven by the fake launcher, and a
	// fast health timeout so a failed launch fails the test quickly.
	e.sup = newSupervisor(cfg, l)
	e.sup.healthTimeout = 2 * time.Second
	// Model live free VRAM the way nvidia-smi reports it: it decreases as managed
	// replicas take their share, so subtract the running fleet's charge from the
	// initial budget. (Admission checks a new replica's cost against this.)
	e.freeVRAMMB = func() int { return freeVRAM - e.sup.vramMB(roleVRAMCostMB) }
	return e
}

func TestSetReplicasScaleUpAndDown(t *testing.T) {
	l := &fakeLauncher{t: t}
	e := newAdminEngine(t, 100000, l)

	if _, err := e.SetReplicas(context.Background(), "tts", "", 3); err != nil {
		t.Fatalf("scale up: %v", err)
	}
	if got := e.ttsPoolSize(""); got != 3 {
		t.Fatalf("pool size after scale up = %d, want 3", got)
	}
	if got := e.ManagedReplicas(); got != 3 {
		t.Fatalf("managed replicas = %d, want 3", got)
	}
	reps := e.Replicas(context.Background())
	for _, r := range reps {
		if !r.Managed || !r.Healthy {
			t.Fatalf("replica %+v: want managed+healthy", r)
		}
	}

	if _, err := e.SetReplicas(context.Background(), "tts", "", 1); err != nil {
		t.Fatalf("scale down: %v", err)
	}
	if got := e.ttsPoolSize(""); got != 1 {
		t.Fatalf("pool size after scale down = %d, want 1", got)
	}
}

func TestSetReplicasVRAMAdmission(t *testing.T) {
	l := &fakeLauncher{t: t}
	e := newAdminEngine(t, 1000, l) // 1000 MB free < 3300 MB (tts) needed

	_, err := e.SetReplicas(context.Background(), "tts", "", 1)
	if err == nil {
		t.Fatal("expected VRAM admission error, got nil")
	}
	if e.ttsPoolSize("") != 0 {
		t.Fatalf("a replica was launched despite VRAM rejection (size=%d)", e.ttsPoolSize(""))
	}
}

func TestSetReplicasUnknownVRAMAdmits(t *testing.T) {
	l := &fakeLauncher{t: t}
	e := newAdminEngine(t, -1, l) // -1 => unknown => best-effort admit

	if _, err := e.SetReplicas(context.Background(), "tts", "", 2); err != nil {
		t.Fatalf("scale up with unknown VRAM: %v", err)
	}
	if e.ttsPoolSize("") != 2 {
		t.Fatalf("pool size = %d, want 2", e.ttsPoolSize(""))
	}
}

func TestSetReplicasMaxCap(t *testing.T) {
	l := &fakeLauncher{t: t}
	e := newAdminEngine(t, 100000, l) // cap is 4

	if _, err := e.SetReplicas(context.Background(), "tts", "", 5); err == nil {
		t.Fatal("expected cap error for 5 > max 4, got nil")
	}
	if e.ttsPoolSize("") != 0 {
		t.Fatalf("replicas launched despite cap rejection (size=%d)", e.ttsPoolSize(""))
	}
}

func TestSetReplicasRoleValidation(t *testing.T) {
	l := &fakeLauncher{t: t}
	e := newAdminEngine(t, 100000, l)

	if _, err := e.SetReplicas(context.Background(), "banana", "", 1); err == nil {
		t.Fatal("expected error for unknown role 'banana', got nil")
	}
	// aliases normalize to their canonical role
	if _, err := e.SetReplicas(context.Background(), "qwen-tts", "", 1); err != nil {
		t.Fatalf("alias qwen-tts should be accepted: %v", err)
	}
	if _, err := e.SetReplicas(context.Background(), "whisper", "", 1); err != nil {
		t.Fatalf("alias whisper (asr) should be accepted: %v", err)
	}
	if got := e.ManagedReplicasFor("asr", ""); got != 1 {
		t.Fatalf("asr replicas via alias = %d, want 1", got)
	}
}

// TestSetReplicasMultiRole brings up a heterogeneous fleet (tts + asr + gemma) on
// one supervisor, lists each with its role, and scales a single role down without
// touching the others.
func TestSetReplicasMultiRole(t *testing.T) {
	l := &fakeLauncher{t: t}
	e := newAdminEngine(t, 100000, l)

	// Mirror POST /admin/replicas {"tts":1,"asr":1,"gemma":1}.
	for role, n := range map[string]int{"tts": 1, "asr": 1, "gemma": 1} {
		if _, err := e.SetReplicas(context.Background(), role, "", n); err != nil {
			t.Fatalf("SetReplicas(%s): %v", role, err)
		}
	}
	if got := e.ManagedReplicas(); got != 3 {
		t.Fatalf("managed replicas = %d, want 3", got)
	}
	roles := map[string]int{}
	for _, r := range e.Replicas(context.Background()) {
		if !r.Managed || !r.Healthy {
			t.Fatalf("replica %+v: want managed+healthy", r)
		}
		roles[r.Role]++
	}
	for _, role := range []string{"tts", "asr", "gemma"} {
		if roles[role] != 1 {
			t.Fatalf("listing has %d %s replicas, want 1 (roles=%v)", roles[role], role, roles)
		}
	}

	// Scale asr down; tts and gemma untouched.
	if _, err := e.SetReplicas(context.Background(), "asr", "", 0); err != nil {
		t.Fatalf("scale asr down: %v", err)
	}
	if got := e.ManagedReplicasFor("asr", ""); got != 0 {
		t.Fatalf("asr after scale-down = %d, want 0", got)
	}
	if got := e.ManagedReplicasFor("tts", ""); got != 1 {
		t.Fatalf("tts should be untouched = %d, want 1", got)
	}
	if got := e.ManagedReplicasFor("gemma", ""); got != 1 {
		t.Fatalf("gemma should be untouched = %d, want 1", got)
	}
}

// TestSetReplicasPerRoleVRAMSum proves admission sums the whole managed fleet's
// VRAM (across roles) against free VRAM: asr (2000) fits in 8000, but adding
// gemma (7000) would make 9000 > 8000 and is rejected.
func TestSetReplicasPerRoleVRAMSum(t *testing.T) {
	l := &fakeLauncher{t: t}
	e := newAdminEngine(t, 8000, l)

	if _, err := e.SetReplicas(context.Background(), "asr", "", 1); err != nil {
		t.Fatalf("asr (2000 MB) should fit in 8000 MB free: %v", err)
	}
	if _, err := e.SetReplicas(context.Background(), "gemma", "", 1); err == nil {
		t.Fatal("expected VRAM rejection: 2000 (asr) + 7000 (gemma) > 8000 free")
	}
	if got := e.ManagedReplicasFor("gemma", ""); got != 0 {
		t.Fatalf("gemma replica launched despite VRAM rejection (n=%d)", got)
	}
	if got := e.ManagedReplicas(); got != 1 {
		t.Fatalf("managed replicas = %d, want 1 (only asr)", got)
	}
}
