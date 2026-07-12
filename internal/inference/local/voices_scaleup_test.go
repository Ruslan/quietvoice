package local

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"
)

// recordingReplica is a fake crispasr tts server that records which voices were
// registered on it via POST /v1/voices, so a test can prove scale-up replay.
type recordingReplica struct {
	srv *httptest.Server
	mu  sync.Mutex
	got map[string]bool
}

func (r *recordingReplica) has(name string) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.got[name]
}

// recordingLauncher launches recordingReplica servers instead of real processes.
type recordingLauncher struct {
	t       *testing.T
	servers []*recordingReplica
}

func (l *recordingLauncher) launch(_ context.Context, role, model string, port int) (*replica, error) {
	rr := &recordingReplica{got: map[string]bool{}}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(http.StatusOK) })
	mux.HandleFunc("GET /v1/voices", func(w http.ResponseWriter, r *http.Request) {
		rr.mu.Lock()
		names := make([]string, 0, len(rr.got))
		for n := range rr.got {
			names = append(names, n)
		}
		rr.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"voices": names})
	})
	mux.HandleFunc("POST /v1/voices", func(w http.ResponseWriter, r *http.Request) {
		_ = r.ParseMultipartForm(1 << 20)
		rr.mu.Lock()
		rr.got[r.FormValue("name")] = true
		rr.mu.Unlock()
		w.WriteHeader(http.StatusCreated)
	})
	srv := httptest.NewServer(mux)
	l.t.Cleanup(srv.Close)
	rr.srv = srv
	l.servers = append(l.servers, rr)
	return &replica{role: role, model: model, url: srv.URL, port: port, healthPath: roleHealthPath(role)}, nil
}

// TestScaleUpProvisionsVoicesToNewReplica proves a tts replica scaled up AFTER a
// voice was registered is auto-given the same voices as its neighbors (crispasr
// caches per-process, so without replay the new replica would be voice-less).
func TestScaleUpProvisionsVoicesToNewReplica(t *testing.T) {
	ctx := context.Background()
	rl := &recordingLauncher{t: t}
	cfg := Config{WorkDir: t.TempDir(), TTSMaxReplicas: 4}
	e := New(cfg)
	e.sup = newSupervisor(cfg, rl)
	e.sup.healthTimeout = 2 * time.Second
	e.freeVRAMMB = func() int { return 1 << 20 } // plenty of VRAM

	if _, err := e.SetReplicas(ctx, "tts", "", 1); err != nil {
		t.Fatalf("scale tts 0->1: %v", err)
	}
	if _, err := e.UploadVoice(ctx, "ded", "reference transcript", []byte("RIFFvoice"), "ded.wav"); err != nil {
		t.Fatalf("upload voice: %v", err)
	}
	if !rl.servers[0].has("ded") {
		t.Fatal("replica #1 did not receive the voice via the upload fan-out")
	}

	// Scale up: the new replica #2 must be auto-provisioned with "ded".
	if _, err := e.SetReplicas(ctx, "tts", "", 2); err != nil {
		t.Fatalf("scale tts 1->2: %v", err)
	}
	if len(rl.servers) != 2 {
		t.Fatalf("want 2 launched replicas, got %d", len(rl.servers))
	}
	if !rl.servers[1].has("ded") {
		t.Fatal("new replica #2 was NOT provisioned with voice 'ded' on scale-up (the bug)")
	}
}

// TestScaleUpNoVoicesNoReplay proves the replay is a no-op when nothing has been
// registered yet (a fresh fabric boot must not error or block).
func TestScaleUpNoVoicesNoReplay(t *testing.T) {
	ctx := context.Background()
	rl := &recordingLauncher{t: t}
	cfg := Config{WorkDir: t.TempDir(), TTSMaxReplicas: 4}
	e := New(cfg)
	e.sup = newSupervisor(cfg, rl)
	e.sup.healthTimeout = 2 * time.Second
	e.freeVRAMMB = func() int { return 1 << 20 }

	if _, err := e.SetReplicas(ctx, "tts", "", 2); err != nil {
		t.Fatalf("scale tts 0->2 with no voices: %v", err)
	}
	for i, s := range rl.servers {
		if s.has("ded") {
			t.Fatalf("replica #%d unexpectedly has a voice with none registered", i)
		}
	}
}
