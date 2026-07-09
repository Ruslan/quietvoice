package local

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"quietvoice/internal/inference"
)

// newFakeTTSServer stands in for a `crispasr --server` instance: it answers the
// readiness probe (GET /v1/voices) and returns the given WAV bytes for
// POST /v1/audio/speech.
func newFakeTTSServer(t *testing.T, wav []byte) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/voices", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"voices": []string{"ded"}})
	})
	mux.HandleFunc("POST /v1/audio/speech", func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, "bad json", http.StatusBadRequest)
			return
		}
		if body["input"] == "" || body["input"] == nil {
			http.Error(w, "empty input", http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "audio/wav")
		_, _ = w.Write(wav)
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv
}

func TestTTSSpeakAndHealth(t *testing.T) {
	want := []byte("RIFFfake-wav-bytes")
	srv := newFakeTTSServer(t, want)
	client := srv.Client()

	if err := ttsHealth(context.Background(), client, srv.URL, ""); err != nil {
		t.Fatalf("ttsHealth: %v", err)
	}
	got, err := ttsSpeak(context.Background(), client, srv.URL, "", "ded", "hello")
	if err != nil {
		t.Fatalf("ttsSpeak: %v", err)
	}
	if string(got) != string(want) {
		t.Fatalf("ttsSpeak returned %q, want %q", got, want)
	}
}

// deadWorkerURL returns the URL of a server that has been shut down, so any dial
// to it is refused (stands in for an OOM'd `crispasr --server` process).
func deadWorkerURL(t *testing.T) string {
	t.Helper()
	srv := httptest.NewServer(http.NewServeMux())
	url := strings.TrimRight(srv.URL, "/")
	srv.Close()
	return url
}

// TestSynthesizeFailsOverAndEvictsDeadWorker proves that when the FIRST
// checked-out worker is dead (dial failure), Synthesize fails over to a healthy
// worker AND the dead worker is evicted so it is never handed out again.
func TestSynthesizeFailsOverAndEvictsDeadWorker(t *testing.T) {
	want := []byte("RIFFhealthy-wav")
	healthy := newFakeTTSServer(t, want)
	deadURL := deadWorkerURL(t)
	healthyURL := strings.TrimRight(healthy.URL, "/")

	dir := t.TempDir()
	e := New(Config{WorkDir: dir, TTSVoice: "ded"})
	e.httpClient = healthy.Client()
	// Add the dead worker FIRST so the FIFO idle queue hands it out first.
	p := e.ensureTTSPool("")
	p.add(&worker{url: deadURL, role: "tts"})
	p.add(&worker{url: healthyURL, role: "tts"})

	res, err := e.Synthesize(context.Background(), inference.SynthesizeRequest{Text: "hello"})
	if err != nil {
		t.Fatalf("Synthesize should fail over to the healthy worker, got: %v", err)
	}
	defer os.Remove(res.WavPath)
	got, err := os.ReadFile(res.WavPath)
	if err != nil {
		t.Fatalf("read wav: %v", err)
	}
	if string(got) != string(want) {
		t.Fatalf("synthesized wav = %q, want %q", got, want)
	}

	// The dead worker must be gone from the pool.
	if p.size() != 1 {
		t.Fatalf("pool size = %d after eviction, want 1", p.size())
	}
	for _, u := range p.urls() {
		if u == deadURL {
			t.Fatalf("dead worker %q still in pool after eviction", deadURL)
		}
	}
	// Subsequent checkouts must never return the evicted worker.
	for i := 0; i < 3; i++ {
		w, err := p.checkout(context.Background())
		if err != nil {
			t.Fatalf("checkout %d: %v", i, err)
		}
		if w.url == deadURL {
			t.Fatalf("checkout returned the evicted dead worker %q", deadURL)
		}
		p.release(w)
	}
}

// TestSynthesizeRetriesNon2xxWithoutEvict proves that an alive worker returning a
// non-2xx response is retried on another worker but NOT evicted from the pool.
func TestSynthesizeRetriesNon2xxWithoutEvict(t *testing.T) {
	want := []byte("RIFFgood-wav")
	good := newFakeTTSServer(t, want)
	goodURL := strings.TrimRight(good.URL, "/")

	// A server that is alive (answers /v1/voices) but 500s on speech.
	badMux := http.NewServeMux()
	badMux.HandleFunc("GET /v1/voices", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"voices": []string{"ded"}})
	})
	badMux.HandleFunc("POST /v1/audio/speech", func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "boom", http.StatusInternalServerError)
	})
	bad := httptest.NewServer(badMux)
	t.Cleanup(bad.Close)
	badURL := strings.TrimRight(bad.URL, "/")

	dir := t.TempDir()
	e := New(Config{WorkDir: dir, TTSVoice: "ded"})
	e.httpClient = good.Client()
	p := e.ensureTTSPool("")
	p.add(&worker{url: badURL, role: "tts"}) // checked out first
	p.add(&worker{url: goodURL, role: "tts"})

	res, err := e.Synthesize(context.Background(), inference.SynthesizeRequest{Text: "hello"})
	if err != nil {
		t.Fatalf("Synthesize should retry past the 500 worker, got: %v", err)
	}
	defer os.Remove(res.WavPath)

	// The bad (but alive) worker must remain in the pool.
	if p.size() != 2 {
		t.Fatalf("pool size = %d, want 2 (alive worker must not be evicted)", p.size())
	}
	foundBad := false
	for _, u := range p.urls() {
		if u == badURL {
			foundBad = true
		}
	}
	if !foundBad {
		t.Fatalf("alive worker %q was wrongly evicted on a non-2xx response", badURL)
	}
}

func TestSynthesizeServerMode(t *testing.T) {
	want := []byte("RIFFserver-wav")
	srv := newFakeTTSServer(t, want)
	dir := t.TempDir()

	e := New(Config{WorkDir: dir, TTSServerURLs: []string{srv.URL}, TTSVoice: "ded"})
	e.httpClient = srv.Client() // use the test server's client

	if e.ttsPoolSize("") != 1 {
		t.Fatalf("pool size = %d, want 1 (server mode not engaged)", e.ttsPoolSize(""))
	}
	if err := e.Health(context.Background()); err != nil {
		t.Fatalf("Health (server mode): %v", err)
	}

	res, err := e.Synthesize(context.Background(), inference.SynthesizeRequest{Text: "hello world"})
	if err != nil {
		t.Fatalf("Synthesize: %v", err)
	}
	defer os.Remove(res.WavPath)
	got, err := os.ReadFile(res.WavPath)
	if err != nil {
		t.Fatalf("read wav: %v", err)
	}
	if string(got) != string(want) {
		t.Fatalf("synthesized wav = %q, want %q", got, want)
	}
}
