package local

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"quietvoice/internal/inference"
)

// TestLangForPrefersRequest proves the per-request Language (forwarded by the
// control plane) wins over the node's configured default, and that an empty /
// whitespace request language falls back to the node default.
func TestLangForPrefersRequest(t *testing.T) {
	e := New(Config{Lang: "ru"})
	cases := []struct {
		reqLang string
		want    string
	}{
		{"en", "en"},
		{"auto", "auto"},
		{"", "ru"},
		{"   ", "ru"},
	}
	for _, tc := range cases {
		if got := e.langFor(inference.InterpretRequest{Language: tc.reqLang}); got != tc.want {
			t.Fatalf("langFor(Language=%q) = %q, want %q", tc.reqLang, got, tc.want)
		}
	}
}

// capturingASRServer records the `language` form field of the last transcription
// request so a test can prove the hint is actually forwarded over the wire.
func capturingASRServer(t *testing.T, gotLang *string, mu *sync.Mutex) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(http.StatusOK) })
	mux.HandleFunc("POST /v1/audio/transcriptions", func(w http.ResponseWriter, r *http.Request) {
		_ = r.ParseMultipartForm(1 << 20)
		mu.Lock()
		*gotLang = r.FormValue("language")
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"text": "ok"})
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv
}

// TestRunRecognizersForwardsLanguage proves runRecognizers forwards the chosen
// language hint to the hot ASR server (the fix: no hardcoded "ru").
func TestRunRecognizersForwardsLanguage(t *testing.T) {
	var mu sync.Mutex
	var gotLang string
	asr := capturingASRServer(t, &gotLang, &mu)

	e := New(Config{
		Lang:        "ru", // node default that must be overridable
		Recognizers: []Recognizer{{Name: "whisper-large", Model: "/m/whisper.bin", Backend: "whisper"}},
	})
	e.ensurePool("asr", "").add(&worker{url: asr.URL, role: "asr", managed: true})

	wav := filepath.Join(t.TempDir(), "in.wav")
	if err := os.WriteFile(wav, []byte("RIFFxxxx"), 0o644); err != nil {
		t.Fatal(err)
	}

	if got := e.runRecognizers(context.Background(), wav, "en"); len(got) != 1 {
		t.Fatalf("want 1 transcript, got %d", len(got))
	}
	mu.Lock()
	defer mu.Unlock()
	if gotLang != "en" {
		t.Fatalf("ASR server received language=%q, want the forwarded \"en\" (not the node default)", gotLang)
	}
}
