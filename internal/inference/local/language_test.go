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
// control plane) wins over the node's configured default, and that a request
// which states NO preference — empty, whitespace, or "auto" — falls back to the
// node default.
func TestLangForPrefersRequest(t *testing.T) {
	e := New(Config{Lang: "ru"})
	cases := []struct {
		reqLang string
		want    string
	}{
		{"en", "en"},
		{"auto", "ru"}, // "auto" is the caller's default, not a decision
		{"AUTO", "ru"}, // and it is not case-sensitive
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

// TestLangForAutoDoesNotOverrideNodeDefault is a regression test for a live
// incident (2026-09-06) and is deliberately separate from the table above,
// because the behaviour it pins is the OPPOSITE of what the code did before.
//
// The node had ASR_LANG=ru. The control plane forwarded its own default,
// "auto". The old langFor treated that as a decision and returned "auto", so
// crispasr ran whisper language-detect ahead of the qwen3-asr backend — a
// combination that crashes the worker on every request (exit 0xC0000409 from
// the CLI, "internal error: bad conversion" from the server). The pool failed
// over to the CPU whisper at 0.4x realtime, so the node stayed "healthy" and
// only the latency gave it away: 22 s to transcribe 9 s of speech.
//
// The rule this pins: a caller that never chose a language must not silently
// override an operator who did.
func TestLangForAutoDoesNotOverrideNodeDefault(t *testing.T) {
	e := New(Config{Lang: "ru"})
	if got := e.langFor(inference.InterpretRequest{Language: "auto"}); got != "ru" {
		t.Fatalf("langFor(auto) = %q, want the node default \"ru\": a pinned ASR_LANG must survive a caller that stated no preference", got)
	}
	// An explicit language still wins — "auto" is special, the mechanism is not.
	if got := e.langFor(inference.InterpretRequest{Language: "en"}); got != "en" {
		t.Fatalf("langFor(en) = %q, want \"en\": an explicit request language must still win", got)
	}
	// A node that itself wants detection keeps it: nothing forces a language.
	auto := New(Config{Lang: "auto"})
	if got := auto.langFor(inference.InterpretRequest{Language: "auto"}); got != "auto" {
		t.Fatalf("langFor(auto) on an auto node = %q, want \"auto\"", got)
	}
}
