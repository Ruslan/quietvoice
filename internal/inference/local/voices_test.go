package local

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"quietvoice/internal/inference"
)

// voiceServer is a fake crispasr instance with a per-process voice registry
// (like the real per-process cache), so fan-out uploads can be asserted per
// instance. It also serves TTS so the same fake doubles as a speech backend.
type voiceServer struct {
	srv     *httptest.Server
	mu      sync.Mutex
	voices  map[string]bool
	uploads int
}

func newVoiceServer(t *testing.T, seed ...string) *voiceServer {
	t.Helper()
	vs := &voiceServer{voices: map[string]bool{}}
	for _, s := range seed {
		vs.voices[s] = true
	}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/voices", func(w http.ResponseWriter, r *http.Request) {
		vs.mu.Lock()
		names := make([]string, 0, len(vs.voices))
		for n := range vs.voices {
			names = append(names, n)
		}
		vs.mu.Unlock()
		_ = json.NewEncoder(w).Encode(map[string]any{"voices": names})
	})
	mux.HandleFunc("POST /v1/voices", func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseMultipartForm(8 << 20); err != nil {
			http.Error(w, "bad multipart", http.StatusBadRequest)
			return
		}
		name := r.FormValue("name")
		if name == "" {
			http.Error(w, "missing name", http.StatusBadRequest)
			return
		}
		if _, _, err := r.FormFile("voice"); err != nil {
			http.Error(w, "missing voice file", http.StatusBadRequest)
			return
		}
		vs.mu.Lock()
		defer vs.mu.Unlock()
		vs.uploads++
		if vs.voices[name] {
			w.WriteHeader(http.StatusConflict)
			return
		}
		vs.voices[name] = true
		w.WriteHeader(http.StatusCreated)
	})
	mux.HandleFunc("POST /v1/audio/speech", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "audio/wav")
		_, _ = w.Write([]byte("RIFFvoice-wav"))
	})
	vs.srv = httptest.NewServer(mux)
	t.Cleanup(vs.srv.Close)
	return vs
}

// newExternalEngine wires an Engine to a set of external (unmanaged) fake
// servers in the default pool, using their shared test client.
func newExternalEngine(t *testing.T, servers ...*voiceServer) *Engine {
	t.Helper()
	urls := make([]string, len(servers))
	for i, s := range servers {
		urls[i] = s.srv.URL
	}
	e := New(Config{WorkDir: t.TempDir(), TTSServerURLs: urls, TTSVoice: "ded"})
	if len(servers) > 0 {
		e.httpClient = servers[0].srv.Client()
	}
	return e
}

func TestUploadVoiceFansOutToAllInstances(t *testing.T) {
	a := newVoiceServer(t)
	b := newVoiceServer(t)
	c := newVoiceServer(t)
	e := newExternalEngine(t, a, b, c)

	results, err := e.UploadVoice(context.Background(), "ded", "hello", []byte("RIFFref"), "ded.wav")
	if err != nil {
		t.Fatalf("UploadVoice: %v", err)
	}
	if len(results) != 3 {
		t.Fatalf("got %d results, want 3 (one per instance)", len(results))
	}
	for _, vs := range []*voiceServer{a, b, c} {
		if vs.uploads != 1 {
			t.Fatalf("instance %s received %d uploads, want 1 (fan-out missed it)", vs.srv.URL, vs.uploads)
		}
		if !vs.voices["ded"] {
			t.Fatalf("instance %s does not have voice 'ded' after fan-out", vs.srv.URL)
		}
	}

	// Re-upload is idempotent: every instance now answers 409, still success.
	results, err = e.UploadVoice(context.Background(), "ded", "hello", []byte("RIFFref"), "ded.wav")
	if err != nil {
		t.Fatalf("idempotent re-upload: %v", err)
	}
	for _, r := range results {
		if r.Status != http.StatusConflict {
			t.Fatalf("re-upload status = %d, want 409 (already present)", r.Status)
		}
	}
}

// TestUploadVoiceRejectsEmptyTranscript asserts a WAV voice with a missing or
// whitespace-only transcript is refused BEFORE any instance is contacted: a
// textless voice would register "successfully" but fail at synth time with
// crispasr's "--ref-text was not set". So it must fail fast, and the upstream
// instances must see zero uploads.
func TestUploadVoiceRejectsEmptyTranscript(t *testing.T) {
	for _, tc := range []struct {
		name       string
		transcript string
	}{
		{"missing", ""},
		{"whitespace", "   \t\n "},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a := newVoiceServer(t)
			b := newVoiceServer(t)
			e := newExternalEngine(t, a, b)

			results, err := e.UploadVoice(context.Background(), "ded", tc.transcript, []byte("RIFFref"), "ded.wav")
			if err == nil {
				t.Fatal("UploadVoice accepted an empty transcript, want rejection")
			}
			if results != nil {
				t.Fatalf("want nil results on rejection, got %v", results)
			}
			for _, vs := range []*voiceServer{a, b} {
				if vs.uploads != 0 {
					t.Fatalf("instance %s received %d uploads; a textless voice must not reach upstream", vs.srv.URL, vs.uploads)
				}
			}
		})
	}
}

func TestListVoicesQueriesPool(t *testing.T) {
	a := newVoiceServer(t, "ded", "narrator")
	e := newExternalEngine(t, a)

	voices, err := e.ListVoices(context.Background())
	if err != nil {
		t.Fatalf("ListVoices: %v", err)
	}
	if len(voices) != 2 {
		t.Fatalf("got %d voices, want 2: %v", len(voices), voices)
	}
}

// TestOpenAISpeechRoundTripThroughPool exercises the TTS path end to end through
// the idle-checkout pool (the inbound /v1/audio/speech handler calls Synthesize).
func TestOpenAISpeechRoundTripThroughPool(t *testing.T) {
	a := newVoiceServer(t)
	e := newExternalEngine(t, a)

	res, err := e.Synthesize(context.Background(), inference.SynthesizeRequest{Text: "hello world"})
	if err != nil {
		t.Fatalf("Synthesize through pool: %v", err)
	}
	if res.WavPath == "" {
		t.Fatal("empty wav path")
	}
}
