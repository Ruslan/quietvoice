package remote

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"

	"quietvoice/internal/inference"
)

// TestRemoteSynthesizeUsesOpenAISpeech asserts the migrated client posts the
// OpenAI /v1/audio/speech body (input/voice/model) — not the old /v1/tts — and
// stores the returned WAV.
func TestRemoteSynthesizeUsesOpenAISpeech(t *testing.T) {
	var gotPath string
	var gotBody inference.SpeechRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		w.Header().Set("Content-Type", "audio/wav")
		_, _ = w.Write([]byte("RIFFremote"))
	}))
	defer srv.Close()

	e := New(Config{BaseURL: srv.URL, WorkDir: t.TempDir()})
	res, err := e.Synthesize(context.Background(), inference.SynthesizeRequest{Text: "hello", Voice: "ded", Model: "qwen3-tts-1.7b"})
	if err != nil {
		t.Fatalf("Synthesize: %v", err)
	}
	if gotPath != inference.RouteSpeech {
		t.Fatalf("posted to %q, want %q", gotPath, inference.RouteSpeech)
	}
	if gotBody.Input != "hello" || gotBody.Voice != "ded" || gotBody.Model != "qwen3-tts-1.7b" {
		t.Fatalf("unexpected OpenAI body: %+v", gotBody)
	}
	if gotBody.ResponseFormat != "wav" {
		t.Fatalf("response_format = %q, want wav", gotBody.ResponseFormat)
	}
	defer os.Remove(res.WavPath)
	b, _ := os.ReadFile(res.WavPath)
	if string(b) != "RIFFremote" {
		t.Fatalf("stored wav = %q", b)
	}
}

// TestEnsureVoice covers the control-plane voice provisioning: no-op without a
// wav, skip when the node already has the voice, and upload (multipart) when it
// is missing.
func TestEnsureVoice(t *testing.T) {
	writeWav := func(t *testing.T) string {
		p := filepath.Join(t.TempDir(), "ded.wav")
		if err := os.WriteFile(p, []byte("RIFFvoice"), 0o644); err != nil {
			t.Fatal(err)
		}
		return p
	}

	t.Run("noop without wav", func(t *testing.T) {
		var hits int32
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			atomic.AddInt32(&hits, 1)
		}))
		defer srv.Close()
		e := New(Config{BaseURL: srv.URL, VoiceName: "ded"}) // no VoiceWav
		if err := e.EnsureVoice(context.Background()); err != nil {
			t.Fatalf("EnsureVoice: %v", err)
		}
		if hits != 0 {
			t.Fatalf("expected no HTTP calls, got %d", hits)
		}
	})

	t.Run("skip when present", func(t *testing.T) {
		var posted int32
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method == http.MethodPost {
				atomic.AddInt32(&posted, 1)
			}
			_ = json.NewEncoder(w).Encode(inference.VoicesResponse{Voices: []string{"ded", "other"}})
		}))
		defer srv.Close()
		e := New(Config{BaseURL: srv.URL, VoiceName: "ded", VoiceWav: writeWav(t), VoiceTranscript: "hi"})
		if err := e.EnsureVoice(context.Background()); err != nil {
			t.Fatalf("EnsureVoice: %v", err)
		}
		if posted != 0 {
			t.Fatalf("voice already present: expected no upload, got %d POSTs", posted)
		}
	})

	t.Run("upload when missing", func(t *testing.T) {
		var gotName, gotTranscript string
		var gotFile bool
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			switch r.Method {
			case http.MethodGet:
				_ = json.NewEncoder(w).Encode(inference.VoicesResponse{Voices: []string{"other"}})
			case http.MethodPost:
				_ = r.ParseMultipartForm(1 << 20)
				gotName = r.FormValue("name")
				gotTranscript = r.FormValue("transcript")
				if f, _, err := r.FormFile("voice"); err == nil {
					gotFile = true
					f.Close()
				}
				w.WriteHeader(http.StatusCreated)
			}
		}))
		defer srv.Close()
		e := New(Config{BaseURL: srv.URL, VoiceName: "ded", VoiceWav: writeWav(t), VoiceTranscript: "hi ded"})
		if err := e.EnsureVoice(context.Background()); err != nil {
			t.Fatalf("EnsureVoice: %v", err)
		}
		if gotName != "ded" || gotTranscript != "hi ded" || !gotFile {
			t.Fatalf("upload multipart = name:%q transcript:%q file:%v", gotName, gotTranscript, gotFile)
		}
	})

	t.Run("missing transcript errors", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			_ = json.NewEncoder(w).Encode(inference.VoicesResponse{Voices: []string{"other"}})
		}))
		defer srv.Close()
		e := New(Config{BaseURL: srv.URL, VoiceName: "ded", VoiceWav: writeWav(t)}) // no transcript
		if err := e.EnsureVoice(context.Background()); err == nil {
			t.Fatal("expected error when transcript missing, got nil")
		}
	})
}

// TestRemoteInterpretUsesNativeRoute asserts the rich intent path still targets
// the native /v1/interpret route (kept alongside the OpenAI surface).
func TestRemoteInterpretUsesNativeRoute(t *testing.T) {
	var gotPath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		_ = r.ParseMultipartForm(1 << 20)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(inference.InterpretResult{Type: "intent", Intent: "do the thing"})
	}))
	defer srv.Close()

	// A small audio file to upload.
	dir := t.TempDir()
	audioPath := filepath.Join(dir, "in.wav")
	if err := os.WriteFile(audioPath, []byte("RIFFaudio"), 0o644); err != nil {
		t.Fatal(err)
	}

	e := New(Config{BaseURL: srv.URL, WorkDir: dir})
	res, err := e.Interpret(context.Background(), inference.AudioInput{Path: audioPath}, inference.InterpretRequest{Mode: inference.ModeAssisted})
	if err != nil {
		t.Fatalf("Interpret: %v", err)
	}
	if gotPath != inference.RouteInterpret {
		t.Fatalf("posted to %q, want %q", gotPath, inference.RouteInterpret)
	}
	if res.Intent != "do the thing" {
		t.Fatalf("intent = %q", res.Intent)
	}
}
