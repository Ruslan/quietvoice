package local

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"quietvoice/internal/audio"
	"quietvoice/internal/inference"
)

// gemmaContentMode selects how the fake gemma server encodes the assistant
// message `content` (string vs array of parts), so both llama-server shapes are
// covered.
type gemmaContentMode int

const (
	contentString gemmaContentMode = iota
	contentArray
)

// newFakeGemmaServer stands in for a hot `llama-server -m gemma … --mmproj …`
// replica: it answers GET /health and, for POST /v1/chat/completions, records
// the received body and returns the given content. sawAudio is set true when the
// request carried an input_audio part.
func newFakeGemmaServer(t *testing.T, content string, mode gemmaContentMode, sawAudio *bool) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(http.StatusOK) })
	mux.HandleFunc("POST /v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		if strings.Contains(string(raw), `"input_audio"`) && strings.Contains(string(raw), `"format":"wav"`) {
			if sawAudio != nil {
				*sawAudio = true
			}
		}
		var msg any
		switch mode {
		case contentArray:
			msg = map[string]any{"content": []any{map[string]any{"type": "text", "text": content}}}
		default:
			msg = map[string]any{"content": content}
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []any{map[string]any{"message": msg}},
		})
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv
}

// writeFakeAudioInput creates a source audio file for Interpret and points the
// audio transcoder at a fake ffmpeg so ToMono16kWav runs codec-free.
func writeFakeAudioInput(t *testing.T) string {
	t.Helper()
	in := filepath.Join(t.TempDir(), "in.ogg")
	if err := os.WriteFile(in, []byte("OggSfake"), 0o644); err != nil {
		t.Fatal(err)
	}
	origFF := audio.FFmpegBin
	audio.FFmpegBin = writeFakeFFmpeg(t)
	t.Cleanup(func() { audio.FFmpegBin = origFF })
	return in
}

func TestDecodeChatContent(t *testing.T) {
	cases := []struct {
		name string
		raw  string
		want string
	}{
		{"string", `"hello world"`, "hello world"},
		{"array of parts", `[{"type":"text","text":"foo"},{"type":"text","text":"bar"}]`, "foobar"},
		{"null", `null`, ""},
		{"empty", ``, ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := decodeChatContent(json.RawMessage(c.raw))
			if err != nil {
				t.Fatalf("decodeChatContent(%q): %v", c.raw, err)
			}
			if got != c.want {
				t.Fatalf("decodeChatContent(%q) = %q, want %q", c.raw, got, c.want)
			}
		})
	}
}

// TestInterpretRoutesToHotGemma proves that when a hot gemma server is in the
// pool, Interpret routes to it (POST /v1/chat/completions with an input_audio
// part), runs extractGemmaFinal on the returned content, and returns the parsed
// intent. Covers both the string and array `content` encodings.
func TestInterpretRoutesToHotGemma(t *testing.T) {
	for _, mode := range []gemmaContentMode{contentString, contentArray} {
		in := writeFakeAudioInput(t)
		sawAudio := false
		// A gemma-style answer with a thought block; extractGemmaFinal must strip it.
		content := "<|channel>thought\nreasoning...<channel|>Run the tests then deploy."
		srv := newFakeGemmaServer(t, content, mode, &sawAudio)

		e := New(Config{WorkDir: t.TempDir()})
		e.httpClient = srv.Client()
		e.ensurePool("gemma", "").add(&worker{url: srv.URL, role: "gemma", managed: true})

		res, err := e.Interpret(context.Background(), inference.AudioInput{Path: in}, inference.InterpretRequest{Mode: inference.ModeIntent})
		if err != nil {
			t.Fatalf("Interpret (hot gemma): %v", err)
		}
		if !sawAudio {
			t.Fatal("gemma server did not receive an input_audio part")
		}
		if res.Intent != "Run the tests then deploy." {
			t.Fatalf("intent = %q, want extracted final answer", res.Intent)
		}
		if res.Type != "intent" {
			t.Fatalf("type = %q, want intent (no refs)", res.Type)
		}
		if res.Raw != content {
			t.Fatalf("raw = %q, want the raw content %q", res.Raw, content)
		}
	}
}

// TestInterpretAssistedRoutesToHotGemma proves the assisted path (ASR ensemble +
// Gemma reconcile) also routes to the hot gemma server, attaches the reference
// transcript, and reports Type assisted-intent.
func TestInterpretAssistedRoutesToHotGemma(t *testing.T) {
	in := writeFakeAudioInput(t)
	sawAudio := false
	srv := newFakeGemmaServer(t, "spin up, not conflict", contentString, &sawAudio)

	// One recognizer via a fake crispasr so the assisted ensemble produces a ref.
	e := New(Config{
		WorkDir:     t.TempDir(),
		CrispasrBin: writeFakeCrispasr(t),
		Recognizers: []Recognizer{{Name: "whisper-large", Model: "/m/whisper.bin", Backend: "whisper"}},
	})
	e.httpClient = srv.Client()
	e.ensurePool("gemma", "").add(&worker{url: srv.URL, role: "gemma", managed: true})

	res, err := e.Interpret(context.Background(), inference.AudioInput{Path: in}, inference.InterpretRequest{Mode: inference.ModeAssisted})
	if err != nil {
		t.Fatalf("Interpret (assisted hot gemma): %v", err)
	}
	if !sawAudio {
		t.Fatal("gemma server did not receive an input_audio part")
	}
	if res.Type != "assisted-intent" {
		t.Fatalf("type = %q, want assisted-intent", res.Type)
	}
	if res.Intent != "spin up, not conflict" {
		t.Fatalf("intent = %q", res.Intent)
	}
	if len(res.Transcripts) != 1 || !strings.Contains(res.Transcripts[0].Text, "spin up") {
		t.Fatalf("transcripts = %+v, want the ensemble ref attached", res.Transcripts)
	}
}

// TestInterpretGemmaFailsOverAndEvictsDeadWorker mirrors the tts/asr failover
// test: the first checked-out gemma worker is dead (dial failure), so Interpret
// evicts it and fails over to a healthy worker.
func TestInterpretGemmaFailsOverAndEvictsDeadWorker(t *testing.T) {
	in := writeFakeAudioInput(t)
	healthy := newFakeGemmaServer(t, "do the thing", contentString, nil)
	deadURL := deadWorkerURL(t)
	healthyURL := strings.TrimRight(healthy.URL, "/")

	e := New(Config{WorkDir: t.TempDir()})
	e.httpClient = healthy.Client()
	p := e.ensurePool("gemma", "")
	p.add(&worker{url: deadURL, role: "gemma"}) // handed out first (FIFO idle queue)
	p.add(&worker{url: healthyURL, role: "gemma"})

	res, err := e.Interpret(context.Background(), inference.AudioInput{Path: in}, inference.InterpretRequest{Mode: inference.ModeIntent})
	if err != nil {
		t.Fatalf("Interpret should fail over to the healthy gemma worker, got: %v", err)
	}
	if res.Intent != "do the thing" {
		t.Fatalf("intent = %q, want the healthy worker's answer", res.Intent)
	}
	if p.size() != 1 {
		t.Fatalf("pool size = %d after eviction, want 1", p.size())
	}
	for _, u := range p.urls() {
		if u == deadURL {
			t.Fatalf("dead gemma worker %q still in pool after eviction", deadURL)
		}
	}
}

// TestInterpretColdSpawnFallback proves that with NO gemma pool, Interpret keeps
// today's behavior: it cold-spawns llama-mtmd-cli on the configured model/mmproj.
func TestInterpretColdSpawnFallback(t *testing.T) {
	in := writeFakeAudioInput(t)
	bin := writeFakeMtmdCLI(t)

	e := New(Config{
		WorkDir:      t.TempDir(),
		LlamaMtmdBin: bin,
		GemmaModel:   "/m/gemma.gguf",
		GemmaMMProj:  "/m/mmproj.gguf",
	})
	// No gemma pool -> pickGemmaPool returns nil -> CLI cold-spawn.
	if e.pickGemmaPool("") != nil {
		t.Fatal("expected no gemma pool")
	}

	res, err := e.Interpret(context.Background(), inference.AudioInput{Path: in}, inference.InterpretRequest{Mode: inference.ModeIntent})
	if err != nil {
		t.Fatalf("Interpret (CLI fallback): %v", err)
	}
	if res.Intent != "Cold spawn answer." {
		t.Fatalf("intent = %q, want the fake CLI's final answer", res.Intent)
	}
	if res.Type != "intent" {
		t.Fatalf("type = %q, want intent", res.Type)
	}
}

// writeFakeMtmdCLI writes a stand-in for llama-mtmd-cli: it prints a log-noise
// line plus a gemma-style final answer, so the cold-spawn interpret path runs
// GPU-free.
func writeFakeMtmdCLI(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "llama-mtmd-cli")
	script := "#!/bin/sh\n" +
		"echo 'main: loading model'\n" +
		"echo 'Cold spawn answer.'\n"
	if err := os.WriteFile(path, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	return path
}
