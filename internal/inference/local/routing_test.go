package local

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"quietvoice/internal/audio"
	"quietvoice/internal/inference"
)

// writeFakeFFmpeg writes a stand-in for ffmpeg that just creates the output file
// (its last argument), so the CLI transcode step in Transcribe runs GPU/codec-free.
func writeFakeFFmpeg(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "ffmpeg")
	script := "#!/bin/sh\n" +
		"for out do :; done\n" + // POSIX: after the loop `out` holds the last positional arg
		"printf 'RIFFfake' > \"$out\"\n"
	if err := os.WriteFile(path, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	return path
}

// TestTranscribeRoutesToHotASR proves that when a hot asr server is in the pool
// Transcribe routes to it (HTTP /v1/audio/transcriptions), and that with no asr
// pool it cold-spawns the crispasr CLI instead — the ASR analogue of the tts
// server-vs-CLI failover test.
func TestTranscribeRoutesToHotASR(t *testing.T) {
	audioFile := filepath.Join(t.TempDir(), "in.wav")
	if err := os.WriteFile(audioFile, []byte("RIFFsomeaudio"), 0o644); err != nil {
		t.Fatal(err)
	}

	// Hot path: a live asr server answers, no CLI involved.
	srv := newFakeReplicaServer(t, []byte("RIFFx")) // its /v1/audio/transcriptions returns "стартуют"
	e := New(Config{WorkDir: t.TempDir()})
	e.httpClient = srv.Client()
	e.ensurePool("asr", "").add(&worker{url: srv.URL, role: "asr", managed: true})

	got, err := e.Transcribe(context.Background(), audioFile, "", "ru")
	if err != nil {
		t.Fatalf("Transcribe (hot asr): %v", err)
	}
	if got != "стартуют" {
		t.Fatalf("hot asr transcript = %q, want %q", got, "стартуют")
	}

	// CLI fallback: no asr pool -> cold-spawn the crispasr CLI (via the fake bin),
	// transcoding through the fake ffmpeg.
	origFF := audio.FFmpegBin
	audio.FFmpegBin = writeFakeFFmpeg(t)
	t.Cleanup(func() { audio.FFmpegBin = origFF })

	bin := writeFakeCrispasr(t)
	e2 := New(Config{
		WorkDir:     t.TempDir(),
		CrispasrBin: bin,
		Recognizers: []Recognizer{{Name: "whisper-large", Model: "/m/whisper.bin", Backend: "whisper"}},
	})
	got2, err := e2.Transcribe(context.Background(), audioFile, "", "ru")
	if err != nil {
		t.Fatalf("Transcribe (CLI fallback): %v", err)
	}
	if !strings.Contains(got2, "стартуют") || !strings.Contains(got2, "lang=ru") {
		t.Fatalf("CLI transcript = %q, want the spoken term + forwarded lang", got2)
	}
}

// writeFakeCrispasr writes a tiny shell script that stands in for the crispasr
// ASR CLI: it echoes a log-noise line (which cleanCLIOutput must drop) plus a
// transcript line embedding the forwarded -l language hint, so the ASR path is
// exercised without a model or a GPU.
func writeFakeCrispasr(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "crispasr")
	script := "#!/bin/sh\n" +
		"lang=ru\n" +
		"while [ $# -gt 0 ]; do\n" +
		"  case \"$1\" in\n" +
		"    -l) lang=\"$2\"; shift 2;;\n" +
		"    *) shift;;\n" +
		"  esac\n" +
		"done\n" +
		"echo 'whisper_init: loading model'\n" +
		"echo \"стартуют lang=$lang\"\n"
	if err := os.WriteFile(path, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	return path
}

// TestRunRecognizersRoutesToHotASR proves the assisted-mode ensemble routes each
// recognizer to the hot asr pool over HTTP (POST /v1/audio/transcriptions) instead
// of cold-spawning the crispasr CLI. Only the default ("") asr pool is up here, so
// routing by rec.Name ("whisper-large") misses its named pool and falls through
// pickASRPool to the default pool where the hot whisper lives.
func TestRunRecognizersRoutesToHotASR(t *testing.T) {
	wav := filepath.Join(t.TempDir(), "asr.wav")
	if err := os.WriteFile(wav, []byte("RIFFsomeaudio"), 0o644); err != nil {
		t.Fatal(err)
	}

	srv := newFakeReplicaServer(t, []byte("RIFFx")) // /v1/audio/transcriptions -> "стартуют"
	// CrispasrBin points at a binary that does NOT exist: if the ensemble tried to
	// cold-spawn instead of using the hot pool, the transcript would be empty.
	e := New(Config{
		WorkDir:     t.TempDir(),
		CrispasrBin: "/nonexistent/crispasr-should-not-run",
		Recognizers: []Recognizer{{Name: "whisper-large", Model: "/m/whisper.bin", Backend: "whisper"}},
	})
	e.httpClient = srv.Client()
	e.ensurePool("asr", "").add(&worker{url: srv.URL, role: "asr", managed: true})

	got := e.runRecognizers(context.Background(), wav, "auto")
	if len(got) != 1 || got[0].Text != "стартуют" {
		t.Fatalf("runRecognizers (hot asr) = %+v, want one transcript %q", got, "стартуют")
	}
}

// TestRunRecognizersColdSpawnFallback proves that with NO asr pool the ensemble
// cold-spawns the crispasr CLI (via the fake bin), forwarding the language hint.
func TestRunRecognizersColdSpawnFallback(t *testing.T) {
	wav := filepath.Join(t.TempDir(), "asr.wav")
	if err := os.WriteFile(wav, []byte("RIFFsomeaudio"), 0o644); err != nil {
		t.Fatal(err)
	}
	bin := writeFakeCrispasr(t)
	e := New(Config{
		WorkDir:     t.TempDir(),
		CrispasrBin: bin,
		Lang:        "ru",
		Recognizers: []Recognizer{{Name: "whisper-large", Model: "/m/whisper.bin", Backend: "whisper"}},
	})
	got := e.runRecognizers(context.Background(), wav, "ru") // explicit hint forwarded to the CLI
	if len(got) != 1 || !strings.Contains(got[0].Text, "стартуют") || !strings.Contains(got[0].Text, "lang=ru") {
		t.Fatalf("runRecognizers (CLI fallback) = %+v, want the spoken term + forwarded lang", got)
	}
}

// TestASRTranscriptServerFailover proves the ensemble hot path evicts a dead first
// worker and retries on a live one (the bounded-retry/dead-worker-evict loop).
func TestASRTranscriptServerFailover(t *testing.T) {
	wav := filepath.Join(t.TempDir(), "asr.wav")
	if err := os.WriteFile(wav, []byte("RIFFsomeaudio"), 0o644); err != nil {
		t.Fatal(err)
	}
	live := newFakeReplicaServer(t, []byte("RIFFx")) // -> "стартуют"

	// A dead worker: a URL whose server is already closed, so a dial fails (evict).
	dead := httptest.NewServer(http.NewServeMux())
	deadURL := dead.URL
	dead.Close()

	e := New(Config{WorkDir: t.TempDir()})
	e.httpClient = live.Client()
	p := e.ensurePool("asr", "")
	p.add(&worker{url: deadURL, role: "asr", managed: true})
	p.add(&worker{url: live.URL, role: "asr", managed: true})

	rec := Recognizer{Name: "whisper-large", Model: "/m/whisper.bin", Backend: "whisper"}
	got, err := e.asrTranscriptLang(context.Background(), wav, rec, "ru")
	if err != nil {
		t.Fatalf("asrTranscriptLang failover: %v", err)
	}
	if got != "стартуют" {
		t.Fatalf("failover transcript = %q, want %q", got, "стартуют")
	}
	if got := e.poolSize("asr", ""); got != 1 {
		t.Fatalf("asr pool size after evicting dead worker = %d, want 1", got)
	}
}

// TestModelRoutingPicksRightPool verifies decision #2: a request routes to an
// idle instance of the requested model, an empty model uses the default pool,
// and an unknown model is an error (no on-demand launch).
func TestModelRoutingPicksRightPool(t *testing.T) {
	def := &fakeLauncher{t: t}
	e := newAdminEngine(t, 100000, def)

	// Launch a replica for the default pool and one for a named model.
	if _, err := e.SetReplicas(context.Background(), "tts", "", 1); err != nil {
		t.Fatalf("scale default: %v", err)
	}
	if _, err := e.SetReplicas(context.Background(), "tts", "qwen3-tts-1.7b", 1); err != nil {
		t.Fatalf("scale model pool: %v", err)
	}
	// Point the engine's client at the fake servers.
	e.httpClient = def.servers[0].Client()

	// Empty model -> default pool (has a worker).
	p, err := e.routeTTS("")
	if err != nil {
		t.Fatalf("routeTTS(default): %v", err)
	}
	if p != e.lookupTTSPool("") {
		t.Fatal("empty model did not route to the default pool")
	}

	// Named model -> its own pool.
	p, err = e.routeTTS("qwen3-tts-1.7b")
	if err != nil {
		t.Fatalf("routeTTS(model): %v", err)
	}
	if p != e.lookupTTSPool("qwen3-tts-1.7b") {
		t.Fatal("named model did not route to its own pool")
	}

	// Unknown model -> error, not a fallback.
	if _, err := e.routeTTS("does-not-exist"); err == nil {
		t.Fatal("expected error routing an unknown model, got nil")
	}

	// Synthesize with an unknown model must error, not silently use another pool.
	if _, err := e.Synthesize(context.Background(), inference.SynthesizeRequest{Text: "hi", Model: "does-not-exist"}); err == nil || !strings.Contains(err.Error(), "does-not-exist") {
		t.Fatalf("Synthesize(unknown model) err = %v, want a routing error", err)
	}

	// Both pools count independently.
	if got := e.ttsPoolSize(""); got != 1 {
		t.Fatalf("default pool size = %d, want 1", got)
	}
	if got := e.ttsPoolSize("qwen3-tts-1.7b"); got != 1 {
		t.Fatalf("model pool size = %d, want 1", got)
	}
	if got := e.ManagedReplicas(); got != 2 {
		t.Fatalf("total managed = %d, want 2", got)
	}
}

// TestPickRecognizer covers the ASR model-selection rules used by the
// /v1/audio/transcriptions route (default = first, named match, unknown = error).
func TestPickRecognizer(t *testing.T) {
	e := New(Config{
		WorkDir: t.TempDir(),
		Recognizers: []Recognizer{
			{Name: "voxtral", Model: "/m/voxtral.gguf", Backend: "voxtral4b"},
			{Name: "whisper-large", Model: "/m/whisper.bin", Backend: "whisper"},
		},
	})
	if r, err := e.pickRecognizer(""); err != nil || r.Name != "voxtral" {
		t.Fatalf("pickRecognizer(default) = %+v, %v; want voxtral", r, err)
	}
	if r, err := e.pickRecognizer("whisper-large"); err != nil || r.Name != "whisper-large" {
		t.Fatalf("pickRecognizer(named) = %+v, %v; want whisper-large", r, err)
	}
	if _, err := e.pickRecognizer("nope"); err == nil {
		t.Fatal("expected error for unknown recognizer, got nil")
	}

	// No recognizers configured => error.
	e2 := New(Config{WorkDir: t.TempDir()})
	if _, err := e2.pickRecognizer(""); err == nil {
		t.Fatal("expected error with no recognizers configured, got nil")
	}
}

// TestASRTranscriptViaCLI drives the raw ASR CLI path with a fake crispasr binary
// that echoes a fixed transcript (GPU-free), asserting the language hint is
// forwarded and log noise is stripped.
func TestASRTranscriptViaCLI(t *testing.T) {
	bin := writeFakeCrispasr(t)
	e := New(Config{WorkDir: t.TempDir(), CrispasrBin: bin})
	rec := Recognizer{Name: "voxtral", Model: "/m/voxtral.gguf", Backend: "voxtral4b"}

	got, err := e.asrTranscriptLang(context.Background(), "/tmp/audio.wav", rec, "ru")
	if err != nil {
		t.Fatalf("asrTranscriptLang: %v", err)
	}
	// The fake echoes the -l value plus a fixed transcript, and prepends a noise
	// line that cleanCLIOutput must drop.
	if !strings.Contains(got, "стартуют") {
		t.Fatalf("transcript = %q, want the spoken term", got)
	}
	if !strings.Contains(got, "lang=ru") {
		t.Fatalf("transcript = %q, want the forwarded language hint lang=ru", got)
	}
	if strings.Contains(got, "whisper_init") {
		t.Fatalf("transcript = %q, log noise was not stripped", got)
	}
}
