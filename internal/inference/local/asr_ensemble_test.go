package local

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

// newASRTextServer is a fake crispasr --server that returns a fixed transcript, so a
// test can tell WHICH hot asr replica a recognizer leg was routed to.
func newASRTextServer(t *testing.T, text string) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(http.StatusOK) })
	mux.HandleFunc("POST /v1/audio/transcriptions", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"text": text})
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv
}

func argValue(args []string, flag string) string {
	for i, a := range args {
		if a == flag && i+1 < len(args) {
			return args[i+1]
		}
	}
	return ""
}

// TestLaunchSpecASRModelKeyed proves launchSpec resolves an asr replica's gguf AND
// crispasr backend from the recognizer registry by name: the default ("") pool uses
// the single global ASR model/backend, a named pool uses that recognizer's model +
// backend, and an unknown name is rejected (not silently launched as whisper).
func TestLaunchSpecASRModelKeyed(t *testing.T) {
	cfg := Config{
		ASRServerBin: "/bin/crispasr",
		ASRModel:     "/m/whisper.bin",
		ASRBackend:   "whisper",
		Recognizers: []Recognizer{
			{Name: "voxtral", Model: "/m/voxtral.gguf", Backend: "voxtral4b"},
			{Name: "whisper-large", Model: "/m/whisper.bin", Backend: "whisper"},
		},
	}

	// Default ("") pool → single global whisper.
	def, err := launchSpec(cfg, "asr", "", 9000)
	if err != nil {
		t.Fatalf("launchSpec asr default: %v", err)
	}
	if got := argValue(def.args, "-m"); got != "/m/whisper.bin" {
		t.Errorf("default -m = %q, want /m/whisper.bin", got)
	}
	if got := argValue(def.args, "--backend"); got != "whisper" {
		t.Errorf("default --backend = %q, want whisper", got)
	}

	// Named "voxtral" pool → voxtral gguf + voxtral4b backend.
	vox, err := launchSpec(cfg, "asr", "voxtral", 9001)
	if err != nil {
		t.Fatalf("launchSpec asr voxtral: %v", err)
	}
	if got := argValue(vox.args, "-m"); got != "/m/voxtral.gguf" {
		t.Errorf("voxtral -m = %q, want /m/voxtral.gguf", got)
	}
	if got := argValue(vox.args, "--backend"); got != "voxtral4b" {
		t.Errorf("voxtral --backend = %q, want voxtral4b", got)
	}

	// Unknown name → error, not a silent whisper launch keyed to the wrong pool.
	if _, err := launchSpec(cfg, "asr", "nope", 9002); err == nil {
		t.Error("launchSpec asr with unknown recognizer name: want error, got nil")
	}
}

// TestLaunchSpecTTSVoiceDir proves the supervisor creates the TTS voice-dir and passes
// --voice-dir when configured (so a fresh node's first POST /v1/voices doesn't 400 on a
// missing directory), and launches without the flag (warning only, no error) when unset.
func TestLaunchSpecTTSVoiceDir(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "voices")
	cfg := Config{
		TTSServerBin:  "/bin/crispasr",
		TTSModel:      "/m/tts.gguf",
		TTSCodecModel: "/m/codec.gguf",
		TTSBackend:    "qwen3-tts",
		TTSVoiceDir:   dir,
	}
	plan, err := launchSpec(cfg, "tts", "", 9000)
	if err != nil {
		t.Fatalf("launchSpec tts: %v", err)
	}
	if got := argValue(plan.args, "--voice-dir"); got != dir {
		t.Errorf("--voice-dir = %q, want %q", got, dir)
	}
	if _, err := os.Stat(dir); err != nil {
		t.Errorf("voice-dir not created: %v", err)
	}

	// Unset → launch without --voice-dir (a warning is logged), but no error.
	cfg.TTSVoiceDir = ""
	plan, err = launchSpec(cfg, "tts", "", 9001)
	if err != nil {
		t.Fatalf("launchSpec tts (no voice-dir): %v", err)
	}
	if got := argValue(plan.args, "--voice-dir"); got != "" {
		t.Errorf("--voice-dir = %q, want none when unset", got)
	}
}

// TestRunRecognizersModelKeyedEnsemble proves the fix: with Voxtral and Whisper each
// launched as their OWN hot asr pool, the assisted ensemble routes each recognizer
// leg to the correct model (by rec.Name) — not both to a single whisper.
func TestRunRecognizersModelKeyedEnsemble(t *testing.T) {
	wav := filepath.Join(t.TempDir(), "asr.wav")
	if err := os.WriteFile(wav, []byte("RIFFx"), 0o644); err != nil {
		t.Fatal(err)
	}
	vox := newASRTextServer(t, "VOX")
	whi := newASRTextServer(t, "WHI")

	e := New(Config{
		WorkDir:     t.TempDir(),
		CrispasrBin: "/nonexistent/crispasr-should-not-run", // cold-spawn would fail → proves hot routing
		Recognizers: []Recognizer{
			{Name: "voxtral", Model: "/m/voxtral.gguf", Backend: "voxtral4b"},
			{Name: "whisper-large", Model: "/m/whisper.bin", Backend: "whisper"},
		},
	})
	e.httpClient = vox.Client()
	e.ensurePool("asr", "voxtral").add(&worker{url: vox.URL, role: "asr", model: "voxtral", managed: true})
	e.ensurePool("asr", "whisper-large").add(&worker{url: whi.URL, role: "asr", model: "whisper-large", managed: true})

	got := e.runRecognizers(context.Background(), wav, "auto")
	byName := map[string]string{}
	for _, tr := range got {
		byName[tr.Name] = tr.Text
	}
	if byName["voxtral"] != "VOX" {
		t.Errorf("voxtral leg = %q, want VOX (routed to the voxtral pool)", byName["voxtral"])
	}
	if byName["whisper-large"] != "WHI" {
		t.Errorf("whisper-large leg = %q, want WHI (routed to the whisper pool)", byName["whisper-large"])
	}
}

// TestRunRecognizersMixedNamedAndDefault is the realistic deployment: Whisper on the
// default ("") pool (also serving single-shot Transcribe) and Voxtral as a named hot
// pool. The whisper-large leg falls through to the default pool; the voxtral leg hits
// its named pool.
func TestRunRecognizersMixedNamedAndDefault(t *testing.T) {
	wav := filepath.Join(t.TempDir(), "asr.wav")
	if err := os.WriteFile(wav, []byte("RIFFx"), 0o644); err != nil {
		t.Fatal(err)
	}
	vox := newASRTextServer(t, "VOX")
	whi := newASRTextServer(t, "WHI")

	e := New(Config{
		WorkDir:     t.TempDir(),
		CrispasrBin: "/nonexistent/crispasr-should-not-run",
		Recognizers: []Recognizer{
			{Name: "voxtral", Model: "/m/voxtral.gguf", Backend: "voxtral4b"},
			{Name: "whisper-large", Model: "/m/whisper.bin", Backend: "whisper"},
		},
	})
	e.httpClient = vox.Client()
	e.ensurePool("asr", "voxtral").add(&worker{url: vox.URL, role: "asr", model: "voxtral", managed: true})
	e.ensurePool("asr", "").add(&worker{url: whi.URL, role: "asr", managed: true}) // default whisper

	got := e.runRecognizers(context.Background(), wav, "auto")
	byName := map[string]string{}
	for _, tr := range got {
		byName[tr.Name] = tr.Text
	}
	if byName["voxtral"] != "VOX" {
		t.Errorf("voxtral leg = %q, want VOX (named pool)", byName["voxtral"])
	}
	if byName["whisper-large"] != "WHI" {
		t.Errorf("whisper-large leg = %q, want WHI (fell through to default pool)", byName["whisper-large"])
	}
}
