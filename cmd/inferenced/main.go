// Command inferenced is the QuietVoice inference node (inference plane): a thin
// HTTP wrapper around the in-process local engine (crispasr TTS + llama.cpp
// Gemma). Run it on the machine with the GPU — a Mac, a box under the desk, or
// a rented CUDA droplet — and point the orchestrator at it with
// INFERENCE_MODE=remote and INFERENCE_URL. It speaks the same contract the
// remote adapter expects, so the orchestrator's logic is unchanged.
//
// With INFERENCE_ADMIN=1 it also exposes /admin/replicas, a small supervisor
// API that launches, lists, and kills hot `crispasr --server` replicas live and
// registers them in the TTS pool — horizontal scaling behind the same contract.
package main

import (
	"context"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"quietvoice/internal/config"
	"quietvoice/internal/engine"
	"quietvoice/internal/inference"
	"quietvoice/internal/inference/local"
)

func main() {
	log.SetOutput(os.Stdout)
	cfg := config.Load()
	eng := engine.BuildLocal(cfg)

	node := &node{eng: eng, local: eng, token: cfg.InferenceToken, workDir: cfg.WorkDir}

	mux := http.NewServeMux()
	mux.HandleFunc("GET "+inference.RouteHealth, node.health)
	// One OpenAI-compatible surface (inbound == outbound), plus the native rich
	// intent route.
	mux.HandleFunc("POST "+inference.RouteSpeech, node.auth(node.speech))
	mux.HandleFunc("GET "+inference.RouteVoices, node.auth(node.listVoices))
	mux.HandleFunc("POST "+inference.RouteVoices, node.auth(node.uploadVoice))
	mux.HandleFunc("POST "+inference.RouteTranscriptions, node.auth(node.transcriptions))
	mux.HandleFunc("POST "+inference.RouteInterpret, node.auth(node.interpret))
	if cfg.AdminEnabled {
		mux.HandleFunc("GET /admin/replicas", node.auth(node.adminList))
		mux.HandleFunc("POST /admin/replicas", node.auth(node.adminSet))
		mux.HandleFunc("DELETE /admin/replicas/{role}", node.auth(node.adminDelete))
		log.Printf("admin API enabled: /admin/replicas (roles tts/asr/gemma, max %d replicas total)", cfg.TTSMaxReplicas)
	}

	srv := &http.Server{Addr: cfg.InferenceListen, Handler: mux, ReadTimeout: 10 * time.Minute, WriteTimeout: 10 * time.Minute}

	// Graceful shutdown: stop the HTTP server and kill any managed replicas so
	// we don't leak crispasr processes.
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-stop
		log.Printf("shutting down: stopping managed replicas")
		eng.Shutdown()
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = srv.Shutdown(ctx)
	}()

	log.Printf("QuietVoice inference node listening on %s", cfg.InferenceListen)
	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("inference node: %v", err)
	}
	eng.Shutdown()
}

type node struct {
	eng     inference.Engine
	local   *local.Engine // concrete handle for the admin API (same instance as eng)
	token   string
	workDir string
}

func (n *node) auth(h http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if n.token != "" && !strings.EqualFold(r.Header.Get("Authorization"), "Bearer "+n.token) {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		h(w, r)
	}
}

func (n *node) health(w http.ResponseWriter, r *http.Request) {
	if err := n.eng.Health(r.Context()); err != nil {
		http.Error(w, err.Error(), http.StatusServiceUnavailable)
		return
	}
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte("ok"))
}

// speech is the OpenAI TTS route (POST /v1/audio/speech). It accepts the OpenAI
// body, routes through the SAME engine + pool the outbound client uses (so in
// server mode it is a load-balancing OpenAI proxy), and streams the WAV back.
func (n *node) speech(w http.ResponseWriter, r *http.Request) {
	var req inference.SpeechRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad request: "+err.Error(), http.StatusBadRequest)
		return
	}
	if strings.TrimSpace(req.Input) == "" {
		http.Error(w, "bad request: empty input", http.StatusBadRequest)
		return
	}
	res, err := n.eng.Synthesize(r.Context(), inference.SynthesizeRequest{Text: req.Input, Voice: req.Voice, Model: req.Model})
	if err != nil {
		// A model-routing miss (unknown model / no default pool) is the caller's
		// fault, not the server's.
		http.Error(w, err.Error(), synthErrStatus(err))
		return
	}
	defer os.Remove(res.WavPath)
	f, err := os.Open(res.WavPath)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer f.Close()
	w.Header().Set("Content-Type", "audio/wav")
	_, _ = io.Copy(w, f)
}

// synthErrStatus maps a routing miss to 400 (client error) and everything else
// to 500.
func synthErrStatus(err error) int {
	if err == nil {
		return http.StatusOK
	}
	msg := err.Error()
	if strings.Contains(msg, "no TTS pool for model") || strings.Contains(msg, "no default TTS pool") {
		return http.StatusBadRequest
	}
	return http.StatusInternalServerError
}

// listVoices aggregates the pool's registered voices (GET /v1/voices).
func (n *node) listVoices(w http.ResponseWriter, r *http.Request) {
	voices, err := n.local.ListVoices(r.Context())
	if err != nil {
		http.Error(w, err.Error(), http.StatusServiceUnavailable)
		return
	}
	writeJSON(w, http.StatusOK, inference.VoicesResponse{Voices: voices})
}

// uploadVoice registers a reference voice, fanning out to ALL live instances
// (POST /v1/voices, multipart name/transcript/voice). Mirrors the crispasr
// contract scripts/tts_gen speaks. NOTE (decision #1): a replica scaled up after
// this call must be re-uploaded to; the fan-out does not auto-heal it.
func (n *node) uploadVoice(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseMultipartForm(64 << 20); err != nil {
		http.Error(w, "bad multipart: "+err.Error(), http.StatusBadRequest)
		return
	}
	name := strings.TrimSpace(r.FormValue("name"))
	if name == "" {
		http.Error(w, "missing 'name'", http.StatusBadRequest)
		return
	}
	// A WAV voice can only be cloned with its reference transcription (--ref-text):
	// without it synthesis fails later at runtime. Reject a textless upload here so
	// it fails fast instead of registering a voice that blows up at synth time.
	transcript := strings.TrimSpace(r.FormValue("transcript"))
	if transcript == "" {
		http.Error(w, "transcript is required to register a voice", http.StatusBadRequest)
		return
	}
	file, hdr, err := r.FormFile("voice")
	if err != nil {
		http.Error(w, "missing 'voice' file", http.StatusBadRequest)
		return
	}
	defer file.Close()
	wav, err := io.ReadAll(file)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	results, err := n.local.UploadVoice(r.Context(), name, transcript, wav, filepath.Base(hdr.Filename))
	status := http.StatusCreated
	if err != nil {
		// Empty results = no live instance to register on (local unavailability);
		// non-empty = instances present but all rejected (upstream failure).
		if len(results) == 0 {
			status = http.StatusServiceUnavailable
		} else {
			status = http.StatusBadGateway
		}
	}
	body := map[string]any{"name": name, "results": results}
	if err != nil {
		body["error"] = err.Error()
	}
	writeJSON(w, status, body)
}

// transcriptions is the OpenAI ASR route (POST /v1/audio/transcriptions):
// multipart file[, model, language] -> JSON {text}. Raw transcript only; the
// rich intent path stays on /v1/interpret.
func (n *node) transcriptions(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseMultipartForm(64 << 20); err != nil {
		http.Error(w, "bad multipart: "+err.Error(), http.StatusBadRequest)
		return
	}
	file, hdr, err := r.FormFile(inference.TranscribeFileField)
	if err != nil {
		http.Error(w, "missing 'file'", http.StatusBadRequest)
		return
	}
	defer file.Close()

	if err := os.MkdirAll(n.workDir, 0o755); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	tmp := filepath.Join(n.workDir, "asr_in_"+time.Now().Format("20060102_150405")+"_"+filepath.Base(hdr.Filename))
	out, err := os.Create(tmp)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer os.Remove(tmp) // registered before the copy so an aborted upload can't orphan the temp file
	if _, err := io.Copy(out, file); err != nil {
		out.Close()
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	out.Close()

	text, err := n.local.Transcribe(r.Context(), tmp, r.FormValue("model"), r.FormValue("language"))
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	writeJSON(w, http.StatusOK, inference.TranscriptionResponse{Text: text})
}

func (n *node) interpret(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseMultipartForm(64 << 20); err != nil {
		http.Error(w, "bad multipart: "+err.Error(), http.StatusBadRequest)
		return
	}
	var req inference.InterpretRequest
	if meta := r.FormValue(inference.InterpretMetaField); meta != "" {
		if err := json.Unmarshal([]byte(meta), &req); err != nil {
			http.Error(w, "bad meta: "+err.Error(), http.StatusBadRequest)
			return
		}
	}
	file, hdr, err := r.FormFile(inference.InterpretAudioField)
	if err != nil {
		http.Error(w, "missing audio", http.StatusBadRequest)
		return
	}
	defer file.Close()

	if err := os.MkdirAll(n.workDir, 0o755); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	tmp := filepath.Join(n.workDir, "in_"+time.Now().Format("20060102_150405")+"_"+filepath.Base(hdr.Filename))
	out, err := os.Create(tmp)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer os.Remove(tmp) // registered before the copy so an aborted upload can't orphan the temp file
	if _, err := io.Copy(out, file); err != nil {
		out.Close()
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	out.Close()

	result, err := n.eng.Interpret(r.Context(), inference.AudioInput{Path: tmp}, req)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(result)
}

// --- admin API: live replica supervision ---------------------------------

// replicasResponse is the shared body for all /admin/replicas responses.
type replicasResponse struct {
	Replicas []local.ReplicaStatus `json:"replicas"`
	Error    string                `json:"error,omitempty"`
}

func (n *node) adminList(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, replicasResponse{Replicas: n.local.Replicas(r.Context())})
}

// adminSet sets the desired replica count per role, e.g. {"tts": 3}. An optional
// ?model=NAME query scopes the change to one model's pool (default: the "" pool).
func (n *node) adminSet(w http.ResponseWriter, r *http.Request) {
	var want map[string]int
	if err := json.NewDecoder(r.Body).Decode(&want); err != nil {
		http.Error(w, "bad request: "+err.Error(), http.StatusBadRequest)
		return
	}
	model := strings.TrimSpace(r.URL.Query().Get("model"))
	for role, n2 := range want {
		if _, err := n.local.SetReplicas(r.Context(), role, model, n2); err != nil {
			writeJSON(w, http.StatusBadRequest, replicasResponse{Replicas: n.local.Replicas(r.Context()), Error: err.Error()})
			return
		}
	}
	writeJSON(w, http.StatusOK, replicasResponse{Replicas: n.local.Replicas(r.Context())})
}

// adminDelete scales a (role, model) down by n (default 1):
// DELETE /admin/replicas/tts?n=2&model=NAME.
func (n *node) adminDelete(w http.ResponseWriter, r *http.Request) {
	role := r.PathValue("role")
	model := strings.TrimSpace(r.URL.Query().Get("model"))
	n2 := 1
	if q := strings.TrimSpace(r.URL.Query().Get("n")); q != "" {
		if v, err := strconv.Atoi(q); err == nil {
			n2 = v
		}
	}
	target := n.local.ManagedReplicasFor(role, model) - n2
	if target < 0 {
		target = 0
	}
	if _, err := n.local.SetReplicas(r.Context(), role, model, target); err != nil {
		writeJSON(w, http.StatusBadRequest, replicasResponse{Replicas: n.local.Replicas(r.Context()), Error: err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, replicasResponse{Replicas: n.local.Replicas(r.Context())})
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}
