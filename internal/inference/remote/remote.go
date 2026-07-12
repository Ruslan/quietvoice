// Package remote implements inference.Engine as an HTTP client to a standalone
// inference node (cmd/inferenced). This is the "split" deployment: the
// orchestrator (control plane) runs anywhere, and the GPU work happens on a
// separate, portable inference node reachable over the network.
package remote

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"quietvoice/internal/inference"
)

// Config configures the remote inference client.
type Config struct {
	BaseURL string // e.g. http://gpu-node:9095
	Token   string // optional bearer token
	WorkDir string // where synthesized audio is written locally

	// Reference voice, provisioned by the control plane. The node can be deployed
	// with no voice; the MCP uploads this one on demand (see EnsureVoice).
	VoiceName       string // default voice for synthesis; the name registered on the node
	VoiceWav        string // local reference wav uploaded to the node when it lacks VoiceName
	VoiceTranscript string // reference transcript (crispasr requires it to register a wav voice)
}

// Engine talks to an inferenced node over HTTP.
type Engine struct {
	cfg    Config
	client *http.Client
}

// New returns a remote Engine.
func New(cfg Config) *Engine {
	cfg.BaseURL = strings.TrimRight(cfg.BaseURL, "/")
	if cfg.WorkDir == "" {
		cfg.WorkDir = os.TempDir()
	}
	return &Engine{
		cfg:    cfg,
		client: &http.Client{Timeout: 5 * time.Minute},
	}
}

func (e *Engine) do(ctx context.Context, method, path string, body io.Reader, contentType string) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, method, e.cfg.BaseURL+path, body)
	if err != nil {
		return nil, err
	}
	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}
	if e.cfg.Token != "" {
		req.Header.Set("Authorization", "Bearer "+e.cfg.Token)
	}
	return e.client.Do(req)
}

// Synthesize posts text in the OpenAI /v1/audio/speech shape and stores the
// returned WAV bytes locally.
func (e *Engine) Synthesize(ctx context.Context, req inference.SynthesizeRequest) (*inference.SynthesizeResult, error) {
	voice := req.Voice
	if voice == "" {
		voice = e.cfg.VoiceName // fall back to the control-plane's configured voice
	}
	payload, _ := json.Marshal(inference.SpeechRequest{
		Model:          req.Model,
		Input:          req.Text,
		Voice:          voice,
		ResponseFormat: "wav",
	})
	resp, err := e.do(ctx, http.MethodPost, inference.RouteSpeech, bytes.NewReader(payload), "application/json")
	if err != nil {
		return nil, fmt.Errorf("remote synthesize: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("remote synthesize: status %d: %s", resp.StatusCode, readErr(resp.Body))
	}

	if err := os.MkdirAll(e.cfg.WorkDir, 0o755); err != nil {
		return nil, err
	}
	out := filepath.Join(e.cfg.WorkDir, fmt.Sprintf("tts_%d.wav", time.Now().UnixNano()))
	f, err := os.Create(out)
	if err != nil {
		return nil, err
	}
	if _, err := io.Copy(f, resp.Body); err != nil {
		f.Close()
		return nil, err
	}
	if err := f.Close(); err != nil {
		return nil, err
	}
	return &inference.SynthesizeResult{WavPath: out}, nil
}

// Interpret uploads the audio file plus context metadata and parses the result.
func (e *Engine) Interpret(ctx context.Context, audio inference.AudioInput, req inference.InterpretRequest) (*inference.InterpretResult, error) {
	f, err := os.Open(audio.Path)
	if err != nil {
		return nil, fmt.Errorf("remote interpret: open audio: %w", err)
	}
	defer f.Close()

	var body bytes.Buffer
	mw := multipart.NewWriter(&body)
	meta, _ := json.Marshal(req)
	if err := mw.WriteField(inference.InterpretMetaField, string(meta)); err != nil {
		return nil, err
	}
	part, err := mw.CreateFormFile(inference.InterpretAudioField, filepath.Base(audio.Path))
	if err != nil {
		return nil, err
	}
	if _, err := io.Copy(part, f); err != nil {
		return nil, err
	}
	if err := mw.Close(); err != nil {
		return nil, err
	}

	resp, err := e.do(ctx, http.MethodPost, inference.RouteInterpret, &body, mw.FormDataContentType())
	if err != nil {
		return nil, fmt.Errorf("remote interpret: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("remote interpret: status %d: %s", resp.StatusCode, readErr(resp.Body))
	}
	var result inference.InterpretResult
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("remote interpret: decode result: %w", err)
	}
	return &result, nil
}

// EnsureVoice registers the configured reference voice on the node when the node
// does not already have it. Idempotent and cheap (a GET, then a POST only when
// missing), so the control plane calls it before every Say: whenever the MCP is
// pointed at a fresh or restarted node, the voice is transparently re-uploaded.
// A no-op when no voice (or no wav) is configured — the node then uses whatever
// voice it already has.
func (e *Engine) EnsureVoice(ctx context.Context) error {
	if e.cfg.VoiceName == "" || e.cfg.VoiceWav == "" {
		return nil
	}
	if has, err := e.hasVoice(ctx, e.cfg.VoiceName); err == nil && has {
		return nil
	}
	if e.cfg.VoiceTranscript == "" {
		return fmt.Errorf("ensure voice %q: a transcript (VOICE_TXT) is required to register a wav voice", e.cfg.VoiceName)
	}
	return e.uploadVoice(ctx)
}

// hasVoice reports whether name is already registered on the node.
func (e *Engine) hasVoice(ctx context.Context, name string) (bool, error) {
	voices, err := e.ListVoices(ctx)
	if err != nil {
		return false, err
	}
	for _, v := range voices {
		if v == name {
			return true, nil
		}
	}
	return false, nil
}

// ListVoices returns the voice names registered on the node (implements
// inference.VoiceLister, used by per-session voice rotation).
func (e *Engine) ListVoices(ctx context.Context) ([]string, error) {
	resp, err := e.do(ctx, http.MethodGet, inference.RouteVoices, nil, "")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("list voices: status %d", resp.StatusCode)
	}
	var vr inference.VoicesResponse
	if err := json.NewDecoder(resp.Body).Decode(&vr); err != nil {
		return nil, err
	}
	return vr.Voices, nil
}

// uploadVoice POSTs the reference wav + transcript to /v1/voices (multipart
// name/transcript/voice), mirroring the crispasr registration contract.
func (e *Engine) uploadVoice(ctx context.Context) error {
	f, err := os.Open(e.cfg.VoiceWav)
	if err != nil {
		return fmt.Errorf("ensure voice: open wav: %w", err)
	}
	defer f.Close()

	var body bytes.Buffer
	mw := multipart.NewWriter(&body)
	if err := mw.WriteField("name", e.cfg.VoiceName); err != nil {
		return err
	}
	if err := mw.WriteField("transcript", e.cfg.VoiceTranscript); err != nil {
		return err
	}
	part, err := mw.CreateFormFile("voice", filepath.Base(e.cfg.VoiceWav))
	if err != nil {
		return err
	}
	if _, err := io.Copy(part, f); err != nil {
		return err
	}
	if err := mw.Close(); err != nil {
		return err
	}

	resp, err := e.do(ctx, http.MethodPost, inference.RouteVoices, &body, mw.FormDataContentType())
	if err != nil {
		return fmt.Errorf("ensure voice: upload: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusCreated && resp.StatusCode != http.StatusOK {
		return fmt.Errorf("ensure voice %q: upload status %d: %s", e.cfg.VoiceName, resp.StatusCode, readErr(resp.Body))
	}
	return nil
}

// Health pings the inference node.
func (e *Engine) Health(ctx context.Context) error {
	resp, err := e.do(ctx, http.MethodGet, inference.RouteHealth, nil, "")
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("inference node unhealthy: status %d", resp.StatusCode)
	}
	return nil
}

func readErr(r io.Reader) string {
	b, _ := io.ReadAll(io.LimitReader(r, 512))
	return strings.TrimSpace(string(b))
}
