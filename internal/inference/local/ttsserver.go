package local

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"syscall"
)

// ttsStatusError is returned by ttsSpeak when the server answered but with a
// non-2xx status (the process is alive — e.g. it is missing the requested voice
// — so the caller should retry elsewhere WITHOUT evicting the worker).
type ttsStatusError struct {
	status  int
	snippet string
}

func (e *ttsStatusError) Error() string {
	return fmt.Sprintf("POST /v1/audio/speech: status %d: %s", e.status, e.snippet)
}

// isDeadWorkerErr reports whether err means the worker process is gone (a dial /
// connection-refused failure), so the pool should evict it. A non-2xx status
// (ttsStatusError) is explicitly NOT dead — the server answered.
func isDeadWorkerErr(err error) bool {
	var statusErr *ttsStatusError
	if errors.As(err, &statusErr) {
		return false
	}
	var opErr *net.OpError
	if errors.As(err, &opErr) {
		return true
	}
	return errors.Is(err, syscall.ECONNREFUSED)
}

// This file speaks the `crispasr --server` HTTP contract (the same one the Ruby
// tts_gen pipeline uses), so the Go inference plane can drive a pool of hot TTS
// servers instead of cold-spawning the crispasr CLI per request:
//
//	POST /v1/audio/speech  {input, voice, response_format:"wav", ...} -> WAV bytes
//	GET  /v1/voices        -> {"voices":[...]}   (also our readiness probe)

// ttsSpeak asks one crispasr server to synthesize text with the given bare voice
// name (registered in the server's --voice-dir) and returns the raw WAV bytes.
func ttsSpeak(ctx context.Context, client *http.Client, base, token, voice, text string) ([]byte, error) {
	payload := map[string]any{
		"input":             text,
		"voice":             voice,
		"response_format":   "wav",
		"spoken_disclaimer": false,
	}
	body, _ := json.Marshal(payload)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, joinURL(base, "/v1/audio/speech"), bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		snippet, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return nil, &ttsStatusError{status: resp.StatusCode, snippet: strings.TrimSpace(string(snippet))}
	}
	wav, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if len(wav) == 0 {
		return nil, fmt.Errorf("POST /v1/audio/speech: empty audio")
	}
	return wav, nil
}

// Per-role readiness probe paths. tts/asr are crispasr; gemma is llama-server.
// crispasr --server serves /v1/voices (tts) and /health; llama-server serves
// /health. NOTE: if a crispasr build does not expose /health for the whisper
// backend, switch asrHealthPath to "/v1/models" (also listed at startup) — see
// inferenced-unified-api-plan.md.
const (
	ttsHealthPath   = "/v1/voices"
	asrHealthPath   = "/health"
	gemmaHealthPath = "/health"
)

// roleHealthPath returns the readiness/health probe path for a role.
func roleHealthPath(role string) string {
	switch role {
	case "asr":
		return asrHealthPath
	case "gemma":
		return gemmaHealthPath
	default:
		return ttsHealthPath
	}
}

// healthGet is the generic readiness probe: a GET on url returning 200 means the
// server is answering. Used to gate a freshly launched replica (role-appropriate
// path) and to report health in the admin listing.
func healthGet(ctx context.Context, client *http.Client, url, token string) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 4096))
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("GET %s: status %d", url, resp.StatusCode)
	}
	return nil
}

// ttsHealth is the tts readiness probe: GET /v1/voices returning 200 means the
// model is loaded and the server is answering.
func ttsHealth(ctx context.Context, client *http.Client, base, token string) error {
	return healthGet(ctx, client, joinURL(base, ttsHealthPath), token)
}

// joinURL concatenates a base URL and an absolute path without doubling slashes.
func joinURL(base, path string) string {
	return strings.TrimRight(base, "/") + path
}
