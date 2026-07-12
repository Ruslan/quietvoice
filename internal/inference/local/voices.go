package local

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"strings"
)

// This file speaks the voice-registry half of the `crispasr --server` OpenAI
// contract, so inferenced can aggregate/register voices across its hot pool:
//
//	GET  /v1/voices  -> {"voices":[...]}                 (names, or {name:...} objects)
//	POST /v1/voices  (multipart name/transcript/voice)   201 created / 409 exists
//
// Voice registration FANS OUT to every live instance: the cache is per-process, so
// a voice must be uploaded to each replica. Uploaded voices are also remembered in
// the Engine (storedVoice) so a replica scaled up AFTER an upload is auto-provisioned
// with the same voices as its neighbors on scale-up (see provisionVoices, called from
// SetReplicas) — no manual re-upload needed.

// VoiceUploadResult is the per-instance outcome of a fan-out voice upload.
type VoiceUploadResult struct {
	URL    string `json:"url"`
	Status int    `json:"status"`          // HTTP status (201 created, 409 already present)
	Error  string `json:"error,omitempty"` // set when the instance failed
}

// listVoicesFrom fetches the voice names registered on one crispasr server.
func listVoicesFrom(ctx context.Context, client *http.Client, base, token string) ([]string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, joinURL(base, "/v1/voices"), nil)
	if err != nil {
		return nil, err
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("GET /v1/voices: status %d", resp.StatusCode)
	}
	var data struct {
		Voices []json.RawMessage `json:"voices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		return nil, err
	}
	out := make([]string, 0, len(data.Voices))
	for _, raw := range data.Voices {
		// Each entry is either a bare string or an object with a "name" field.
		var s string
		if json.Unmarshal(raw, &s) == nil && s != "" {
			out = append(out, s)
			continue
		}
		var obj struct {
			Name string `json:"name"`
		}
		if json.Unmarshal(raw, &obj) == nil && obj.Name != "" {
			out = append(out, obj.Name)
		}
	}
	return out, nil
}

// uploadVoiceTo registers a reference voice on one crispasr server as multipart
// (name/transcript/voice-file), mirroring what scripts/tts_gen's upload_voice
// sends. 201 = created, 409 = already present; both are treated as success.
func uploadVoiceTo(ctx context.Context, client *http.Client, base, token, name, transcript string, wav []byte, filename string) (int, error) {
	if strings.TrimSpace(transcript) == "" {
		return 0, fmt.Errorf("transcript is required to register voice %q", name)
	}
	if filename == "" {
		filename = name + ".wav"
	}
	var body bytes.Buffer
	mw := multipart.NewWriter(&body)
	if err := mw.WriteField("name", name); err != nil {
		return 0, err
	}
	if err := mw.WriteField("transcript", transcript); err != nil {
		return 0, err
	}
	part, err := mw.CreateFormFile("voice", filename)
	if err != nil {
		return 0, err
	}
	if _, err := part.Write(wav); err != nil {
		return 0, err
	}
	if err := mw.Close(); err != nil {
		return 0, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, joinURL(base, "/v1/voices"), &body)
	if err != nil {
		return 0, err
	}
	req.Header.Set("Content-Type", mw.FormDataContentType())
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	resp, err := client.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusCreated || resp.StatusCode == http.StatusConflict {
		return resp.StatusCode, nil
	}
	snippet, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
	return resp.StatusCode, fmt.Errorf("POST /v1/voices: status %d: %s", resp.StatusCode, strings.TrimSpace(string(snippet)))
}

// ListVoices returns the voices registered in the pool. The voice-dir is shared
// across replicas, so it queries live instances in turn and returns the first
// successful answer (a union would be redundant).
func (e *Engine) ListVoices(ctx context.Context) ([]string, error) {
	ws := e.allTTSWorkers()
	if len(ws) == 0 {
		return nil, fmt.Errorf("no TTS server in pool to list voices")
	}
	var lastErr error
	for _, w := range ws {
		voices, err := listVoicesFrom(ctx, e.httpClient, w.url, e.cfg.TTSServerToken)
		if err == nil {
			return voices, nil
		}
		lastErr = err
	}
	return nil, fmt.Errorf("no instance could list voices: %w", lastErr)
}

// UploadVoice registers a reference voice on EVERY live instance in the pool
// (decision #1: idempotent fan-out, 409 = already there). It returns a
// per-instance result list and an error only when no instance accepted the
// voice. A replica scaled up after this call must be re-uploaded to.
func (e *Engine) UploadVoice(ctx context.Context, name, transcript string, wav []byte, filename string) ([]VoiceUploadResult, error) {
	// A WAV voice needs its reference transcription (crispasr --ref-text) to clone:
	// without it the upload "succeeds" but every later synthesis fails. Reject an
	// empty transcript up front, before fanning out to any instance.
	if strings.TrimSpace(transcript) == "" {
		return nil, fmt.Errorf("transcript is required to register voice %q", name)
	}
	ws := e.allTTSWorkers()
	if len(ws) == 0 {
		return nil, fmt.Errorf("no TTS server in pool to register voice %q", name)
	}
	results := make([]VoiceUploadResult, 0, len(ws))
	ok := 0
	for _, w := range ws {
		status, err := uploadVoiceTo(ctx, e.httpClient, w.url, e.cfg.TTSServerToken, name, transcript, wav, filename)
		res := VoiceUploadResult{URL: w.url, Status: status}
		if err != nil {
			res.Error = err.Error()
		} else {
			ok++
		}
		results = append(results, res)
	}
	if ok == 0 {
		return results, fmt.Errorf("voice %q rejected by all %d instance(s)", name, len(ws))
	}
	// Remember it so a tts replica scaled up later gets the same voices (crispasr
	// caches per-process). Copy the wav — the caller's buffer may be reused.
	e.rememberVoice(name, transcript, wav, filename)
	return results, nil
}

// rememberVoice records a successfully-registered reference voice for later
// scale-up replay. The wav is copied so the caller may reuse its buffer.
func (e *Engine) rememberVoice(name, transcript string, wav []byte, filename string) {
	cp := make([]byte, len(wav))
	copy(cp, wav)
	e.voiceMu.Lock()
	defer e.voiceMu.Unlock()
	if e.voices == nil {
		e.voices = map[string]storedVoice{}
	}
	e.voices[name] = storedVoice{name: name, transcript: transcript, wav: cp, filename: filename}
}

// provisionVoices re-uploads every remembered reference voice to a single freshly
// launched tts replica, so it matches its neighbors. Called from SetReplicas after a
// tts scale-up. Best-effort: a failed replay is logged, not fatal — the scale-up
// still succeeds, and a later Synthesize can still recover via the pool's retry.
func (e *Engine) provisionVoices(ctx context.Context, url string) {
	e.voiceMu.Lock()
	pending := make([]storedVoice, 0, len(e.voices))
	for _, v := range e.voices {
		pending = append(pending, v)
	}
	e.voiceMu.Unlock()
	for _, v := range pending {
		if _, err := uploadVoiceTo(ctx, e.httpClient, url, e.cfg.TTSServerToken, v.name, v.transcript, v.wav, v.filename); err != nil {
			log.Printf("[tts pool] scale-up voice replay of %q -> %s failed: %v", v.name, url, err)
		}
	}
}
