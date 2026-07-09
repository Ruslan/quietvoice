package local

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
)

// This file speaks the ASR half of the `crispasr --server` OpenAI contract, so a
// hot `crispasr --server --backend whisper` replica can serve transcriptions over
// HTTP instead of the CLI cold-spawning per request:
//
//	POST /v1/audio/transcriptions  (multipart file[, model, language]) -> {"text": ...}

// asrTranscribeServer asks one crispasr ASR server to transcribe the audio file
// and returns the plain transcript text. model/lang are forwarded as the OpenAI
// form fields when non-empty.
func asrTranscribeServer(ctx context.Context, client *http.Client, base, token, audioPath, model, lang string) (string, error) {
	f, err := os.Open(audioPath)
	if err != nil {
		return "", err
	}
	defer f.Close()

	var body bytes.Buffer
	mw := multipart.NewWriter(&body)
	part, err := mw.CreateFormFile("file", filepath.Base(audioPath))
	if err != nil {
		return "", err
	}
	if _, err := io.Copy(part, f); err != nil {
		return "", err
	}
	if model != "" {
		if err := mw.WriteField("model", model); err != nil {
			return "", err
		}
	}
	if lang != "" {
		if err := mw.WriteField("language", lang); err != nil {
			return "", err
		}
	}
	if err := mw.Close(); err != nil {
		return "", err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, joinURL(base, "/v1/audio/transcriptions"), &body)
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", mw.FormDataContentType())
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		snippet, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return "", fmt.Errorf("POST /v1/audio/transcriptions: status %d: %s", resp.StatusCode, strings.TrimSpace(string(snippet)))
	}
	var out struct {
		Text string `json:"text"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return "", fmt.Errorf("decode transcription response: %w", err)
	}
	return strings.TrimSpace(out.Text), nil
}
