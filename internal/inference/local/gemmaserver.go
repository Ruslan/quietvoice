package local

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
)

// This file speaks the llama-server OpenAI chat contract for the hot `gemma`
// replica (Gemma 4 E4B + mmproj), so the interpret step can route to a process
// the supervisor keeps hot instead of cold-spawning llama-mtmd-cli per request
// (the CLI is not even on PATH on the GPU nodes — only llama-server is).
//
// llama-server understands audio through the standard chat body: an audio part
// is passed inline alongside the text, and the server applies Gemma's chat
// template itself (so no --jinja / -sys / -p CLI plumbing is needed):
//
//	POST /v1/chat/completions
//	{
//	  "messages": [
//	    {"role":"system","content":"<systemPrompt>"},
//	    {"role":"user","content":[
//	      {"type":"text","text":"<userPrompt>"},
//	      {"type":"input_audio","input_audio":{"data":"<base64 wav>","format":"wav"}}
//	    ]}
//	  ],
//	  "temperature": 0.2,
//	  "max_tokens": 512
//	}
//	-> {"choices":[{"message":{"content":"<answer>"}}]}

// gemmaChatServer sends the interpret chat request (system prompt + user text +
// the audio inline) to one gemma llama-server worker and returns the raw
// assistant message content (Gemma's `<thought>` block, if any, is left intact
// for extractGemmaFinal to strip). A dead-worker (dial) failure is surfaced so
// the pool can evict the worker; a non-2xx answer is a *ttsStatusError (alive
// worker, retry elsewhere WITHOUT eviction) — the same classification the tts
// and asr clients use.
func gemmaChatServer(ctx context.Context, client *http.Client, base, token, wavPath, systemPrompt, userPrompt string) (string, error) {
	wav, err := os.ReadFile(wavPath)
	if err != nil {
		return "", err
	}
	payload := map[string]any{
		"messages": []any{
			map[string]any{"role": "system", "content": systemPrompt},
			map[string]any{"role": "user", "content": []any{
				map[string]any{"type": "text", "text": userPrompt},
				map[string]any{"type": "input_audio", "input_audio": map[string]any{
					"data":   base64.StdEncoding.EncodeToString(wav),
					"format": "wav",
				}},
			}},
		},
		"temperature": 0.2,
		// Gemma E4B emits a reasoning block before its answer; 512 was consumed
		// entirely by reasoning on assisted reconcile, cutting off the final
		// answer (empty content). Give room for think + concise answer.
		"max_tokens": 1536,
	}
	body, _ := json.Marshal(payload)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, joinURL(base, "/v1/chat/completions"), bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
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
		return "", &ttsStatusError{status: resp.StatusCode, snippet: strings.TrimSpace(string(snippet))}
	}
	var out struct {
		Choices []struct {
			Message struct {
				Content json.RawMessage `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return "", fmt.Errorf("decode chat response: %w", err)
	}
	if len(out.Choices) == 0 {
		return "", fmt.Errorf("chat response had no choices")
	}
	return decodeChatContent(out.Choices[0].Message.Content)
}

// decodeChatContent extracts the assistant text from a chat message `content`
// field. llama-server builds differ: some encode it as a plain JSON string,
// others as an OpenAI-style array of typed parts ([{"type":"text","text":…}]).
// Both are handled so interpret is robust across builds.
func decodeChatContent(raw json.RawMessage) (string, error) {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 || string(raw) == "null" {
		return "", nil
	}
	switch raw[0] {
	case '"': // plain string
		var s string
		if err := json.Unmarshal(raw, &s); err != nil {
			return "", fmt.Errorf("decode chat content string: %w", err)
		}
		return s, nil
	case '[': // array of typed parts; concatenate the text parts
		var parts []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		}
		if err := json.Unmarshal(raw, &parts); err != nil {
			return "", fmt.Errorf("decode chat content parts: %w", err)
		}
		var b strings.Builder
		for _, p := range parts {
			b.WriteString(p.Text)
		}
		return b.String(), nil
	default:
		return "", fmt.Errorf("unexpected chat content shape: %s", tail(string(raw)))
	}
}
