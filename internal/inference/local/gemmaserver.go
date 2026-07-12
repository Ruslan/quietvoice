package local

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
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
		// temp 0 for determinism: same audio must not yield different intents
		// (one run kept "Claude", another dropped it — doc/eval-listen-gemma-drops-meaning.md).
		"temperature": 0.0,
		// Gemma emits a reasoning block before its answer; fidelity mode returns the
		// FULL message (no ~100-token cap), so give ample room for think + a complete,
		// non-truncated answer. Env-tunable: the assisted ensemble with 2 long ASR refs
		// can reason past 3072 and hit the cap BEFORE emitting the answer -> empty content
		// (verified on MI300X 2026-07-12). Bump GEMMA_MAX_TOKENS (and GEMMA_CTX_SIZE) on a
		// big-VRAM node. Must stay <= the gemma server's --ctx-size.
		"max_tokens": gemmaMaxTokens(),
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
				Content          json.RawMessage `json:"content"`
				ReasoningContent json.RawMessage `json:"reasoning_content"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return "", fmt.Errorf("decode chat response: %w", err)
	}
	if len(out.Choices) == 0 {
		return "", fmt.Errorf("chat response had no choices")
	}
	text, err := decodeChatContent(out.Choices[0].Message.Content)
	if err != nil {
		return "", err
	}
	// Diagnostic: when the answer is empty but the model reasoned (some llama.cpp
	// builds route <think> to reasoning_content), the request hit max_tokens mid-think.
	// Surface it loudly instead of silently returning "" — raise GEMMA_MAX_TOKENS/CTX_SIZE.
	if strings.TrimSpace(text) == "" {
		rc, _ := decodeChatContent(out.Choices[0].Message.ReasoningContent)
		log.Printf("[gemma pool] empty content (finish=%q); reasoning_content=%d chars — likely hit max_tokens mid-reasoning, raise GEMMA_MAX_TOKENS/GEMMA_CTX_SIZE", out.Choices[0].FinishReason, len(strings.TrimSpace(rc)))
	}
	return text, nil
}

// gemmaMaxTokens is the interpret generation cap (env GEMMA_MAX_TOKENS, default 3072).
func gemmaMaxTokens() int {
	if v := os.Getenv("GEMMA_MAX_TOKENS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return n
		}
	}
	return 3072
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
