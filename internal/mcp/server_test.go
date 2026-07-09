package mcp

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

// fakeSpeaker implements the Speaker interface for exercising handleToolCall.
type fakeSpeaker struct {
	intent      string
	fromPending bool
}

func (f fakeSpeaker) Say(ctx context.Context, sessionID, text string) (string, error) {
	return "ok", nil
}
func (f fakeSpeaker) ListenVoice(ctx context.Context, sessionID, promptText string) (string, bool, error) {
	return f.intent, f.fromPending, nil
}

// resultText extracts the single text content block from a tools/call result.
func resultText(t *testing.T, result any) string {
	t.Helper()
	m, ok := result.(map[string]any)
	if !ok {
		t.Fatalf("result is not a map: %T", result)
	}
	content, ok := m["content"].([]any)
	if !ok || len(content) == 0 {
		t.Fatalf("result has no content: %v", m)
	}
	block, ok := content[0].(map[string]any)
	if !ok {
		t.Fatalf("content[0] is not a map: %T", content[0])
	}
	text, _ := block["text"].(string)
	return text
}

func callListen(t *testing.T, svc Speaker) string {
	t.Helper()
	srv := New(svc, "")
	params, _ := json.Marshal(toolCallParams{
		Name:      "listen_voice",
		Arguments: json.RawMessage(`{"prompt_text":"which client?"}`),
	})
	result, rerr := srv.handleToolCall(context.Background(), "sess", params)
	if rerr != nil {
		t.Fatalf("handleToolCall: %+v", rerr)
	}
	return resultText(t, result)
}

func TestListenVoiceResultMarksPending(t *testing.T) {
	text := callListen(t, fakeSpeaker{intent: "PATCH NEW CLIENT ONLY", fromPending: true})
	if !strings.Contains(text, pendingMarker) {
		t.Fatalf("pending result missing marker %q; got %q", pendingMarker, text)
	}
	if !strings.Contains(text, "PATCH NEW CLIENT ONLY") {
		t.Fatalf("pending result missing intent; got %q", text)
	}
}

func TestListenVoiceResultOmitsMarkerWhenLive(t *testing.T) {
	text := callListen(t, fakeSpeaker{intent: "PATCH NEW CLIENT ONLY", fromPending: false})
	if strings.Contains(text, pendingMarker) {
		t.Fatalf("live result must not contain pending marker; got %q", text)
	}
	if text != "PATCH NEW CLIENT ONLY" {
		t.Fatalf("live result = %q, want plain intent", text)
	}
}
