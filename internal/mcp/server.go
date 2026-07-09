// Package mcp implements the QuietVoice MCP endpoint: a JSON-RPC 2.0 server
// exposing the say and listen_voice tools. It is pure control plane — it holds
// no GPU logic and delegates every tool call to the voice.Service. This is the
// evolved core of the original voice_mcp_server.go, with the macOS mic/speaker
// paths removed in favour of the Telegram transport.
package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
)

// Speaker is the control-plane action surface the MCP tools drive.
type Speaker interface {
	Say(ctx context.Context, sessionID, text string) (string, error)
	// ListenVoice returns the interpreted intent plus fromPending, which is true
	// when the intent came from a voice the user recorded BEFORE the agent asked.
	ListenVoice(ctx context.Context, sessionID, promptText string) (intent string, fromPending bool, err error)
}

// pendingMarker is prefixed to the listen_voice result text when the returned
// intent came from a pre-buffered (pending) voice, so the calling agent can tell
// the reply was recorded before it asked rather than captured live.
const pendingMarker = "ℹ️ (pending voice — recorded before you asked)"

// Server is the MCP JSON-RPC handler.
type Server struct {
	svc   Speaker
	token string // optional bearer token for the /rpc endpoint
}

// New builds an MCP server. token may be empty to disable auth (dev only).
func New(svc Speaker, token string) *Server {
	return &Server{svc: svc, token: token}
}

// Routes registers the MCP and health endpoints on mux.
func (s *Server) Routes(mux *http.ServeMux) {
	mux.HandleFunc("POST /rpc", s.handleRPC)
	mux.HandleFunc("GET /health", s.handleHealth)
}

// --- JSON-RPC types ---

type request struct {
	JSONRPC string          `json:"jsonrpc"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
	ID      json.RawMessage `json:"id"`
}

type response struct {
	JSONRPC string          `json:"jsonrpc"`
	Result  any             `json:"result,omitempty"`
	Error   *rpcError       `json:"error,omitempty"`
	ID      json.RawMessage `json:"id"`
}

type rpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

const (
	errParse          = -32700
	errInvalidRequest = -32600
	errMethodNotFound = -32601
	errInvalidParams  = -32602
	errInternal       = -32603
)

type toolCallParams struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

type sayArgs struct {
	Text string `json:"text"`
}

type listenVoiceArgs struct {
	PromptText string `json:"prompt_text"`
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"status":  "ok",
		"service": "quietvoice-mcp",
	})
}

func (s *Server) handleRPC(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	if s.token != "" && !strings.EqualFold(r.Header.Get("Authorization"), "Bearer "+s.token) {
		sendError(w, nil, errInvalidRequest, "unauthorized")
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		sendError(w, nil, errParse, "Parse error")
		return
	}
	var req request
	if err := json.Unmarshal(body, &req); err != nil {
		sendError(w, nil, errParse, "Invalid JSON-RPC format")
		return
	}
	if req.JSONRPC != "2.0" || req.Method == "" {
		sendError(w, req.ID, errInvalidRequest, "Invalid jsonrpc version or missing method")
		return
	}
	log.Printf("MCP request id=%s method=%s", string(req.ID), req.Method)

	sessionID := strings.TrimSpace(r.Header.Get("X-Voice-Client-Token"))
	ctx := r.Context()

	var result any
	var rerr *rpcError

	switch req.Method {
	case "initialize":
		result = map[string]any{
			"protocolVersion": "2024-11-05",
			"capabilities":    map[string]any{"tools": map[string]any{}},
			"serverInfo":      map[string]any{"name": "quietvoice-mcp", "version": "2.0.0"},
		}
	case "tools/list":
		result = toolsList()
	case "notifications/initialized", "notifications/cancelled":
		return
	case "tools/call":
		result, rerr = s.handleToolCall(ctx, sessionID, req.Params)
	default:
		rerr = &rpcError{Code: errMethodNotFound, Message: "Method not found"}
	}

	if isNotification(req.ID) {
		return
	}
	if rerr != nil {
		sendError(w, req.ID, rerr.Code, rerr.Message)
		return
	}
	_ = json.NewEncoder(w).Encode(response{JSONRPC: "2.0", Result: result, ID: req.ID})
}

func (s *Server) handleToolCall(ctx context.Context, sessionID string, params json.RawMessage) (any, *rpcError) {
	var p toolCallParams
	if err := json.Unmarshal(params, &p); err != nil {
		return nil, &rpcError{Code: errInvalidParams, Message: fmt.Sprintf("Invalid tools/call params: %v", err)}
	}
	if len(p.Arguments) == 0 {
		p.Arguments = json.RawMessage(`{}`)
	}

	switch p.Name {
	case "say":
		var a sayArgs
		if err := json.Unmarshal(p.Arguments, &a); err != nil || strings.TrimSpace(a.Text) == "" {
			return nil, &rpcError{Code: errInvalidParams, Message: "Argument 'text' is required and cannot be empty"}
		}
		status, err := s.svc.Say(ctx, sessionID, a.Text)
		if err != nil {
			return nil, &rpcError{Code: errInternal, Message: err.Error()}
		}
		return textResult(status), nil

	case "listen_voice":
		var a listenVoiceArgs
		if err := json.Unmarshal(p.Arguments, &a); err != nil {
			return nil, &rpcError{Code: errInvalidParams, Message: fmt.Sprintf("Invalid listen_voice arguments: %v", err)}
		}
		if len(a.PromptText) > 200 {
			return nil, &rpcError{Code: errInvalidParams, Message: "Argument 'prompt_text' must be at most 200 characters"}
		}
		intent, fromPending, err := s.svc.ListenVoice(ctx, sessionID, a.PromptText)
		if err != nil {
			return nil, &rpcError{Code: errInternal, Message: err.Error()}
		}
		text := intent
		if fromPending {
			// Keep the transcript/intent as the primary content; add the pending
			// note as a clearly separated prefix line.
			text = pendingMarker + "\n\n" + intent
		}
		return textResult(text), nil

	default:
		return nil, &rpcError{Code: errMethodNotFound, Message: "Unknown tool name"}
	}
}

func toolsList() map[string]any {
	return map[string]any{
		"tools": []any{
			map[string]any{
				"name":        "say",
				"description": "Provides the user with a concise spoken update, delivered as a Telegram voice message. This tool is a separate human-awareness channel, not a screen reader and not a verbatim copy of the assistant's normal response. Use it when the user would benefit from hearing a meaningful update about progress, a plan change, a discovered cause, a blocker, a risk, a completed milestone, or a need for attention. The input MUST already be rewritten by the caller for natural listening. Do not include raw git hashes, file paths, JSON, code, hex values, UUIDs, logs, or long technical identifiers literally.",
				"inputSchema": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"text": map[string]any{
							"type":        "string",
							"description": "Concise natural speech prepared specifically for listening. Summarize what matters to the user now; do not copy the normal textual response verbatim.",
						},
					},
					"required": []string{"text"},
				},
			},
			map[string]any{
				"name":        "listen_voice",
				"description": "Obtains the user's next spoken intent. Consumes a voice message the user pre-recorded in Telegram, or notifies the user and waits for one, then returns concise agent-ready text interpreted from the audio (not a blind transcript). Use it to get the user's decision or instruction when their spoken reply is the actual next message in the conversation.",
				"inputSchema": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"prompt_text": map[string]any{
							"type":        "string",
							"description": "Optional short text of the question being asked, shown to the user and used as context when interpreting their reply (max 200 chars).",
						},
					},
				},
			},
		},
	}
}

func textResult(text string) map[string]any {
	return map[string]any{
		"content": []any{map[string]any{"type": "text", "text": text}},
	}
}

func isNotification(id json.RawMessage) bool {
	t := strings.TrimSpace(string(id))
	return t == "" || t == "null"
}

func sendError(w http.ResponseWriter, id json.RawMessage, code int, message string) {
	log.Printf("MCP error id=%s code=%d msg=%s", string(id), code, message)
	_ = json.NewEncoder(w).Encode(response{JSONRPC: "2.0", Error: &rpcError{Code: code, Message: message}, ID: id})
}
