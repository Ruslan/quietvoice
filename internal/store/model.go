// Package store holds QuietVoice control-plane state: MCP sessions, the voice
// requests an agent is waiting on, incoming Telegram voice messages, and the
// history of spoken updates. The Store interface hides the backend; the MVP
// uses a zero-dependency JSON file so the orchestrator stays portable and
// cgo-free. A SQLite implementation can drop in behind the same interface later.
package store

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"time"
)

// VoiceStatus is the lifecycle of an incoming voice message (handoff §8).
type VoiceStatus string

const (
	VoicePending    VoiceStatus = "pending"    // recorded before any agent asked
	VoiceClaimed    VoiceStatus = "claimed"    // bound to a waiting request
	VoiceProcessing VoiceStatus = "processing" // being interpreted
	VoiceConsumed   VoiceStatus = "consumed"   // delivered to an agent
	VoiceFailed     VoiceStatus = "failed"
	VoiceExpired    VoiceStatus = "expired"
	VoiceDiscarded  VoiceStatus = "discarded"
)

// RequestStatus is the lifecycle of an agent's listen_voice request.
type RequestStatus string

const (
	RequestWaiting   RequestStatus = "waiting"
	RequestCompleted RequestStatus = "completed"
	RequestExpired   RequestStatus = "expired"
	RequestCancelled RequestStatus = "cancelled"
)

// Session is an MCP client session bound to a Telegram chat.
type Session struct {
	ID             string    `json:"id"`
	ClientName     string    `json:"client_name,omitempty"`
	ProjectName    string    `json:"project_name,omitempty"`
	TelegramChatID int64     `json:"telegram_chat_id,omitempty"`
	CreatedAt      time.Time `json:"created_at"`
	LastSeenAt     time.Time `json:"last_seen_at"`
}

// VoiceRequest is created each time an agent calls listen_voice and blocks.
type VoiceRequest struct {
	ID          string        `json:"id"`
	SessionID   string        `json:"session_id"`
	PromptText  string        `json:"prompt_text,omitempty"`
	Status      RequestStatus `json:"status"`
	CreatedAt   time.Time     `json:"created_at"`
	ExpiresAt   time.Time     `json:"expires_at"`
	CompletedAt *time.Time    `json:"completed_at,omitempty"`
}

// VoiceMessage is an incoming Telegram voice note.
type VoiceMessage struct {
	ID                string      `json:"id"`
	TelegramChatID    int64       `json:"telegram_chat_id"`
	TelegramMessageID int64       `json:"telegram_message_id"`
	TelegramFileID    string      `json:"telegram_file_id"`
	LocalPath         string      `json:"local_path"`
	Status            VoiceStatus `json:"status"`
	SessionID         string      `json:"session_id,omitempty"`
	RequestID         string      `json:"request_id,omitempty"`
	CreatedAt         time.Time   `json:"created_at"`
	ClaimedAt         *time.Time  `json:"claimed_at,omitempty"`
	ConsumedAt        *time.Time  `json:"consumed_at,omitempty"`
}

// SpokenUpdate records a say() so it can seed context for later interpretation.
type SpokenUpdate struct {
	ID                string    `json:"id"`
	SessionID         string    `json:"session_id,omitempty"`
	Text              string    `json:"text"`
	AudioPath         string    `json:"audio_path,omitempty"`
	TelegramMessageID int64     `json:"telegram_message_id,omitempty"`
	CreatedAt         time.Time `json:"created_at"`
}

// NewID returns a short unique id with the given prefix, e.g. "voice_ab12cd34".
func NewID(prefix string) string {
	var b [6]byte
	_, _ = rand.Read(b[:])
	return fmt.Sprintf("%s_%s", prefix, hex.EncodeToString(b[:]))
}
