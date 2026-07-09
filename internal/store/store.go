package store

// Store is the control-plane persistence contract. All methods are safe for
// concurrent use.
type Store interface {
	// Sessions.
	UpsertSession(s *Session) error
	GetSession(id string) (*Session, error)

	// Voice requests (an agent waiting for a spoken reply).
	CreateRequest(r *VoiceRequest) error
	GetRequest(id string) (*VoiceRequest, error)
	UpdateRequest(r *VoiceRequest) error
	// WaitingRequests returns still-waiting requests, optionally filtered to a
	// chat (chatID == 0 means all chats).
	WaitingRequests(chatID int64) ([]*VoiceRequest, error)

	// Voice messages (incoming Telegram voice notes).
	AddVoiceMessage(m *VoiceMessage) error
	GetVoiceMessage(id string) (*VoiceMessage, error)
	UpdateVoiceMessage(m *VoiceMessage) error
	// OldestPendingVoice returns the oldest pending voice for a chat (chatID == 0
	// means any chat), or (nil, nil) when none exists.
	OldestPendingVoice(chatID int64) (*VoiceMessage, error)

	// Spoken updates (say history for context seeding).
	AddSpokenUpdate(u *SpokenUpdate) error
	RecentSpokenUpdates(sessionID string, n int) ([]*SpokenUpdate, error)

	Close() error
}
