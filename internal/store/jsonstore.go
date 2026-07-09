package store

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"
)

// database is the on-disk shape of the JSON store.
type database struct {
	Sessions      map[string]*Session      `json:"sessions"`
	Requests      map[string]*VoiceRequest `json:"requests"`
	VoiceMessages map[string]*VoiceMessage `json:"voice_messages"`
	SpokenUpdates map[string]*SpokenUpdate `json:"spoken_updates"`
}

type jsonStore struct {
	path string
	mu   sync.Mutex
	db   database
}

// Open loads (or creates) a JSON-backed Store at path.
func Open(path string) (Store, error) {
	s := &jsonStore{
		path: path,
		db: database{
			Sessions:      map[string]*Session{},
			Requests:      map[string]*VoiceRequest{},
			VoiceMessages: map[string]*VoiceMessage{},
			SpokenUpdates: map[string]*SpokenUpdate{},
		},
	}
	data, err := os.ReadFile(path)
	if err == nil {
		if err := json.Unmarshal(data, &s.db); err != nil {
			return nil, fmt.Errorf("store: corrupt db %s: %w", path, err)
		}
		s.ensureMaps()
	} else if !os.IsNotExist(err) {
		return nil, fmt.Errorf("store: read %s: %w", path, err)
	}
	return s, nil
}

func (s *jsonStore) ensureMaps() {
	if s.db.Sessions == nil {
		s.db.Sessions = map[string]*Session{}
	}
	if s.db.Requests == nil {
		s.db.Requests = map[string]*VoiceRequest{}
	}
	if s.db.VoiceMessages == nil {
		s.db.VoiceMessages = map[string]*VoiceMessage{}
	}
	if s.db.SpokenUpdates == nil {
		s.db.SpokenUpdates = map[string]*SpokenUpdate{}
	}
}

// persist writes the whole db atomically. Caller holds s.mu.
func (s *jsonStore) persist() error {
	if s.path == "" {
		return nil
	}
	if dir := filepath.Dir(s.path); dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	data, err := json.MarshalIndent(&s.db, "", "  ")
	if err != nil {
		return err
	}
	tmp := s.path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, s.path)
}

// --- Sessions ---

func (s *jsonStore) UpsertSession(sess *Session) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	cp := *sess
	s.db.Sessions[cp.ID] = &cp
	return s.persist()
}

func (s *jsonStore) GetSession(id string) (*Session, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	sess, ok := s.db.Sessions[id]
	if !ok {
		return nil, nil
	}
	cp := *sess
	return &cp, nil
}

// --- Voice requests ---

func (s *jsonStore) CreateRequest(r *VoiceRequest) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	cp := *r
	s.db.Requests[cp.ID] = &cp
	return s.persist()
}

func (s *jsonStore) GetRequest(id string) (*VoiceRequest, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	r, ok := s.db.Requests[id]
	if !ok {
		return nil, nil
	}
	cp := *r
	return &cp, nil
}

func (s *jsonStore) UpdateRequest(r *VoiceRequest) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	cp := *r
	s.db.Requests[cp.ID] = &cp
	return s.persist()
}

func (s *jsonStore) WaitingRequests(chatID int64) ([]*VoiceRequest, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var out []*VoiceRequest
	for _, r := range s.db.Requests {
		if r.Status != RequestWaiting {
			continue
		}
		if chatID != 0 {
			sess := s.db.Sessions[r.SessionID]
			if sess == nil || sess.TelegramChatID != chatID {
				continue
			}
		}
		cp := *r
		out = append(out, &cp)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].CreatedAt.Before(out[j].CreatedAt) })
	return out, nil
}

// --- Voice messages ---

func (s *jsonStore) AddVoiceMessage(m *VoiceMessage) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	cp := *m
	s.db.VoiceMessages[cp.ID] = &cp
	return s.persist()
}

func (s *jsonStore) GetVoiceMessage(id string) (*VoiceMessage, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	m, ok := s.db.VoiceMessages[id]
	if !ok {
		return nil, nil
	}
	cp := *m
	return &cp, nil
}

func (s *jsonStore) UpdateVoiceMessage(m *VoiceMessage) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	cp := *m
	s.db.VoiceMessages[cp.ID] = &cp
	return s.persist()
}

func (s *jsonStore) OldestPendingVoice(chatID int64) (*VoiceMessage, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var best *VoiceMessage
	for _, m := range s.db.VoiceMessages {
		if m.Status != VoicePending {
			continue
		}
		if chatID != 0 && m.TelegramChatID != chatID {
			continue
		}
		if best == nil || m.CreatedAt.Before(best.CreatedAt) {
			best = m
		}
	}
	if best == nil {
		return nil, nil
	}
	cp := *best
	return &cp, nil
}

// --- Spoken updates ---

func (s *jsonStore) AddSpokenUpdate(u *SpokenUpdate) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	cp := *u
	s.db.SpokenUpdates[cp.ID] = &cp
	return s.persist()
}

// RecentSpokenUpdates returns up to n most recent updates for a session
// (sessionID == "" means all), oldest first so the newest is last.
func (s *jsonStore) RecentSpokenUpdates(sessionID string, n int) ([]*SpokenUpdate, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var all []*SpokenUpdate
	for _, u := range s.db.SpokenUpdates {
		if sessionID != "" && u.SessionID != sessionID {
			continue
		}
		cp := *u
		all = append(all, &cp)
	}
	sort.Slice(all, func(i, j int) bool { return all[i].CreatedAt.Before(all[j].CreatedAt) })
	if n > 0 && len(all) > n {
		all = all[len(all)-n:]
	}
	return all, nil
}

func (s *jsonStore) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.persist()
}
