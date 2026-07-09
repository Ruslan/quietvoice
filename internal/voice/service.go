// Package voice is the QuietVoice control-plane orchestration: it turns the two
// MCP tools (say, listen_voice) into transport + inference + state actions. It
// owns the pending-voice queue, session→chat routing, and the previous-say
// context that seeds audio interpretation. It depends only on interfaces
// (Transport, inference.Engine, store.Store), never on Telegram directly.
package voice

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sync"
	"time"

	"quietvoice/internal/audio"
	"quietvoice/internal/inference"
	"quietvoice/internal/store"
)

// Transport delivers spoken audio and text notices to the user. Satisfied by
// *telegram.Bot.
type Transport interface {
	SendVoice(ctx context.Context, chatID int64, oggPath, caption string) (int64, error)
	SendMessage(ctx context.Context, chatID int64, text string) error
}

// IncomingVoice is a downloaded user voice note handed to the service by the
// transport's poll loop (transport-agnostic mirror of telegram.IncomingVoice).
type IncomingVoice struct {
	ChatID    int64
	UserID    int64
	MessageID int64
	FileID    string
	LocalPath string
	Duration  int
}

// Config tunes the service.
type Config struct {
	DefaultChatID   int64
	WaitTimeout     time.Duration        // how long listen_voice blocks (default 10m)
	SayContextCount int                  // previous say updates fed to interpret (default 2)
	WorkDir         string               // scratch dir for transcoded audio
	ListenMode      inference.ListenMode // how audio is turned into text (default assisted)
	EvalLogPath     string               // JSONL log of interpretations for offline re-eval
}

// Service ties transport, inference and storage together.
type Service struct {
	cfg    Config
	store  store.Store
	engine inference.Engine
	tp     Transport

	mu      sync.Mutex
	waiters map[string]chan store.VoiceMessage // requestID -> delivery channel

	evalMu sync.Mutex // serializes appends to the eval log
}

// New builds a Service, applying config defaults.
func New(cfg Config, st store.Store, engine inference.Engine, tp Transport) *Service {
	if cfg.WaitTimeout <= 0 {
		cfg.WaitTimeout = 10 * time.Minute
	}
	if cfg.SayContextCount <= 0 {
		cfg.SayContextCount = 2
	}
	if cfg.WorkDir == "" {
		cfg.WorkDir = filepath.Join(os.TempDir(), "quietvoice-work")
	}
	if cfg.ListenMode == "" {
		cfg.ListenMode = inference.ModeAssisted
	}
	return &Service{
		cfg:     cfg,
		store:   st,
		engine:  engine,
		tp:      tp,
		waiters: map[string]chan store.VoiceMessage{},
	}
}

// sayMaxConcurrency bounds how many chunks we synthesize at once; the inferenced
// pool runs a handful of hot replicas, so a small fan-out is the useful range.
const sayMaxConcurrency = 4

// sayMaxAttempts is the per-chunk retry budget: remote synthesis intermittently
// returns empty audio, and a silent gap in the delivered speech is worse than a
// retry. A chunk that still fails after this many tries fails the whole Say.
const sayMaxAttempts = 3

// Say synthesizes text, delivers it as a Telegram voice note, and records it as
// a spoken update for later context. Returns a short human-readable status.
//
// Long text is split into sentence-ish chunks (the TTS backend caps generation
// at a few hundred characters before returning empty audio) that are synthesized
// concurrently across the replica pool, then concatenated into a single WAV so
// the user still receives exactly ONE voice note per say.
func (s *Service) Say(ctx context.Context, sessionID, text string) (string, error) {
	sess := s.ensureSession(sessionID)

	if err := os.MkdirAll(s.cfg.WorkDir, 0o755); err != nil {
		return "", err
	}

	// Provision the reference voice on the inference backend if needed (remote
	// mode). The voice lives in the control-plane config, so a node deployed
	// without a voice still speaks with it, and switching nodes re-registers it.
	if reg, ok := s.engine.(inference.VoiceRegistrar); ok {
		if err := reg.EnsureVoice(ctx); err != nil {
			return "", fmt.Errorf("say: ensure voice: %w", err)
		}
	}

	chunks := splitText(text)
	if len(chunks) == 0 {
		// Nothing splittable (empty/whitespace text): fall back to one chunk so
		// behavior matches the pre-chunking path.
		chunks = []string{text}
	}

	wavs, cleanup, err := s.synthesizeChunks(ctx, chunks)
	defer cleanup()
	if err != nil {
		return "", fmt.Errorf("say: %w", err)
	}

	// One chunk keeps today's simple path (no concat overhead).
	wavPath := wavs[0]
	if len(wavs) > 1 {
		joined := filepath.Join(s.cfg.WorkDir, fmt.Sprintf("say_%d.wav", time.Now().UnixNano()))
		if err := audio.ConcatWavs(wavs, joined); err != nil {
			return "", fmt.Errorf("say: concat: %w", err)
		}
		defer os.Remove(joined)
		wavPath = joined
	}

	ogg := filepath.Join(s.cfg.WorkDir, fmt.Sprintf("say_%d.ogg", time.Now().UnixNano()))
	if err := audio.WavToOggOpus(ctx, wavPath, ogg); err != nil {
		return "", fmt.Errorf("say: transcode: %w", err)
	}
	defer os.Remove(ogg)

	msgID, err := s.tp.SendVoice(ctx, sess.TelegramChatID, ogg, text)
	if err != nil {
		return "", fmt.Errorf("say: deliver: %w", err)
	}

	_ = s.store.AddSpokenUpdate(&store.SpokenUpdate{
		ID:                store.NewID("say"),
		SessionID:         sess.ID,
		Text:              text,
		TelegramMessageID: msgID,
		CreatedAt:         time.Now(),
	})
	return "Spoken update delivered to the user via Telegram voice message.", nil
}

// synthesizeChunks synthesizes chunks concurrently (bounded) and returns their
// WAV paths in the SAME order as chunks, so concatenation preserves speech
// order regardless of completion order. Each chunk gets up to sayMaxAttempts
// tries; if any chunk ultimately fails, the first error is returned. The
// returned cleanup removes every produced WAV and must always be deferred by the
// caller, including on error.
func (s *Service) synthesizeChunks(ctx context.Context, chunks []string) ([]string, func(), error) {
	wavs := make([]string, len(chunks))
	cleanup := func() {
		for _, w := range wavs {
			if w != "" {
				_ = os.Remove(w)
			}
		}
	}

	concurrency := len(chunks)
	if concurrency > sayMaxConcurrency {
		concurrency = sayMaxConcurrency
	}

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	sem := make(chan struct{}, concurrency)

	var wg sync.WaitGroup
	var errMu sync.Mutex
	var firstErr error
	fail := func(err error) {
		errMu.Lock()
		if firstErr == nil {
			firstErr = err
		}
		errMu.Unlock()
		cancel() // stop the remaining chunks; the whole say is lost anyway.
	}

	for i, chunk := range chunks {
		wg.Add(1)
		go func(i int, chunk string) {
			defer wg.Done()
			select {
			case sem <- struct{}{}:
			case <-ctx.Done():
				return
			}
			defer func() { <-sem }()

			var lastErr error
			for attempt := 1; attempt <= sayMaxAttempts; attempt++ {
				if ctx.Err() != nil {
					return
				}
				res, err := s.engine.Synthesize(ctx, inference.SynthesizeRequest{Text: chunk})
				switch {
				case err != nil:
					lastErr = err
				case res == nil || res.WavPath == "":
					lastErr = fmt.Errorf("backend returned empty audio")
				default:
					wavs[i] = res.WavPath // distinct index per goroutine: race-free.
					return
				}
			}
			fail(fmt.Errorf("chunk %d/%d failed after %d attempts: %w", i+1, len(chunks), sayMaxAttempts, lastErr))
		}(i, chunk)
	}
	wg.Wait()

	return wavs, cleanup, firstErr
}

// ListenVoice returns the user's next spoken intent. It first consumes a pending
// voice if one exists; otherwise it registers a waiting request, notifies the
// user, and blocks until a voice arrives or the timeout elapses.
//
// fromPending reports whether the returned intent came from a voice the user had
// recorded BEFORE the agent asked (the pending branch), so callers can tell the
// input was pre-buffered rather than captured live.
func (s *Service) ListenVoice(ctx context.Context, sessionID, promptText string) (intent string, fromPending bool, err error) {
	sess := s.ensureSession(sessionID)

	// 1. Consume a pending voice recorded before the agent asked.
	if pending, _ := s.store.OldestPendingVoice(sess.TelegramChatID); pending != nil {
		log.Printf("listen_voice: consuming pending voice %s", pending.ID)
		intent, err = s.processVoice(ctx, sess, promptText, pending, true)
		return intent, true, err
	}

	// 2. No pending voice: create a waiting request and block for one.
	req := &store.VoiceRequest{
		ID:         store.NewID("req"),
		SessionID:  sess.ID,
		PromptText: promptText,
		Status:     store.RequestWaiting,
		CreatedAt:  time.Now(),
		ExpiresAt:  time.Now().Add(s.cfg.WaitTimeout),
	}
	if err := s.store.CreateRequest(req); err != nil {
		return "", false, fmt.Errorf("listen_voice: create request: %w", err)
	}

	ch := make(chan store.VoiceMessage, 1)
	s.mu.Lock()
	s.waiters[req.ID] = ch
	s.mu.Unlock()
	defer func() {
		s.mu.Lock()
		delete(s.waiters, req.ID)
		s.mu.Unlock()
	}()

	s.notifyWaiting(ctx, sess.TelegramChatID, promptText)

	select {
	case vm := <-ch:
		intent, err = s.processVoice(ctx, sess, promptText, &vm, false)
		return intent, false, err
	case <-time.After(s.cfg.WaitTimeout):
		s.expireRequest(req)
		return "", false, fmt.Errorf("no voice reply received before timeout")
	case <-ctx.Done():
		s.expireRequest(req)
		return "", false, fmt.Errorf("listen_voice cancelled: %w", ctx.Err())
	}
}

// HandleIncomingVoice routes a downloaded user voice note: bind it to a single
// waiting request when there is exactly one, otherwise store it as pending.
func (s *Service) HandleIncomingVoice(ctx context.Context, in IncomingVoice) {
	vm := &store.VoiceMessage{
		ID:                store.NewID("voice"),
		TelegramChatID:    in.ChatID,
		TelegramMessageID: in.MessageID,
		TelegramFileID:    in.FileID,
		LocalPath:         in.LocalPath,
		Status:            store.VoicePending,
		CreatedAt:         time.Now(),
	}
	if err := s.store.AddVoiceMessage(vm); err != nil {
		log.Printf("incoming voice: store: %v", err)
		return
	}

	waiting, _ := s.store.WaitingRequests(in.ChatID)
	if len(waiting) == 0 {
		// Rule 2: no waiting request → keep as pending.
		_ = s.tp.SendMessage(ctx, in.ChatID, "🎙 Voice saved as pending. It will be used the next time your agent listens.")
		return
	}
	// Rule 1 (and MVP Rule 3): bind to the oldest waiting request.
	req := waiting[0]
	if len(waiting) > 1 {
		log.Printf("incoming voice: %d waiting requests; binding to oldest %s", len(waiting), req.ID)
	}

	now := time.Now()
	vm.Status = store.VoiceClaimed
	vm.SessionID = req.SessionID
	vm.RequestID = req.ID
	vm.ClaimedAt = &now
	_ = s.store.UpdateVoiceMessage(vm)

	s.mu.Lock()
	ch := s.waiters[req.ID]
	s.mu.Unlock()
	if ch == nil {
		// The waiter is gone (e.g. restart). Leave it claimed; nothing to deliver.
		log.Printf("incoming voice: no live waiter for request %s", req.ID)
		return
	}
	select {
	case ch <- *vm:
	default:
	}
}

// processVoice interprets a claimed/pending voice and returns agent-ready text.
func (s *Service) processVoice(ctx context.Context, sess *store.Session, promptText string, vm *store.VoiceMessage, fromPending bool) (string, error) {
	now := time.Now()
	vm.Status = store.VoiceProcessing
	vm.SessionID = sess.ID
	if vm.ClaimedAt == nil {
		vm.ClaimedAt = &now
	}
	_ = s.store.UpdateVoiceMessage(vm)

	prev := s.previousSay(sess.ID)
	result, err := s.engine.Interpret(ctx, inference.AudioInput{Path: vm.LocalPath}, inference.InterpretRequest{
		Mode:          s.cfg.ListenMode,
		AgentQuestion: promptText,
		PreviousSay:   prev,
	})
	if err != nil {
		vm.Status = store.VoiceFailed
		_ = s.store.UpdateVoiceMessage(vm)
		if vm.RequestID != "" {
			s.completeRequest(vm.RequestID, store.RequestCompleted)
		}
		return "", fmt.Errorf("interpret voice: %w", err)
	}

	consumed := time.Now()
	vm.Status = store.VoiceConsumed
	vm.ConsumedAt = &consumed
	_ = s.store.UpdateVoiceMessage(vm)
	if vm.RequestID != "" {
		s.completeRequest(vm.RequestID, store.RequestCompleted)
	}
	s.logEval(vm, promptText, result)

	// Confirm back to the user what we recognized and that it reached the agent.
	// When it was a buffered (pending) voice, say so in the SAME message.
	// Best-effort: a failed notice must never lose the recognition result.
	label := "Recognized"
	if fromPending {
		label = "Recognized (saved voice)"
	}
	feedback := fmt.Sprintf("%s: %s\n↳ sent to your agent", label, result.Intent)
	if err := s.tp.SendMessage(ctx, sess.TelegramChatID, feedback); err != nil {
		log.Printf("listen_voice: recognition feedback: %v", err)
	}

	return result.Intent, nil
}

// evalRecord is one appended line in the re-evaluation corpus: it captures the
// audio, every recognizer's transcript, and the final intent so models can be
// compared or the LLM step re-run later WITHOUT re-running ASR/inference.
type evalRecord struct {
	Time        string                    `json:"time"`
	VoiceID     string                    `json:"voice_id"`
	SessionID   string                    `json:"session_id,omitempty"`
	Audio       string                    `json:"audio"`
	Mode        string                    `json:"mode"`
	Prompt      string                    `json:"prompt,omitempty"`
	Transcripts []inference.TranscriptRef `json:"transcripts,omitempty"`
	Intent      string                    `json:"intent"`
}

// logEval appends one JSONL record for offline re-evaluation. Best-effort.
func (s *Service) logEval(vm *store.VoiceMessage, prompt string, result *inference.InterpretResult) {
	if s.cfg.EvalLogPath == "" {
		return
	}
	rec := evalRecord{
		Time:        time.Now().Format(time.RFC3339),
		VoiceID:     vm.ID,
		SessionID:   vm.SessionID,
		Audio:       vm.LocalPath,
		Mode:        string(s.cfg.ListenMode),
		Prompt:      prompt,
		Transcripts: result.Transcripts,
		Intent:      result.Intent,
	}
	line, err := json.Marshal(rec)
	if err != nil {
		return
	}
	s.evalMu.Lock()
	defer s.evalMu.Unlock()
	if dir := filepath.Dir(s.cfg.EvalLogPath); dir != "" {
		_ = os.MkdirAll(dir, 0o755)
	}
	f, err := os.OpenFile(s.cfg.EvalLogPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		log.Printf("eval log: %v", err)
		return
	}
	defer f.Close()
	_, _ = f.Write(append(line, '\n'))
}

func (s *Service) previousSay(sessionID string) []string {
	updates, _ := s.store.RecentSpokenUpdates(sessionID, s.cfg.SayContextCount)
	out := make([]string, 0, len(updates))
	for _, u := range updates {
		out = append(out, u.Text)
	}
	return out
}

func (s *Service) notifyWaiting(ctx context.Context, chatID int64, promptText string) {
	msg := "🎙 Your coding agent is waiting for a voice reply.\n\nSend a voice message."
	if promptText != "" {
		msg = fmt.Sprintf("🎙 Your coding agent is waiting for a voice reply.\n\nQuestion:\n%s\n\nSend a voice message.", promptText)
	}
	if err := s.tp.SendMessage(ctx, chatID, msg); err != nil {
		log.Printf("listen_voice: notify: %v", err)
	}
}

func (s *Service) expireRequest(req *store.VoiceRequest) {
	fresh, _ := s.store.GetRequest(req.ID)
	if fresh == nil || fresh.Status != store.RequestWaiting {
		return
	}
	fresh.Status = store.RequestExpired
	_ = s.store.UpdateRequest(fresh)
}

func (s *Service) completeRequest(reqID string, status store.RequestStatus) {
	req, _ := s.store.GetRequest(reqID)
	if req == nil {
		return
	}
	now := time.Now()
	req.Status = status
	req.CompletedAt = &now
	_ = s.store.UpdateRequest(req)
}

// ensureSession returns the session for id, creating it bound to the default
// chat when absent, and refreshing last-seen.
func (s *Service) ensureSession(id string) *store.Session {
	if id == "" {
		id = "default"
	}
	sess, _ := s.store.GetSession(id)
	now := time.Now()
	if sess == nil {
		sess = &store.Session{ID: id, CreatedAt: now}
	}
	sess.LastSeenAt = now
	// The configured default chat is authoritative for routing in the
	// single-chat MVP: a changed QUIET_VOICE_CHAT_ID must take effect even when
	// an older session bound to a different chat is already persisted. Never let
	// stale state silently misroute spoken updates.
	if s.cfg.DefaultChatID != 0 {
		sess.TelegramChatID = s.cfg.DefaultChatID
	}
	_ = s.store.UpsertSession(sess)
	return sess
}
