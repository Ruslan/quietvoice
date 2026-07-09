package voice_test

import (
	"context"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"quietvoice/internal/inference"
	"quietvoice/internal/store"
	"quietvoice/internal/voice"
)

type fakeEngine struct{ intent string }

func (e fakeEngine) Synthesize(ctx context.Context, req inference.SynthesizeRequest) (*inference.SynthesizeResult, error) {
	return &inference.SynthesizeResult{WavPath: "unused.wav"}, nil
}
func (e fakeEngine) Interpret(ctx context.Context, a inference.AudioInput, r inference.InterpretRequest) (*inference.InterpretResult, error) {
	return &inference.InterpretResult{Type: "intent", Intent: e.intent}, nil
}
func (e fakeEngine) Health(ctx context.Context) error { return nil }

type fakeTransport struct {
	mu          sync.Mutex
	notices     []string
	voices      int
	lastMsgChat int64
}

func (f *fakeTransport) SendVoice(ctx context.Context, chatID int64, ogg, caption string) (int64, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.voices++
	return int64(f.voices), nil
}
func (f *fakeTransport) SendMessage(ctx context.Context, chatID int64, text string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.notices = append(f.notices, text)
	f.lastMsgChat = chatID
	return nil
}
func (f *fakeTransport) noticeCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.notices)
}
func (f *fakeTransport) hasNotice(substr string) bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	for _, n := range f.notices {
		if strings.Contains(n, substr) {
			return true
		}
	}
	return false
}

func newStore(t *testing.T) store.Store {
	t.Helper()
	st, err := store.Open(filepath.Join(t.TempDir(), "state.json"))
	if err != nil {
		t.Fatalf("open store: %v", err)
	}
	return st
}

func TestListenVoiceConsumesPending(t *testing.T) {
	st := newStore(t)
	svc := voice.New(voice.Config{DefaultChatID: 42, WaitTimeout: time.Second}, st, fakeEngine{intent: "DO THE THING"}, &fakeTransport{})

	if err := st.AddVoiceMessage(&store.VoiceMessage{
		ID: "v1", TelegramChatID: 42, Status: store.VoicePending,
		LocalPath: "/tmp/x.ogg", CreatedAt: time.Now(),
	}); err != nil {
		t.Fatal(err)
	}

	tp := &fakeTransport{}
	svc = voice.New(voice.Config{DefaultChatID: 42, WaitTimeout: time.Second}, st, fakeEngine{intent: "DO THE THING"}, tp)

	got, fromPending, err := svc.ListenVoice(context.Background(), "sess", "question?")
	if err != nil {
		t.Fatalf("ListenVoice: %v", err)
	}
	if got != "DO THE THING" {
		t.Fatalf("intent = %q, want %q", got, "DO THE THING")
	}
	if !fromPending {
		t.Fatal("fromPending = false, want true for a consumed pending voice")
	}
	vm, _ := st.GetVoiceMessage("v1")
	if vm.Status != store.VoiceConsumed {
		t.Fatalf("pending status = %q, want consumed", vm.Status)
	}
	// Feature 2: the user is told what was recognized and that it reached the agent.
	if !tp.hasNotice("Recognized (saved voice): DO THE THING") {
		t.Fatalf("missing recognition feedback; notices=%v", tp.notices)
	}
}

func TestListenVoiceWaitsAndRoutesIncoming(t *testing.T) {
	st := newStore(t)
	tp := &fakeTransport{}
	svc := voice.New(voice.Config{DefaultChatID: 42, WaitTimeout: 3 * time.Second}, st, fakeEngine{intent: "PATCH NEW CLIENT ONLY"}, tp)

	type res struct {
		text        string
		fromPending bool
		err         error
	}
	done := make(chan res, 1)
	go func() {
		text, fromPending, err := svc.ListenVoice(context.Background(), "sess", "which client?")
		done <- res{text, fromPending, err}
	}()

	// Wait until the waiting request has been registered.
	deadline := time.Now().Add(time.Second)
	for {
		if reqs, _ := st.WaitingRequests(42); len(reqs) == 1 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("waiting request was not created")
		}
		time.Sleep(5 * time.Millisecond)
	}
	if tp.noticeCount() == 0 {
		t.Fatal("expected a Telegram waiting notice")
	}

	svc.HandleIncomingVoice(context.Background(), voice.IncomingVoice{
		ChatID: 42, MessageID: 5, FileID: "f1", LocalPath: "/tmp/y.ogg",
	})

	select {
	case r := <-done:
		if r.err != nil {
			t.Fatalf("ListenVoice: %v", r.err)
		}
		if r.text != "PATCH NEW CLIENT ONLY" {
			t.Fatalf("intent = %q", r.text)
		}
		if r.fromPending {
			t.Fatal("fromPending = true, want false for a live-wait voice")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("ListenVoice did not return after incoming voice")
	}

	// Feature 2: recognition feedback fires for the live-wait path too.
	if !tp.hasNotice("Recognized: PATCH NEW CLIENT ONLY") {
		t.Fatalf("missing recognition feedback; notices=%v", tp.notices)
	}

	if reqs, _ := st.WaitingRequests(42); len(reqs) != 0 {
		t.Fatalf("request still waiting after completion: %d", len(reqs))
	}
}

// Regression: a session persisted under an earlier, wrong chat id must not
// override the configured default chat. Config is the routing source of truth.
func TestConfigChatOverridesStalePersistedSession(t *testing.T) {
	st := newStore(t)
	// Simulate state left by an earlier run bound to the wrong chat (id 1).
	if err := st.UpsertSession(&store.Session{
		ID: "default", TelegramChatID: 1, CreatedAt: time.Now(), LastSeenAt: time.Now(),
	}); err != nil {
		t.Fatal(err)
	}
	tp := &fakeTransport{}
	svc := voice.New(voice.Config{DefaultChatID: 42, WaitTimeout: 500 * time.Millisecond}, st, fakeEngine{intent: "OK"}, tp)

	// A pending voice exists on the CORRECT (configured) chat 42.
	if err := st.AddVoiceMessage(&store.VoiceMessage{
		ID: "v1", TelegramChatID: 42, Status: store.VoicePending,
		LocalPath: "/tmp/x.ogg", CreatedAt: time.Now(),
	}); err != nil {
		t.Fatal(err)
	}

	// With the bug, ensureSession keeps chat=1, OldestPendingVoice(1) finds
	// nothing, and ListenVoice blocks then times out. With the fix it consumes.
	got, _, err := svc.ListenVoice(context.Background(), "", "q?")
	if err != nil {
		t.Fatalf("ListenVoice: %v (stale chat likely not overridden)", err)
	}
	if got != "OK" {
		t.Fatalf("intent = %q, want OK", got)
	}
	sess, _ := st.GetSession("default")
	if sess.TelegramChatID != 42 {
		t.Fatalf("session chat = %d, want 42 (config must win over stale state)", sess.TelegramChatID)
	}
}

func TestHandleIncomingVoiceKeepsPendingWhenNoWaiter(t *testing.T) {
	st := newStore(t)
	tp := &fakeTransport{}
	svc := voice.New(voice.Config{DefaultChatID: 42}, st, fakeEngine{intent: "x"}, tp)

	svc.HandleIncomingVoice(context.Background(), voice.IncomingVoice{
		ChatID: 42, MessageID: 7, FileID: "f2", LocalPath: "/tmp/z.ogg",
	})

	pending, _ := st.OldestPendingVoice(42)
	if pending == nil {
		t.Fatal("expected a pending voice to be stored")
	}
	if tp.noticeCount() != 1 {
		t.Fatalf("expected one 'saved as pending' notice, got %d", tp.noticeCount())
	}
}
