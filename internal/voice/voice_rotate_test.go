package voice_test

import (
	"context"
	"os"
	"os/exec"
	"sync"
	"testing"

	"quietvoice/internal/inference"
	"quietvoice/internal/voice"
)

// voiceCapturingEngine is a synthesize-only fake that ALSO implements
// inference.VoiceLister (mirroring how the real local/remote engines expose
// their voice registry) and records the SynthesizeRequest.Voice used on every
// call, so rotation tests can assert exactly which voice was picked without a
// real TTS backend. minimalWav (from say_test.go, same package) builds the
// tiny WAV Say's transcode step needs.
type voiceCapturingEngine struct {
	dir  string   // scratch dir for the WAVs Synthesize writes
	pool []string // ListVoices() return value; nil/empty simulates "no pool"

	mu          sync.Mutex
	synthVoices []string
	listCalls   int
}

func (e *voiceCapturingEngine) Synthesize(ctx context.Context, req inference.SynthesizeRequest) (*inference.SynthesizeResult, error) {
	e.mu.Lock()
	e.synthVoices = append(e.synthVoices, req.Voice)
	e.mu.Unlock()

	f, err := os.CreateTemp(e.dir, "chunk_*.wav")
	if err != nil {
		return nil, err
	}
	if _, err := f.Write(minimalWav([]byte{1, 2, 3, 4})); err != nil {
		f.Close()
		return nil, err
	}
	f.Close()
	return &inference.SynthesizeResult{WavPath: f.Name()}, nil
}

func (e *voiceCapturingEngine) Interpret(ctx context.Context, a inference.AudioInput, r inference.InterpretRequest) (*inference.InterpretResult, error) {
	return &inference.InterpretResult{Intent: "unused"}, nil
}

func (e *voiceCapturingEngine) Health(ctx context.Context) error { return nil }

// ListVoices implements inference.VoiceLister.
func (e *voiceCapturingEngine) ListVoices(ctx context.Context) ([]string, error) {
	e.mu.Lock()
	e.listCalls++
	pool := append([]string(nil), e.pool...)
	e.mu.Unlock()
	return pool, nil
}

func (e *voiceCapturingEngine) lastVoice() string {
	e.mu.Lock()
	defer e.mu.Unlock()
	if len(e.synthVoices) == 0 {
		return ""
	}
	return e.synthVoices[len(e.synthVoices)-1]
}

func (e *voiceCapturingEngine) listCallCount() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.listCalls
}

func requireFFmpegForSay(t *testing.T) {
	t.Helper()
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		t.Skip("ffmpeg not available; Say end-to-end needs wav->ogg transcode")
	}
}

// TestVoiceRotateAssignsAndPersistsStickyVoice: the first say() for a new
// session picks a voice from the pool and persists it on the session; a
// second say() for the SAME session must reuse that exact voice rather than
// re-deriving (or re-listing).
func TestVoiceRotateAssignsAndPersistsStickyVoice(t *testing.T) {
	requireFFmpegForSay(t)
	eng := &voiceCapturingEngine{dir: t.TempDir(), pool: []string{"voice01", "voice02", "voice03"}}
	st := newStore(t)
	svc := voice.New(voice.Config{DefaultChatID: 42, WorkDir: t.TempDir(), VoiceRotate: true}, st, eng, &fakeTransport{})

	if _, err := svc.Say(context.Background(), "sessA", "hello there"); err != nil {
		t.Fatalf("Say: %v", err)
	}
	first := eng.lastVoice()
	if first == "" {
		t.Fatal("expected a non-empty voice when rotation is ON and the pool is non-empty")
	}

	sess, err := st.GetSession("sessA")
	if err != nil || sess == nil {
		t.Fatalf("GetSession: %v", err)
	}
	if sess.Voice != first {
		t.Fatalf("session.Voice = %q, want persisted %q", sess.Voice, first)
	}
	if calls := eng.listCallCount(); calls != 1 {
		t.Fatalf("ListVoices called %d times on first say, want 1", calls)
	}

	// Second say for the SAME session must reuse the SAME voice, and must NOT
	// hit ListVoices again (the pool is cached and the session already has a
	// sticky assignment).
	if _, err := svc.Say(context.Background(), "sessA", "second message"); err != nil {
		t.Fatalf("Say: %v", err)
	}
	if got := eng.lastVoice(); got != first {
		t.Fatalf("second say used voice %q, want reused %q", got, first)
	}
	if calls := eng.listCallCount(); calls != 1 {
		t.Fatalf("ListVoices called %d times after second say, want still 1 (cached)", calls)
	}
}

// TestVoiceRotateDifferentSessionsCanGetDifferentVoices: two distinct session
// IDs deterministically hash into the pool and — for these two IDs against
// this pool size — land on DIFFERENT voices, demonstrating rotation actually
// spreads sessions across the pool rather than collapsing them onto one voice.
func TestVoiceRotateDifferentSessionsCanGetDifferentVoices(t *testing.T) {
	requireFFmpegForSay(t)
	pool := []string{"voice01", "voice02", "voice03"}
	eng := &voiceCapturingEngine{dir: t.TempDir(), pool: pool}
	st := newStore(t)
	svc := voice.New(voice.Config{DefaultChatID: 42, WorkDir: t.TempDir(), VoiceRotate: true}, st, eng, &fakeTransport{})

	// fnv32a("sessA") % 3 == 1 ("voice02"); fnv32a("sessB") % 3 == 2 ("voice03") —
	// verified offline so this test is not flaky.
	if _, err := svc.Say(context.Background(), "sessA", "hi"); err != nil {
		t.Fatalf("Say(sessA): %v", err)
	}
	voiceA := eng.lastVoice()

	if _, err := svc.Say(context.Background(), "sessB", "hi"); err != nil {
		t.Fatalf("Say(sessB): %v", err)
	}
	voiceB := eng.lastVoice()

	if voiceA == "" || voiceB == "" {
		t.Fatalf("expected non-empty voices, got %q and %q", voiceA, voiceB)
	}
	if voiceA == voiceB {
		t.Fatalf("expected sessA and sessB to land on different voices, both got %q", voiceA)
	}
	for _, v := range []string{voiceA, voiceB} {
		found := false
		for _, p := range pool {
			if p == v {
				found = true
			}
		}
		if !found {
			t.Fatalf("voice %q not in pool %v", v, pool)
		}
	}
}

// TestVoiceRotateStableAcrossPoolGrowth: a session already bound to a voice
// keeps that EXACT voice even after the pool grows (simulated here by pointing
// a second Service — sharing the same persisted store — at an engine with a
// bigger pool). The persisted binding must win over any re-derivation.
func TestVoiceRotateStableAcrossPoolGrowth(t *testing.T) {
	requireFFmpegForSay(t)
	st := newStore(t)

	smallPool := &voiceCapturingEngine{dir: t.TempDir(), pool: []string{"voice01", "voice02", "voice03"}}
	svc1 := voice.New(voice.Config{DefaultChatID: 42, WorkDir: t.TempDir(), VoiceRotate: true}, st, smallPool, &fakeTransport{})
	if _, err := svc1.Say(context.Background(), "sessA", "hi"); err != nil {
		t.Fatalf("Say (small pool): %v", err)
	}
	original := smallPool.lastVoice()
	if original == "" {
		t.Fatal("expected a voice to be assigned")
	}

	// Simulate the pool changing (a voice was added) by pointing a NEW Service
	// (same persisted store) at an engine reporting a bigger pool. If rotation
	// re-derived instead of honoring the sticky binding, this would very likely
	// pick a different index.
	grownPool := &voiceCapturingEngine{
		dir:  t.TempDir(),
		pool: []string{"voice01", "voice02", "voice03", "voice04", "voice05", "voice06"},
	}
	svc2 := voice.New(voice.Config{DefaultChatID: 42, WorkDir: t.TempDir(), VoiceRotate: true}, st, grownPool, &fakeTransport{})
	if _, err := svc2.Say(context.Background(), "sessA", "hi again"); err != nil {
		t.Fatalf("Say (grown pool): %v", err)
	}
	after := grownPool.lastVoice()
	if after != original {
		t.Fatalf("voice changed after pool grew: got %q, want sticky %q", after, original)
	}
	// The grown-pool engine must never even be asked to list voices — the
	// session already had a persisted assignment.
	if calls := grownPool.listCallCount(); calls != 0 {
		t.Fatalf("ListVoices called %d times on an already-bound session, want 0", calls)
	}
}

// TestVoiceRotateOffUsesConfiguredVoice: with VoiceRotate false, Say must
// behave exactly like today — no voice is chosen (the engine's own default
// applies), ListVoices is never called, and no voice is written to the store.
func TestVoiceRotateOffUsesConfiguredVoice(t *testing.T) {
	requireFFmpegForSay(t)
	eng := &voiceCapturingEngine{dir: t.TempDir(), pool: []string{"voice01", "voice02", "voice03"}}
	st := newStore(t)
	svc := voice.New(voice.Config{DefaultChatID: 42, WorkDir: t.TempDir(), VoiceRotate: false}, st, eng, &fakeTransport{})

	if _, err := svc.Say(context.Background(), "sess", "hi"); err != nil {
		t.Fatalf("Say: %v", err)
	}
	if got := eng.lastVoice(); got != "" {
		t.Fatalf("Voice = %q, want empty (rotation off => engine's own default)", got)
	}
	if calls := eng.listCallCount(); calls != 0 {
		t.Fatalf("ListVoices called %d times with rotation off, want 0", calls)
	}
	sess, err := st.GetSession("sess")
	if err != nil || sess == nil {
		t.Fatalf("GetSession: %v", err)
	}
	if sess.Voice != "" {
		t.Fatalf("session.Voice = %q, want empty (rotation off must not write a voice)", sess.Voice)
	}
}

// TestVoiceRotateEmptyPoolFallsBackToConfiguredVoice: rotation ON but the pool
// is empty/unavailable must never fail say() — it falls back to the engine's
// own configured voice (empty Voice on the request), same as rotation off.
func TestVoiceRotateEmptyPoolFallsBackToConfiguredVoice(t *testing.T) {
	requireFFmpegForSay(t)
	eng := &voiceCapturingEngine{dir: t.TempDir(), pool: nil} // no voices registered on the node
	st := newStore(t)
	svc := voice.New(voice.Config{DefaultChatID: 42, WorkDir: t.TempDir(), VoiceRotate: true}, st, eng, &fakeTransport{})

	if _, err := svc.Say(context.Background(), "sess", "hi"); err != nil {
		t.Fatalf("Say should succeed even when the voice pool is empty: %v", err)
	}
	if got := eng.lastVoice(); got != "" {
		t.Fatalf("Voice = %q, want empty (fallback to configured voice)", got)
	}
	sess, err := st.GetSession("sess")
	if err != nil || sess == nil {
		t.Fatalf("GetSession: %v", err)
	}
	if sess.Voice != "" {
		t.Fatalf("session.Voice = %q, want empty (no assignment could be made)", sess.Voice)
	}
}
