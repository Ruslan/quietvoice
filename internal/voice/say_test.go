package voice_test

import (
	"bytes"
	"context"
	"encoding/binary"
	"os"
	"os/exec"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"quietvoice/internal/inference"
	"quietvoice/internal/voice"
)

// minimalWav builds a valid PCM16 mono 24 kHz WAV carrying the given PCM bytes.
func minimalWav(pcm []byte) []byte {
	var buf bytes.Buffer
	dataLen := uint32(len(pcm))
	var (
		numChannels   uint16 = 1
		sampleRate    uint32 = 24000
		bitsPerSample uint16 = 16
	)
	byteRate := sampleRate * uint32(numChannels) * uint32(bitsPerSample) / 8
	blockAlign := numChannels * bitsPerSample / 8
	buf.WriteString("RIFF")
	_ = binary.Write(&buf, binary.LittleEndian, uint32(36+dataLen))
	buf.WriteString("WAVE")
	buf.WriteString("fmt ")
	_ = binary.Write(&buf, binary.LittleEndian, uint32(16))
	_ = binary.Write(&buf, binary.LittleEndian, uint16(1))
	_ = binary.Write(&buf, binary.LittleEndian, numChannels)
	_ = binary.Write(&buf, binary.LittleEndian, sampleRate)
	_ = binary.Write(&buf, binary.LittleEndian, byteRate)
	_ = binary.Write(&buf, binary.LittleEndian, blockAlign)
	_ = binary.Write(&buf, binary.LittleEndian, bitsPerSample)
	buf.WriteString("data")
	_ = binary.Write(&buf, binary.LittleEndian, dataLen)
	buf.Write(pcm)
	return buf.Bytes()
}

// wavFakeEngine writes a real tiny WAV per Synthesize call, counts calls, and
// tracks peak concurrency so a test can assert the fan-out actually overlaps.
type wavFakeEngine struct {
	dir      string
	calls    int64
	inFlight int64
	maxInFly int64

	mu    sync.Mutex
	texts []string
}

func (e *wavFakeEngine) Synthesize(ctx context.Context, req inference.SynthesizeRequest) (*inference.SynthesizeResult, error) {
	atomic.AddInt64(&e.calls, 1)
	cur := atomic.AddInt64(&e.inFlight, 1)
	for {
		m := atomic.LoadInt64(&e.maxInFly)
		if cur <= m || atomic.CompareAndSwapInt64(&e.maxInFly, m, cur) {
			break
		}
	}
	time.Sleep(20 * time.Millisecond) // widen the concurrency window
	atomic.AddInt64(&e.inFlight, -1)

	e.mu.Lock()
	e.texts = append(e.texts, req.Text)
	e.mu.Unlock()

	f, err := os.CreateTemp(e.dir, "chunk_*.wav")
	if err != nil {
		return nil, err
	}
	// A couple of PCM samples; content is irrelevant to the wiring test.
	if _, err := f.Write(minimalWav([]byte{1, 2, 3, 4})); err != nil {
		f.Close()
		return nil, err
	}
	f.Close()
	return &inference.SynthesizeResult{WavPath: f.Name()}, nil
}

func (e *wavFakeEngine) Interpret(ctx context.Context, a inference.AudioInput, r inference.InterpretRequest) (*inference.InterpretResult, error) {
	return &inference.InterpretResult{Intent: "unused"}, nil
}
func (e *wavFakeEngine) Health(ctx context.Context) error { return nil }

// oggCapturingTransport records SendVoice invocations and the delivered ogg's
// size (read while the file still exists, before Say's deferred cleanup).
type oggCapturingTransport struct {
	mu       sync.Mutex
	voices   int
	oggBytes int
	caption  string
}

func (t *oggCapturingTransport) SendVoice(ctx context.Context, chatID int64, ogg, caption string) (int64, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.voices++
	t.caption = caption
	if b, err := os.ReadFile(ogg); err == nil {
		t.oggBytes = len(b)
	}
	return int64(t.voices), nil
}
func (t *oggCapturingTransport) SendMessage(ctx context.Context, chatID int64, text string) error {
	return nil
}

func TestSayChunksConcurrentlySingleVoice(t *testing.T) {
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		t.Skip("ffmpeg not available; Say end-to-end needs wav->ogg transcode")
	}

	eng := &wavFakeEngine{dir: t.TempDir()}
	tp := &oggCapturingTransport{}
	st := newStore(t)
	svc := voice.New(voice.Config{DefaultChatID: 42, WorkDir: t.TempDir()}, st, eng, tp)

	// Six sentences → several chunks (target ~220 chars). Each sentence is padded
	// so packing yields > 1 chunk and the fan-out has something to parallelize.
	pad := strings.Repeat("more words here ", 4)
	var sb strings.Builder
	for i := 0; i < 6; i++ {
		sb.WriteString("Sentence number ")
		sb.WriteString(pad)
		sb.WriteString("end. ")
	}
	text := sb.String()

	status, err := svc.Say(context.Background(), "sess", text)
	if err != nil {
		t.Fatalf("Say: %v", err)
	}
	if status == "" {
		t.Fatal("empty status")
	}

	chunks := len(eng.texts)
	if chunks < 2 {
		t.Fatalf("expected multi-chunk synthesis, got %d Synthesize calls", chunks)
	}
	if got := atomic.LoadInt64(&eng.calls); int(got) != chunks {
		t.Fatalf("call count %d != recorded texts %d", got, chunks)
	}
	if max := atomic.LoadInt64(&eng.maxInFly); max < 2 {
		t.Fatalf("expected concurrent synthesis (maxInFlight>=2), got %d", max)
	}
	tp.mu.Lock()
	voices, oggBytes, caption := tp.voices, tp.oggBytes, tp.caption
	tp.mu.Unlock()
	if voices != 1 {
		t.Fatalf("expected exactly ONE SendVoice, got %d", voices)
	}
	if oggBytes == 0 {
		t.Fatal("delivered ogg was empty or unreadable")
	}
	if caption != text {
		t.Fatalf("caption should be the full text")
	}

	// The chunks that were synthesized must together reconstruct the full text
	// (order-agnostic here; order preservation is covered by ConcatWavs tests).
	joined := strings.Join(eng.texts, " ")
	for _, w := range []string{"Sentence", "end."} {
		if !strings.Contains(joined, w) {
			t.Fatalf("synthesized chunks missing %q", w)
		}
	}

	// The spoken update recorded for context must carry the full original text.
	ups, _ := st.RecentSpokenUpdates("sess", 5)
	if len(ups) != 1 || ups[0].Text != text {
		t.Fatalf("spoken update not recorded with full text: %+v", ups)
	}
}

func TestSaySingleChunkPath(t *testing.T) {
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		t.Skip("ffmpeg not available")
	}
	eng := &wavFakeEngine{dir: t.TempDir()}
	tp := &oggCapturingTransport{}
	st := newStore(t)
	svc := voice.New(voice.Config{DefaultChatID: 42, WorkDir: t.TempDir()}, st, eng, tp)

	if _, err := svc.Say(context.Background(), "sess", "Short and sweet."); err != nil {
		t.Fatalf("Say: %v", err)
	}
	if got := atomic.LoadInt64(&eng.calls); got != 1 {
		t.Fatalf("single chunk should be one Synthesize call, got %d", got)
	}
	tp.mu.Lock()
	voices := tp.voices
	tp.mu.Unlock()
	if voices != 1 {
		t.Fatalf("expected one SendVoice, got %d", voices)
	}
}

// failingEngine always returns empty audio, exercising the retry-then-fail path.
type failingEngine struct{ calls int64 }

func (e *failingEngine) Synthesize(ctx context.Context, req inference.SynthesizeRequest) (*inference.SynthesizeResult, error) {
	atomic.AddInt64(&e.calls, 1)
	return &inference.SynthesizeResult{WavPath: ""}, nil // empty audio
}
func (e *failingEngine) Interpret(ctx context.Context, a inference.AudioInput, r inference.InterpretRequest) (*inference.InterpretResult, error) {
	return &inference.InterpretResult{}, nil
}
func (e *failingEngine) Health(ctx context.Context) error { return nil }

func TestSayRetriesThenFails(t *testing.T) {
	eng := &failingEngine{}
	tp := &oggCapturingTransport{}
	st := newStore(t)
	svc := voice.New(voice.Config{DefaultChatID: 42, WorkDir: t.TempDir()}, st, eng, tp)

	if _, err := svc.Say(context.Background(), "sess", "Just one sentence."); err == nil {
		t.Fatal("expected Say to fail when synthesis returns empty audio")
	}
	if got := atomic.LoadInt64(&eng.calls); got != 3 {
		t.Fatalf("expected 3 attempts (retry budget), got %d", got)
	}
	tp.mu.Lock()
	voices := tp.voices
	tp.mu.Unlock()
	if voices != 0 {
		t.Fatalf("no voice should be delivered on failure, got %d", voices)
	}
}
