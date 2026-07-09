package audio

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

// makeWav builds a minimal valid PCM WAV in memory with the given format and
// PCM payload.
func makeWav(t *testing.T, numChannels uint16, sampleRate uint32, bits uint16, pcm []byte) []byte {
	t.Helper()
	f := &wavInfo{audioFormat: 1, numChannels: numChannels, sampleRate: sampleRate, bitsPerSample: bits}
	dir := t.TempDir()
	p := filepath.Join(dir, "m.wav")
	if err := writeWav(p, f, pcm); err != nil {
		t.Fatalf("writeWav: %v", err)
	}
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func writeTemp(t *testing.T, b []byte) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "in.wav")
	if err := os.WriteFile(p, b, 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestConcatWavsInOrder(t *testing.T) {
	pcmA := []byte{1, 1, 2, 2}
	pcmB := []byte{3, 3, 4, 4}
	pcmC := []byte{5, 5, 6, 6}
	a := writeTemp(t, makeWav(t, 1, 24000, 16, pcmA))
	b := writeTemp(t, makeWav(t, 1, 24000, 16, pcmB))
	c := writeTemp(t, makeWav(t, 1, 24000, 16, pcmC))

	out := filepath.Join(t.TempDir(), "out.wav")
	if err := ConcatWavs([]string{a, b, c}, out); err != nil {
		t.Fatalf("ConcatWavs: %v", err)
	}

	w, err := parseWav(out)
	if err != nil {
		t.Fatalf("parse out: %v", err)
	}
	want := append(append(append([]byte{}, pcmA...), pcmB...), pcmC...)
	if !bytes.Equal(w.pcm, want) {
		t.Fatalf("concatenated PCM = %v, want %v (order not preserved?)", w.pcm, want)
	}
	if w.numChannels != 1 || w.sampleRate != 24000 || w.bitsPerSample != 16 {
		t.Fatalf("format not preserved: %+v", w)
	}
	// Header must advertise the summed data length.
	raw, _ := os.ReadFile(out)
	if got := binary.LittleEndian.Uint32(raw[4:8]); got != uint32(36+len(want)) {
		t.Fatalf("RIFF size = %d, want %d", got, 36+len(want))
	}
}

func TestConcatWavsFormatMismatch(t *testing.T) {
	a := writeTemp(t, makeWav(t, 1, 24000, 16, []byte{1, 2, 3, 4}))
	b := writeTemp(t, makeWav(t, 1, 16000, 16, []byte{5, 6, 7, 8})) // different rate
	out := filepath.Join(t.TempDir(), "out.wav")
	if err := ConcatWavs([]string{a, b}, out); err == nil {
		t.Fatal("expected format-mismatch error, got nil")
	}
}

func TestConcatWavsChannelMismatch(t *testing.T) {
	a := writeTemp(t, makeWav(t, 1, 24000, 16, []byte{1, 2, 3, 4}))
	b := writeTemp(t, makeWav(t, 2, 24000, 16, []byte{5, 6, 7, 8})) // stereo
	out := filepath.Join(t.TempDir(), "out.wav")
	if err := ConcatWavs([]string{a, b}, out); err == nil {
		t.Fatal("expected channel-mismatch error, got nil")
	}
}

func TestConcatWavsMalformed(t *testing.T) {
	bad := writeTemp(t, []byte("NOTAWAVFILE!!"))
	out := filepath.Join(t.TempDir(), "out.wav")
	if err := ConcatWavs([]string{bad}, out); err == nil {
		t.Fatal("expected error for non-RIFF file, got nil")
	}
}

func TestConcatWavsEmptyInput(t *testing.T) {
	out := filepath.Join(t.TempDir(), "out.wav")
	if err := ConcatWavs(nil, out); err == nil {
		t.Fatal("expected error for empty input, got nil")
	}
}

func TestConcatWavsSingle(t *testing.T) {
	pcm := []byte{9, 9, 8, 8}
	a := writeTemp(t, makeWav(t, 1, 24000, 16, pcm))
	out := filepath.Join(t.TempDir(), "out.wav")
	if err := ConcatWavs([]string{a}, out); err != nil {
		t.Fatalf("ConcatWavs single: %v", err)
	}
	w, err := parseWav(out)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(w.pcm, pcm) {
		t.Fatalf("pcm = %v, want %v", w.pcm, pcm)
	}
}
