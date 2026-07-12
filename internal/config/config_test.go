package config

import (
	"path/filepath"
	"testing"
)

// TestLoadDefaultsTTSVoiceDir proves the safe default: with TTS_VOICE_DIR unset, the
// voice registry lands under WorkDir so crispasr always has somewhere to persist an
// uploaded voice (a missing --voice-dir makes POST /v1/voices 400 and breaks say).
// An explicit TTS_VOICE_DIR still wins.
func TestLoadDefaultsTTSVoiceDir(t *testing.T) {
	// env() treats a set-but-empty var as unset, so "" simulates an unset var while
	// t.Setenv still restores the environment after the test.
	t.Setenv("TTS_VOICE_DIR", "")
	t.Setenv("WORK_DIR", "")

	c := Load()
	want := filepath.Join("voice_sessions/work", "voices") // WorkDir default + /voices
	if c.TTSVoiceDir != want {
		t.Fatalf("TTSVoiceDir = %q, want default %q", c.TTSVoiceDir, want)
	}

	t.Setenv("WORK_DIR", "/data/qv")
	if got := Load().TTSVoiceDir; got != "/data/qv/voices" {
		t.Fatalf("TTSVoiceDir = %q, want /data/qv/voices (derived from WorkDir)", got)
	}

	t.Setenv("TTS_VOICE_DIR", "/custom/voices")
	if got := Load().TTSVoiceDir; got != "/custom/voices" {
		t.Fatalf("TTSVoiceDir = %q, want /custom/voices (explicit env wins)", got)
	}
}
