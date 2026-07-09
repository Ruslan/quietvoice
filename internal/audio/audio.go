// Package audio handles container/codec conversion between the inference tools
// and Telegram. TTS produces WAV; Telegram voice notes require OGG/Opus. All
// conversions shell out to ffmpeg, which is available on both macOS and Linux.
package audio

import (
	"bytes"
	"context"
	"fmt"
	"os/exec"
	"strings"
)

// FFmpegBin is the ffmpeg binary name/path (overridable for tests).
var FFmpegBin = "ffmpeg"

// WavToOggOpus transcodes a WAV file to OGG/Opus (mono) suitable for Telegram
// sendVoice, writing to oggPath.
func WavToOggOpus(ctx context.Context, wavPath, oggPath string) error {
	cmd := exec.CommandContext(ctx, FFmpegBin,
		"-y",
		"-i", wavPath,
		"-ac", "1",
		"-c:a", "libopus",
		"-b:a", "32k",
		oggPath,
	)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("ffmpeg wav->ogg: %w: %s", err, lastLine(stderr.String()))
	}
	return nil
}

// ToMono16kWav transcodes any input audio to 16 kHz mono PCM WAV, the format
// llama.cpp multimodal (llama-mtmd-cli) reliably decodes for audio input.
func ToMono16kWav(ctx context.Context, srcPath, wavPath string) error {
	cmd := exec.CommandContext(ctx, FFmpegBin,
		"-y",
		"-i", srcPath,
		"-ar", "16000",
		"-ac", "1",
		"-c:a", "pcm_s16le",
		wavPath,
	)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("ffmpeg ->16k wav: %w: %s", err, lastLine(stderr.String()))
	}
	return nil
}

func lastLine(s string) string {
	s = strings.TrimSpace(s)
	if i := strings.LastIndexByte(s, '\n'); i >= 0 {
		return s[i+1:]
	}
	return s
}
