package audio

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"os"
)

// wavInfo is the format + PCM payload parsed out of a RIFF/WAVE file.
type wavInfo struct {
	audioFormat   uint16
	numChannels   uint16
	sampleRate    uint32
	bitsPerSample uint16
	pcm           []byte
}

// ConcatWavs concatenates the PCM payloads of the given WAV files, in order,
// into a single valid WAV at out. Every input must share the same audio format
// (codec, channels, sample rate, bit depth); a mismatch or a malformed/short
// file is reported as an error. crispasr emits 24 kHz mono PCM16, so in normal
// operation all chunks agree trivially.
func ConcatWavs(paths []string, out string) error {
	if len(paths) == 0 {
		return fmt.Errorf("ConcatWavs: no input files")
	}
	var first *wavInfo
	var pcm []byte
	for _, p := range paths {
		w, err := parseWav(p)
		if err != nil {
			return err
		}
		if first == nil {
			first = w
		} else if w.audioFormat != first.audioFormat ||
			w.numChannels != first.numChannels ||
			w.sampleRate != first.sampleRate ||
			w.bitsPerSample != first.bitsPerSample {
			return fmt.Errorf("ConcatWavs: format mismatch in %s (got fmt=%d %dch/%dHz/%dbit, want fmt=%d %dch/%dHz/%dbit)",
				p, w.audioFormat, w.numChannels, w.sampleRate, w.bitsPerSample,
				first.audioFormat, first.numChannels, first.sampleRate, first.bitsPerSample)
		}
		pcm = append(pcm, w.pcm...)
	}
	return writeWav(out, first, pcm)
}

// parseWav reads a RIFF/WAVE file and extracts its fmt parameters and data PCM.
func parseWav(path string) (*wavInfo, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("ConcatWavs: read %s: %w", path, err)
	}
	if len(b) < 12 || string(b[0:4]) != "RIFF" || string(b[8:12]) != "WAVE" {
		return nil, fmt.Errorf("ConcatWavs: %s is not a RIFF/WAVE file", path)
	}
	var w wavInfo
	var haveFmt, haveData bool
	off := 12
	for off+8 <= len(b) {
		id := string(b[off : off+4])
		size := int(binary.LittleEndian.Uint32(b[off+4 : off+8]))
		body := off + 8
		end := body + size
		if end > len(b) {
			return nil, fmt.Errorf("ConcatWavs: %s chunk %q overruns file (size %d)", path, id, size)
		}
		switch id {
		case "fmt ":
			if size < 16 {
				return nil, fmt.Errorf("ConcatWavs: %s has a short fmt chunk (%d bytes)", path, size)
			}
			w.audioFormat = binary.LittleEndian.Uint16(b[body : body+2])
			w.numChannels = binary.LittleEndian.Uint16(b[body+2 : body+4])
			w.sampleRate = binary.LittleEndian.Uint32(b[body+4 : body+8])
			w.bitsPerSample = binary.LittleEndian.Uint16(b[body+14 : body+16])
			haveFmt = true
		case "data":
			w.pcm = b[body:end]
			haveData = true
		}
		off = end
		if size%2 == 1 { // chunks are word-aligned; skip pad byte
			off++
		}
	}
	if !haveFmt {
		return nil, fmt.Errorf("ConcatWavs: %s has no fmt chunk", path)
	}
	if !haveData {
		return nil, fmt.Errorf("ConcatWavs: %s has no data chunk", path)
	}
	return &w, nil
}

// writeWav writes a canonical 44-byte-header PCM WAV combining f's format with
// the supplied PCM payload.
func writeWav(path string, f *wavInfo, pcm []byte) error {
	dataLen := uint32(len(pcm))
	byteRate := f.sampleRate * uint32(f.numChannels) * uint32(f.bitsPerSample) / 8
	blockAlign := f.numChannels * f.bitsPerSample / 8

	var buf bytes.Buffer
	buf.Grow(44 + len(pcm))
	buf.WriteString("RIFF")
	_ = binary.Write(&buf, binary.LittleEndian, uint32(36+dataLen))
	buf.WriteString("WAVE")
	buf.WriteString("fmt ")
	_ = binary.Write(&buf, binary.LittleEndian, uint32(16))
	_ = binary.Write(&buf, binary.LittleEndian, f.audioFormat)
	_ = binary.Write(&buf, binary.LittleEndian, f.numChannels)
	_ = binary.Write(&buf, binary.LittleEndian, f.sampleRate)
	_ = binary.Write(&buf, binary.LittleEndian, byteRate)
	_ = binary.Write(&buf, binary.LittleEndian, blockAlign)
	_ = binary.Write(&buf, binary.LittleEndian, f.bitsPerSample)
	buf.WriteString("data")
	_ = binary.Write(&buf, binary.LittleEndian, dataLen)
	buf.Write(pcm)

	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		return fmt.Errorf("ConcatWavs: write %s: %w", path, err)
	}
	return nil
}
