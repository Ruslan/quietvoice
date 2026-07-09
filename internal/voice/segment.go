package voice

import (
	"strings"
	"unicode/utf8"
)

// TTS chunk sizing. The crispasr qwen3-tts backend caps generation at ~512
// audio tokens (~50-55 s), which empirically maps to a few hundred characters
// before it returns empty audio. We therefore pack text into small chunks and
// synthesize them separately (and concurrently across the replica pool).
//
// Sizes are measured in RUNES (characters), not bytes, so Cyrillic — 2 bytes
// per letter in UTF-8 — is treated on the same footing as Latin.
const (
	// chunkTarget is the size we greedily pack up to when combining sentences.
	chunkTarget = 220
	// chunkHardCap is the ceiling any single chunk may reach; a sentence longer
	// than this is broken on word boundaries.
	chunkHardCap = 300
)

// sentenceTerminators end a sentence-ish segment. Kept with their sentence.
func isTerminator(r rune) bool {
	switch r {
	case '.', '!', '?', ';', '…':
		return true
	}
	return false
}

// splitText breaks text into ordered synthesis chunks. It first splits on
// sentence terminators (. ! ? … ;) and newlines — for both Cyrillic and Latin —
// keeping each terminator attached to its sentence, then greedily packs whole
// sentences into chunks up to chunkTarget characters without ever splitting a
// word. A single sentence longer than chunkHardCap is split on word (space)
// boundaries so every chunk stays under the cap. Whitespace is collapsed.
// Returns chunks in original order; an all-whitespace input yields nil.
func splitText(text string) []string {
	segments := splitSentences(text)
	var chunks []string
	var cur string
	flush := func() {
		if cur != "" {
			chunks = append(chunks, cur)
			cur = ""
		}
	}
	for _, seg := range segments {
		if seg == "" {
			continue
		}
		// A sentence that alone exceeds the hard cap is broken on words.
		if utf8.RuneCountInString(seg) > chunkHardCap {
			flush()
			chunks = append(chunks, packWords(seg)...)
			continue
		}
		switch {
		case cur == "":
			cur = seg
		case utf8.RuneCountInString(cur)+1+utf8.RuneCountInString(seg) <= chunkTarget:
			cur = cur + " " + seg
		default:
			flush()
			cur = seg
		}
	}
	flush()
	return chunks
}

// splitSentences splits text on sentence terminators and newlines, keeping the
// terminator (and any run of trailing terminators, e.g. "?!" or "...") with the
// sentence, and collapsing internal whitespace in each returned segment.
func splitSentences(text string) []string {
	var segs []string
	var b strings.Builder
	flush := func() {
		if s := collapseSpaces(b.String()); s != "" {
			segs = append(segs, s)
		}
		b.Reset()
	}
	runes := []rune(text)
	for i := 0; i < len(runes); i++ {
		r := runes[i]
		if r == '\n' || r == '\r' {
			flush()
			continue
		}
		b.WriteRune(r)
		if isTerminator(r) {
			for i+1 < len(runes) && isTerminator(runes[i+1]) {
				i++
				b.WriteRune(runes[i])
			}
			flush()
		}
	}
	flush()
	return segs
}

// packWords greedily packs the space-separated words of seg into chunks up to
// chunkTarget characters, never splitting a word. A lone word longer than the
// cap becomes its own (oversized) chunk — the only way to avoid a mid-word cut.
func packWords(seg string) []string {
	words := strings.Fields(seg)
	var out []string
	var cur string
	for _, w := range words {
		switch {
		case cur == "":
			cur = w
		case utf8.RuneCountInString(cur)+1+utf8.RuneCountInString(w) <= chunkTarget:
			cur = cur + " " + w
		default:
			out = append(out, cur)
			cur = w
		}
	}
	if cur != "" {
		out = append(out, cur)
	}
	return out
}

// collapseSpaces trims and collapses every run of Unicode whitespace to a single
// ASCII space.
func collapseSpaces(s string) string {
	return strings.Join(strings.Fields(s), " ")
}
