package voice

import (
	"strings"
	"testing"
	"unicode/utf8"
)

// maxChunkRunes is the invariant every chunk must satisfy unless it is a single
// unsplittable word longer than the cap.
func assertUnderCap(t *testing.T, chunks []string) {
	t.Helper()
	for _, c := range chunks {
		if n := utf8.RuneCountInString(c); n > chunkHardCap {
			if len(strings.Fields(c)) > 1 {
				t.Fatalf("chunk of %d runes exceeds hard cap %d and is splittable: %q", n, chunkHardCap, c)
			}
		}
	}
}

// assertNoSplitWord verifies every word in the original text appears intact in
// some chunk (i.e. no word was cut in half).
func assertNoSplitWord(t *testing.T, text string, chunks []string) {
	t.Helper()
	joined := strings.Join(chunks, " ")
	for _, w := range strings.Fields(text) {
		if !strings.Contains(joined, w) {
			t.Fatalf("word %q was split or lost; chunks=%q", w, chunks)
		}
	}
}

func TestSplitSingleShort(t *testing.T) {
	got := splitText("Just a short line.")
	if len(got) != 1 || got[0] != "Just a short line." {
		t.Fatalf("got %q, want single chunk", got)
	}
}

func TestSplitEmpty(t *testing.T) {
	if got := splitText("   \n\t  "); got != nil {
		t.Fatalf("whitespace-only should yield nil, got %q", got)
	}
}

func TestSplitSentencePacking(t *testing.T) {
	// Several short sentences should pack together up to the target, not become
	// one chunk per sentence.
	text := "One. Two. Three. Four. Five. Six."
	got := splitText(text)
	if len(got) != 1 {
		t.Fatalf("short sentences should pack into one chunk, got %d: %q", len(got), got)
	}
	if got[0] != "One. Two. Three. Four. Five. Six." {
		t.Fatalf("packed chunk = %q", got[0])
	}
}

func TestSplitPacksToCap(t *testing.T) {
	// 20 sentences of ~15 chars each = ~300 chars → must break into >1 chunk,
	// each within the cap, with terminators kept.
	var b strings.Builder
	for i := 0; i < 20; i++ {
		b.WriteString("Sentence here. ")
	}
	got := splitText(b.String())
	if len(got) < 2 {
		t.Fatalf("expected multiple chunks, got %d", len(got))
	}
	assertUnderCap(t, got)
	for _, c := range got {
		if n := utf8.RuneCountInString(c); n > chunkTarget && strings.Count(c, ".") > 1 {
			t.Fatalf("packed chunk %d runes exceeds target %d: %q", n, chunkTarget, c)
		}
	}
}

func TestSplitMultibyte(t *testing.T) {
	text := "Primer oración. ¡Segunda oración! ¿Tercera oración? Cuarta."
	got := splitText(text)
	if len(got) == 0 {
		t.Fatal("no chunks for multibyte text")
	}
	assertUnderCap(t, got)
	assertNoSplitWord(t, text, got)
	// The three terminator types must all act as boundaries but stay attached.
	joined := strings.Join(got, " ")
	for _, term := range []string{"oración.", "oración!", "oración?", "Cuarta."} {
		if !strings.Contains(joined, term) {
			t.Fatalf("terminator not kept with sentence: missing %q in %q", term, got)
		}
	}
}

func TestSplitMultibyteRespectsRuneCap(t *testing.T) {
	// Long Multibyte text: Spanish accented characters are 2 bytes/char, so a byte-based cap would
	// over-fragment. Verify chunks are sized by runes, comfortably multi-chunk
	// but each under the rune cap.
	sentence := "Esta es una oración bastante larga en español para fines de prueba. "
	got := splitText(strings.Repeat(sentence, 12))
	if len(got) < 2 {
		t.Fatalf("expected multiple chunks, got %d", len(got))
	}
	assertUnderCap(t, got)
}

func TestSplitNoPunctuation(t *testing.T) {
	// No terminators at all → word packing.
	var words []string
	for i := 0; i < 120; i++ {
		words = append(words, "word")
	}
	text := strings.Join(words, " ") // ~600 chars
	got := splitText(text)
	if len(got) < 2 {
		t.Fatalf("expected word-packed multi-chunk, got %d", len(got))
	}
	assertUnderCap(t, got)
	assertNoSplitWord(t, text, got)
}

func TestSplitOverlongSentenceWordFallback(t *testing.T) {
	// A single sentence far over the hard cap, no internal terminators → must be
	// split on word boundaries, never mid-word, each chunk under the cap.
	var words []string
	for i := 0; i < 100; i++ {
		words = append(words, "alpha")
	}
	text := strings.Join(words, " ") + "." // one 500+ char sentence
	got := splitText(text)
	if len(got) < 2 {
		t.Fatalf("overlong sentence must split, got %d chunks", len(got))
	}
	assertUnderCap(t, got)
	assertNoSplitWord(t, text, got)
}

func TestSplitNewlineBoundary(t *testing.T) {
	// Newlines are boundaries even without punctuation.
	got := splitText("first line no dot\nsecond line no dot")
	if len(got) != 1 {
		// Two short lines pack into one chunk but must both survive intact.
		t.Logf("chunks: %q", got)
	}
	joined := strings.Join(got, " ")
	if !strings.Contains(joined, "first line no dot") || !strings.Contains(joined, "second line no dot") {
		t.Fatalf("newline lines lost: %q", got)
	}
}

func TestSplitNeverSplitsMidWordLongWord(t *testing.T) {
	// A single word longer than the cap cannot be split without cutting a word;
	// it must survive intact as its own chunk.
	long := strings.Repeat("x", chunkHardCap+50)
	got := splitText(long)
	if len(got) != 1 || got[0] != long {
		t.Fatalf("overlong single word must stay intact, got %d chunks", len(got))
	}
}
