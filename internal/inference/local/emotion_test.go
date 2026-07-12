package local

import "testing"

// TestStripMetaLine covers the lenient parse of Gemma's trailing type/tone/urgency
// line out of the raw output. Small quantized Gemmas mangle the format constantly, so
// the parser must survive wrong brackets, ':'/'=' mixups, '|'/'·'/';' separators,
// missing fields, a missing line, and a hallucinated type — and it must only treat a
// line as metadata when it really is one (prose like "ringtone" / "what type" is safe).
func TestStripMetaLine(t *testing.T) {
	cases := []struct {
		name                                        string
		in                                          string
		wantClean, wantType, wantTone, wantUrgen string
	}{
		{"canonical", "Turn off that server.\n[[type: command | tone: frustrated | urgency: high]]", "Turn off that server.", "command", "frustrated", "high"},
		{"equals-and-dot-sep", "Deploy it.\ntype = command · tone = frustrated · urgency = high", "Deploy it.", "command", "frustrated", "high"},
		{"semicolon-sep-no-brackets", "Maybe we cache it?\ntype: question; tone: calm; urgency: low", "Maybe we cache it?", "question", "calm", "low"},
		{"quoted-values", "Ship now.\n[[type: \"decision\" | tone: \"angry\" | urgency: \"high\"]]", "Ship now.", "decision", "angry", "high"},
		{"no-type-tone-urgency-only", "Wait.\ntone: neutral | urgency: high", "Wait.", "", "neutral", "high"},
		{"hallucinated-type-dropped", "Do the thing.\n[[type: banana | tone: neutral | urgency: low]]", "Do the thing.", "", "neutral", "low"},
		{"trailing-blank-lines", "Ok.\n[[type: decision | tone: calm | urgency: low]]\n\n", "Ok.", "decision", "calm", "low"},
		{"multiline-clean-preserved", "First point.\nSecond POINT (emphatic).\n[[type: suggestion | tone: emphatic | urgency: normal]]", "First point.\nSecond POINT (emphatic).", "suggestion", "emphatic", "normal"},
		{"no-meta-line", "Just a calm sentence with no tags.", "Just a calm sentence with no tags.", "", "", ""},
		{"meta-only-cleaned-empty", "[[type: command | tone: angry | urgency: high]]", "", "command", "angry", "high"},
		{"tone-word-in-real-text-not-eaten", "Set the ringtone to silent.", "Set the ringtone to silent.", "", "", ""},
		{"type-word-in-prose-not-eaten", "What type of cache should we use?", "What type of cache should we use?", "", "", ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			gotClean, gotType, gotTone, gotUrgen := stripMetaLine(c.in)
			if gotClean != c.wantClean {
				t.Errorf("cleaned = %q, want %q", gotClean, c.wantClean)
			}
			if gotType != c.wantType {
				t.Errorf("type = %q, want %q", gotType, c.wantType)
			}
			if gotTone != c.wantTone {
				t.Errorf("tone = %q, want %q", gotTone, c.wantTone)
			}
			if gotUrgen != c.wantUrgen {
				t.Errorf("urgency = %q, want %q", gotUrgen, c.wantUrgen)
			}
		})
	}
}
