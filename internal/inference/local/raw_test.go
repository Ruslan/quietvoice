package local

import "testing"

// TestRawTargetPortOrdering proves the raw tunnel indexes replicas by ascending
// port (stable, human-meaningful) regardless of the order they were added, and
// that out-of-range / unknown lookups fail closed.
func TestRawTargetPortOrdering(t *testing.T) {
	e := New(Config{})
	p := e.ensurePool("tts", "")
	// Added out of port order on purpose; index must still follow port order.
	p.add(&worker{url: "http://127.0.0.1:9102", role: "tts"})
	p.add(&worker{url: "http://127.0.0.1:9100", role: "tts"})
	p.add(&worker{url: "http://127.0.0.1:9101", role: "tts"})

	want := []string{"http://127.0.0.1:9100", "http://127.0.0.1:9101", "http://127.0.0.1:9102"}
	for i, w := range want {
		got, ok := e.RawTarget("tts", "", i+1)
		if !ok || got != w {
			t.Fatalf("RawTarget(tts,#%d) = %q,%v; want %q,true", i+1, got, ok, w)
		}
	}

	for _, bad := range []int{0, -1, 4} {
		if got, ok := e.RawTarget("tts", "", bad); ok {
			t.Fatalf("RawTarget(tts,#%d) = %q,true; want out-of-range miss", bad, got)
		}
	}
	if _, ok := e.RawTarget("bogus", "", 1); ok {
		t.Fatal("RawTarget with unknown role should miss")
	}
	if _, ok := e.RawTarget("gemma", "", 1); ok {
		t.Fatal("RawTarget on a pool that was never created should miss")
	}
}

// TestRawTargetModelScoping proves ?model= selects a non-default pool and that
// the default ("") and a named model pool index independently.
func TestRawTargetModelScoping(t *testing.T) {
	e := New(Config{})
	e.ensurePool("tts", "").add(&worker{url: "http://127.0.0.1:9100", role: "tts"})
	e.ensurePool("tts", "big").add(&worker{url: "http://127.0.0.1:9200", role: "tts", model: "big"})

	if got, ok := e.RawTarget("tts", "", 1); !ok || got != "http://127.0.0.1:9100" {
		t.Fatalf("default pool #1 = %q,%v", got, ok)
	}
	if got, ok := e.RawTarget("tts", "big", 1); !ok || got != "http://127.0.0.1:9200" {
		t.Fatalf("model 'big' pool #1 = %q,%v", got, ok)
	}
	// The default pool has only one replica; #2 must miss even though 'big' exists.
	if _, ok := e.RawTarget("tts", "", 2); ok {
		t.Fatal("default pool #2 should miss (only one replica)")
	}
}
