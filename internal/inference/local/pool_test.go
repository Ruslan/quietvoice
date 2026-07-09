package local

import (
	"context"
	"testing"
	"time"
)

func TestPoolIdleCheckout(t *testing.T) {
	p := newPool(2)
	p.add(&worker{url: "a"})
	p.add(&worker{url: "b"})
	if p.size() != 2 {
		t.Fatalf("size = %d, want 2", p.size())
	}

	ctx := context.Background()
	w1, err := p.checkout(ctx)
	if err != nil {
		t.Fatalf("checkout 1: %v", err)
	}
	w2, err := p.checkout(ctx)
	if err != nil {
		t.Fatalf("checkout 2: %v", err)
	}
	if w1.url == w2.url {
		t.Fatalf("idle-checkout handed the same worker twice: %q", w1.url)
	}

	// A third checkout must block until one is released (idle-checkout, not
	// round-robin stacking).
	got := make(chan *worker, 1)
	go func() {
		w, err := p.checkout(ctx)
		if err == nil {
			got <- w
		}
	}()
	select {
	case <-got:
		t.Fatal("checkout returned while all workers were busy")
	case <-time.After(50 * time.Millisecond):
	}

	p.release(w1)
	select {
	case w := <-got:
		if w.url != w1.url {
			t.Fatalf("released %q but got %q", w1.url, w.url)
		}
	case <-time.After(time.Second):
		t.Fatal("checkout did not unblock after release")
	}
	p.release(w2)
}

func TestPoolRemoveDropsWorker(t *testing.T) {
	p := newPool(2)
	p.add(&worker{url: "a"})
	if p.remove("a") == nil {
		t.Fatal("remove(a) returned nil for a present worker")
	}
	if p.size() != 0 {
		t.Fatalf("size after remove = %d, want 0", p.size())
	}
	// The stale idle token must be dropped, not handed out: checkout should now
	// block and thus honour a deadline.
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	if _, err := p.checkout(ctx); err == nil {
		t.Fatal("checkout returned a removed worker instead of blocking")
	}
}

func TestPoolReleaseAfterRemoveDoesNotRequeue(t *testing.T) {
	p := newPool(2)
	p.add(&worker{url: "a"})
	w, err := p.checkout(context.Background())
	if err != nil {
		t.Fatalf("checkout: %v", err)
	}
	p.remove("a") // removed while busy
	p.release(w)  // must NOT requeue the dead worker
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	if _, err := p.checkout(ctx); err == nil {
		t.Fatal("released a removed worker back into the pool")
	}
}
