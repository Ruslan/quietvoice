package local

import (
	"context"
	"sync"
)

// worker is one hot inference server endpoint in a role pool. A crispasr /
// llama-server process serializes internally, so a worker handles exactly one
// request at a time — the pool enforces that by handing a worker to a single
// caller until it is released.
type worker struct {
	url     string
	role    string
	model   string // model this worker serves ("" = default pool)
	managed bool   // true when the supervisor launched the process (has pid)
	pid     int    // 0 when unmanaged (an external URL from config)
	removed bool   // set under pool.mu; a removed worker is never handed out again
}

// pool routes requests to an IDLE worker (idle-checkout, deliberately NOT
// round-robin): because each backend serializes internally, blindly rotating
// would stack a second request on a busy worker while a neighbour sits idle.
// Membership is mutable at runtime so the admin API can scale a role up and
// down live without disturbing in-flight requests.
type pool struct {
	mu   sync.Mutex
	all  map[string]*worker // current membership, by url
	idle chan *worker       // the set of currently-idle workers
}

// newPool returns a pool whose idle queue can hold up to capacity workers.
// capacity must be >= the maximum number of workers ever added concurrently,
// otherwise add/release would block; callers size it as maxReplicas + externals.
func newPool(capacity int) *pool {
	if capacity < 1 {
		capacity = 1
	}
	return &pool{all: make(map[string]*worker), idle: make(chan *worker, capacity)}
}

// add registers a worker and marks it idle. A duplicate url is ignored.
func (p *pool) add(w *worker) {
	p.mu.Lock()
	if _, ok := p.all[w.url]; ok {
		p.mu.Unlock()
		return
	}
	p.all[w.url] = w
	p.mu.Unlock()
	p.idle <- w
}

// checkout blocks until an idle, non-removed worker is available or ctx is done.
// The caller MUST release the returned worker.
func (p *pool) checkout(ctx context.Context) (*worker, error) {
	for {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case w := <-p.idle:
			p.mu.Lock()
			removed := w.removed
			p.mu.Unlock()
			if removed {
				continue // its process is gone; drop this idle token and try the next
			}
			return w, nil
		}
	}
}

// release returns a worker to the idle set. A worker removed while busy is
// dropped instead of requeued.
func (p *pool) release(w *worker) {
	p.mu.Lock()
	removed := w.removed
	p.mu.Unlock()
	if removed {
		return
	}
	p.idle <- w
}

// remove drops a worker from membership. If it is currently idle its token is
// discarded lazily on the next checkout; if busy it is dropped on release.
// Returns the removed worker (nil if unknown).
func (p *pool) remove(url string) *worker {
	p.mu.Lock()
	defer p.mu.Unlock()
	w, ok := p.all[url]
	if !ok {
		return nil
	}
	w.removed = true
	delete(p.all, url)
	return w
}

// size reports the current number of workers (idle or busy).
func (p *pool) size() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.all)
}

// urls returns a snapshot of member urls.
func (p *pool) urls() []string {
	p.mu.Lock()
	defer p.mu.Unlock()
	out := make([]string, 0, len(p.all))
	for u := range p.all {
		out = append(out, u)
	}
	return out
}

// workers returns a snapshot of the member workers (value copies are unsafe to
// mutate; used read-only for status reporting).
func (p *pool) workers() []worker {
	p.mu.Lock()
	defer p.mu.Unlock()
	out := make([]worker, 0, len(p.all))
	for _, w := range p.all {
		out = append(out, *w)
	}
	return out
}
