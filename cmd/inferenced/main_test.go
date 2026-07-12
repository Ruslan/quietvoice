package main

import (
	"bytes"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"quietvoice/internal/inference/local"
)

// fakeCrispasr is a minimal stand-in for a `crispasr --server` voice registry:
// it counts POST /v1/voices uploads so a test can assert the handler never
// reaches upstream when it rejects a request.
func fakeCrispasr(t *testing.T, uploads *int32) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/voices", func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"voices":[]}`))
	})
	mux.HandleFunc("POST /v1/voices", func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(uploads, 1)
		w.WriteHeader(http.StatusCreated)
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv
}

// buildVoiceRequest builds a POST /v1/voices multipart request. A nil transcript
// omits the field entirely; a non-nil value sets it (possibly empty).
func buildVoiceRequest(t *testing.T, name string, transcript *string, wav []byte) *http.Request {
	t.Helper()
	var body bytes.Buffer
	mw := multipart.NewWriter(&body)
	if name != "" {
		_ = mw.WriteField("name", name)
	}
	if transcript != nil {
		_ = mw.WriteField("transcript", *transcript)
	}
	part, _ := mw.CreateFormFile("voice", "ded.wav")
	_, _ = part.Write(wav)
	_ = mw.Close()
	req := httptest.NewRequest(http.MethodPost, "/v1/voices", &body)
	req.Header.Set("Content-Type", mw.FormDataContentType())
	return req
}

// TestUploadVoiceRejectsEmptyTranscript asserts the handler returns 400 and never
// contacts the upstream crispasr pool when the transcript is missing or blank.
func TestUploadVoiceRejectsEmptyTranscript(t *testing.T) {
	var uploads int32
	srv := fakeCrispasr(t, &uploads)
	eng := local.New(local.Config{WorkDir: t.TempDir(), TTSServerURLs: []string{srv.URL}, TTSVoice: "ded"})
	n := &node{eng: eng, local: eng, workDir: t.TempDir()}

	blank := "   \t\n "
	empty := ""
	for _, tc := range []struct {
		name       string
		transcript *string
	}{
		{"missing", nil},
		{"empty", &empty},
		{"whitespace", &blank},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rr := httptest.NewRecorder()
			n.uploadVoice(rr, buildVoiceRequest(t, "ded", tc.transcript, []byte("RIFFref")))
			if rr.Code != http.StatusBadRequest {
				t.Fatalf("status = %d, want 400; body = %q", rr.Code, rr.Body.String())
			}
			if !strings.Contains(rr.Body.String(), "transcript is required") {
				t.Fatalf("body = %q, want it to mention the transcript requirement", rr.Body.String())
			}
		})
	}
	if got := atomic.LoadInt32(&uploads); got != 0 {
		t.Fatalf("upstream received %d uploads; a textless voice must never reach it", got)
	}
}

// TestUploadVoiceAcceptsTranscript confirms a valid transcript still registers
// (201) and reaches the upstream pool — the fix must not break valid uploads.
func TestUploadVoiceAcceptsTranscript(t *testing.T) {
	var uploads int32
	srv := fakeCrispasr(t, &uploads)
	eng := local.New(local.Config{WorkDir: t.TempDir(), TTSServerURLs: []string{srv.URL}, TTSVoice: "ded"})
	n := &node{eng: eng, local: eng, workDir: t.TempDir()}

	transcript := "привет"
	rr := httptest.NewRecorder()
	n.uploadVoice(rr, buildVoiceRequest(t, "ded", &transcript, []byte("RIFFref")))
	if rr.Code != http.StatusCreated {
		t.Fatalf("status = %d, want 201; body = %q", rr.Code, rr.Body.String())
	}
	if got := atomic.LoadInt32(&uploads); got != 1 {
		t.Fatalf("upstream received %d uploads, want 1", got)
	}
}

// recordingBackend is a fake replica that records the exact path + query it was
// reached at, so a raw-tunnel test can prove the request was forwarded verbatim
// (and that the routing ?model= param was stripped).
func recordingBackend(t *testing.T, gotPath, gotQuery *string) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		*gotPath = r.URL.Path
		*gotQuery = r.URL.RawQuery
		_, _ = w.Write([]byte("hello from replica"))
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv
}

// rawMux builds a ServeMux with only the raw tunnel route registered on n, so a
// test can drive it through the same pattern-matching the real server uses.
func rawMux(n *node) *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("/raw/{role}/{index}/{path...}", n.rawProxy)
	return mux
}

// TestRawProxyForwardsToReplica proves /raw/tts/1/<path> reverse-proxies verbatim
// to the selected replica, forwarding the upstream path + query while stripping
// the routing ?model= param.
func TestRawProxyForwardsToReplica(t *testing.T) {
	var gotPath, gotQuery string
	backend := recordingBackend(t, &gotPath, &gotQuery)
	eng := local.New(local.Config{WorkDir: t.TempDir(), TTSServerURLs: []string{backend.URL}})
	n := &node{eng: eng, local: eng, workDir: t.TempDir()}
	mux := rawMux(n)

	rr := httptest.NewRecorder()
	mux.ServeHTTP(rr, httptest.NewRequest(http.MethodGet, "/raw/tts/1/v1/voices?model=&limit=5", nil))

	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body = %q", rr.Code, rr.Body.String())
	}
	if rr.Body.String() != "hello from replica" {
		t.Fatalf("body = %q, want the replica's response", rr.Body.String())
	}
	if gotPath != "/v1/voices" {
		t.Fatalf("upstream path = %q, want /v1/voices", gotPath)
	}
	if strings.Contains(gotQuery, "model") {
		t.Fatalf("upstream query = %q, must not carry the routing 'model' param", gotQuery)
	}
	if !strings.Contains(gotQuery, "limit=5") {
		t.Fatalf("upstream query = %q, want the real 'limit=5' preserved", gotQuery)
	}
}

// TestRawProxyMissingReplica proves an out-of-range index is a clean 404, not a
// proxy crash.
func TestRawProxyMissingReplica(t *testing.T) {
	eng := local.New(local.Config{WorkDir: t.TempDir()})
	n := &node{eng: eng, local: eng, workDir: t.TempDir()}
	mux := rawMux(n)

	rr := httptest.NewRecorder()
	mux.ServeHTTP(rr, httptest.NewRequest(http.MethodGet, "/raw/tts/1/v1/voices", nil))
	if rr.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want 404 for a nonexistent replica", rr.Code)
	}
}
