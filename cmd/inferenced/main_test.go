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

	transcript := "hello"
	rr := httptest.NewRecorder()
	n.uploadVoice(rr, buildVoiceRequest(t, "ded", &transcript, []byte("RIFFref")))
	if rr.Code != http.StatusCreated {
		t.Fatalf("status = %d, want 201; body = %q", rr.Code, rr.Body.String())
	}
	if got := atomic.LoadInt32(&uploads); got != 1 {
		t.Fatalf("upstream received %d uploads, want 1", got)
	}
}
