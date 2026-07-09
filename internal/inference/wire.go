package inference

// HTTP wire contract between the remote Engine adapter (control plane) and the
// standalone inference node (inferenced). inferenced speaks ONE OpenAI-compatible
// contract inbound and outbound (see inferenced-unified-api-plan.md): the same
// routes a raw `crispasr --server` exposes, so a client can hit either.
const (
	// RouteSpeech is the OpenAI TTS route: a JSON {model?, input, voice,
	// response_format} body returns raw audio bytes. Replaces the old /v1/tts.
	RouteSpeech = "/v1/audio/speech"
	// RouteTranscriptions is the OpenAI ASR route: multipart file[, model,
	// language] returns JSON {text}.
	RouteTranscriptions = "/v1/audio/transcriptions"
	// RouteVoices lists (GET) and registers (POST multipart name/transcript/voice)
	// reference voices across the hot pool.
	RouteVoices = "/v1/voices"
	// RouteInterpret is the QuietVoice-native rich-intent route (NOT OpenAI): it
	// accepts multipart "meta" (JSON InterpretRequest) + file "audio" and returns
	// a JSON InterpretResult (intent/uncertainties/per-recognizer transcripts).
	// Retained alongside the OpenAI routes because that structure does not fit
	// OpenAI's flat {text} transcription schema — see the assisted-intent decision
	// in inferenced-unified-api-plan.md §4.1.
	RouteInterpret = "/v1/interpret"
	// RouteHealth returns 200 when the inference backend is ready.
	RouteHealth = "/healthz"

	// InterpretMetaField and InterpretAudioField are the multipart field names
	// for RouteInterpret.
	InterpretMetaField  = "meta"
	InterpretAudioField = "audio"

	// TranscribeFileField is the multipart file field for RouteTranscriptions
	// (OpenAI names it "file").
	TranscribeFileField = "file"
)

// SpeechRequest is the OpenAI /v1/audio/speech body inferenced accepts inbound
// and the remote adapter sends outbound.
type SpeechRequest struct {
	Model          string `json:"model,omitempty"`
	Input          string `json:"input"`
	Voice          string `json:"voice,omitempty"`
	ResponseFormat string `json:"response_format,omitempty"`
}

// TranscriptionResponse is the OpenAI /v1/audio/transcriptions reply.
type TranscriptionResponse struct {
	Text string `json:"text"`
}

// VoicesResponse is the GET /v1/voices reply.
type VoicesResponse struct {
	Voices []string `json:"voices"`
}
