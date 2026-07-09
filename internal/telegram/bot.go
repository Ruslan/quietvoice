// Package telegram implements the QuietVoice voice transport over the Telegram
// Bot API using long polling (getUpdates) — no public URL or webhook required,
// so it works from a laptop or behind NAT. It sends spoken updates as voice
// notes, notifies the user when an agent is waiting, and delivers incoming
// voice messages (already downloaded to disk) to a handler.
package telegram

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"time"
)

// Config configures the bot and its security allowlist.
type Config struct {
	Token          string
	AllowedUserIDs []int64
	AllowedChatIDs []int64
	DownloadDir    string // where incoming voice files are stored
}

// Bot is a minimal Telegram Bot API client.
type Bot struct {
	cfg     Config
	client  *http.Client
	apiBase string
	fileAPI string
}

// IncomingVoice is a downloaded Telegram voice note handed to the handler.
type IncomingVoice struct {
	ChatID    int64
	UserID    int64
	MessageID int64
	FileID    string
	LocalPath string
	Duration  int
}

// VoiceHandler processes an incoming voice note.
type VoiceHandler func(ctx context.Context, v IncomingVoice)

// New builds a Bot. Token must be non-empty.
func New(cfg Config) (*Bot, error) {
	if cfg.Token == "" {
		return nil, fmt.Errorf("telegram: empty bot token")
	}
	if cfg.DownloadDir == "" {
		cfg.DownloadDir = filepath.Join(os.TempDir(), "quietvoice-voice")
	}
	return &Bot{
		cfg:     cfg,
		client:  &http.Client{Timeout: 60 * time.Second},
		apiBase: "https://api.telegram.org/bot" + cfg.Token,
		fileAPI: "https://api.telegram.org/file/bot" + cfg.Token,
	}, nil
}

// IsAllowed reports whether a user/chat pair passes the allowlist. Empty
// allowlists mean "allow anyone" (dev convenience); set them in production.
func (b *Bot) IsAllowed(userID, chatID int64) bool {
	return allowed(b.cfg.AllowedUserIDs, userID) && allowed(b.cfg.AllowedChatIDs, chatID)
}

func allowed(list []int64, id int64) bool {
	if len(list) == 0 {
		return true
	}
	for _, v := range list {
		if v == id {
			return true
		}
	}
	return false
}

// apiResponse is the standard Telegram envelope.
type apiResponse struct {
	OK          bool            `json:"ok"`
	Result      json.RawMessage `json:"result"`
	Description string          `json:"description"`
}

func (b *Bot) callJSON(ctx context.Context, method string, params url.Values) (json.RawMessage, error) {
	endpoint := b.apiBase + "/" + method
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewBufferString(params.Encode()))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	return b.execute(req, method)
}

func (b *Bot) execute(req *http.Request, method string) (json.RawMessage, error) {
	resp, err := b.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("telegram %s: %w", method, err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	var env apiResponse
	if err := json.Unmarshal(body, &env); err != nil {
		return nil, fmt.Errorf("telegram %s: bad response: %w", method, err)
	}
	if !env.OK {
		return nil, fmt.Errorf("telegram %s: %s", method, env.Description)
	}
	return env.Result, nil
}

// SendMessage sends a plain text message (used for notices).
func (b *Bot) SendMessage(ctx context.Context, chatID int64, text string) error {
	params := url.Values{}
	params.Set("chat_id", strconv.FormatInt(chatID, 10))
	params.Set("text", text)
	_, err := b.callJSON(ctx, "sendMessage", params)
	return err
}

// SendVoice uploads an OGG/Opus file as a voice note and returns its message id.
func (b *Bot) SendVoice(ctx context.Context, chatID int64, oggPath, caption string) (int64, error) {
	f, err := os.Open(oggPath)
	if err != nil {
		return 0, fmt.Errorf("telegram sendVoice: open %s: %w", oggPath, err)
	}
	defer f.Close()

	var body bytes.Buffer
	mw := multipart.NewWriter(&body)
	_ = mw.WriteField("chat_id", strconv.FormatInt(chatID, 10))
	if caption != "" {
		_ = mw.WriteField("caption", caption)
	}
	part, err := mw.CreateFormFile("voice", filepath.Base(oggPath))
	if err != nil {
		return 0, err
	}
	if _, err := io.Copy(part, f); err != nil {
		return 0, err
	}
	if err := mw.Close(); err != nil {
		return 0, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, b.apiBase+"/sendVoice", &body)
	if err != nil {
		return 0, err
	}
	req.Header.Set("Content-Type", mw.FormDataContentType())
	raw, err := b.execute(req, "sendVoice")
	if err != nil {
		return 0, err
	}
	var msg struct {
		MessageID int64 `json:"message_id"`
	}
	_ = json.Unmarshal(raw, &msg)
	return msg.MessageID, nil
}

// downloadFile resolves a file_id to a path and downloads it to dest.
func (b *Bot) downloadFile(ctx context.Context, fileID, dest string) error {
	params := url.Values{}
	params.Set("file_id", fileID)
	raw, err := b.callJSON(ctx, "getFile", params)
	if err != nil {
		return err
	}
	var file struct {
		FilePath string `json:"file_path"`
	}
	if err := json.Unmarshal(raw, &file); err != nil {
		return err
	}
	if file.FilePath == "" {
		return fmt.Errorf("telegram getFile: empty file_path")
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, b.fileAPI+"/"+file.FilePath, nil)
	if err != nil {
		return err
	}
	resp, err := b.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("telegram download: status %d", resp.StatusCode)
	}
	if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
		return err
	}
	out, err := os.Create(dest)
	if err != nil {
		return err
	}
	defer out.Close()
	_, err = io.Copy(out, resp.Body)
	return err
}
