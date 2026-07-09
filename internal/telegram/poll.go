package telegram

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/url"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// Minimal subset of the Telegram update model.
type update struct {
	UpdateID int64    `json:"update_id"`
	Message  *message `json:"message"`
}

type message struct {
	MessageID int64  `json:"message_id"`
	From      *user  `json:"from"`
	Chat      *chat  `json:"chat"`
	Text      string `json:"text"`
	Voice     *voice `json:"voice"`
	Audio     *voice `json:"audio"`
	Document  *voice `json:"document"`
}

type user struct {
	ID int64 `json:"id"`
}

type chat struct {
	ID int64 `json:"id"`
}

type voice struct {
	FileID   string `json:"file_id"`
	Duration int    `json:"duration"`
	MimeType string `json:"mime_type"`
}

// Poll runs the long-polling loop until ctx is cancelled. Voice notes that pass
// the allowlist are downloaded and passed to handler; unknown senders are
// ignored. /start replies with a short hello.
func (b *Bot) Poll(ctx context.Context, handler VoiceHandler) error {
	var offset int64
	for {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		updates, err := b.getUpdates(ctx, offset)
		if err != nil {
			if ctx.Err() != nil {
				return ctx.Err()
			}
			log.Printf("telegram poll: %v; retrying in 3s", err)
			select {
			case <-time.After(3 * time.Second):
			case <-ctx.Done():
				return ctx.Err()
			}
			continue
		}
		for _, u := range updates {
			offset = u.UpdateID + 1
			b.dispatch(ctx, u, handler)
		}
	}
}

func (b *Bot) getUpdates(ctx context.Context, offset int64) ([]update, error) {
	params := url.Values{}
	params.Set("offset", strconv.FormatInt(offset, 10))
	params.Set("timeout", "30")
	params.Set("allowed_updates", `["message"]`)
	raw, err := b.callJSON(ctx, "getUpdates", params)
	if err != nil {
		return nil, err
	}
	var updates []update
	if err := json.Unmarshal(raw, &updates); err != nil {
		return nil, fmt.Errorf("getUpdates decode: %w", err)
	}
	return updates, nil
}

func (b *Bot) dispatch(ctx context.Context, u update, handler VoiceHandler) {
	msg := u.Message
	if msg == nil || msg.Chat == nil {
		return
	}
	var userID int64
	if msg.From != nil {
		userID = msg.From.ID
	}
	if !b.IsAllowed(userID, msg.Chat.ID) {
		log.Printf("telegram: ignoring message from unauthorized user=%d chat=%d", userID, msg.Chat.ID)
		return
	}

	v := msg.Voice
	if v == nil {
		v = msg.Audio
	}
	// Telegram Desktop drag-and-drop delivers an audio clip as a document, not a
	// voice note (there is no client way to attach a synthetic clip "as a voice").
	// Accept audio documents so pre-recorded .ogg replies still reach listen_voice.
	// Telegram tags .ogg as audio/ogg or application/ogg depending on the client,
	// so match audio/* plus explicit ogg/opus rather than a single prefix.
	if v == nil && msg.Document != nil {
		if mt := msg.Document.MimeType; strings.HasPrefix(mt, "audio/") ||
			strings.Contains(mt, "ogg") || strings.Contains(mt, "opus") {
			v = msg.Document
		}
	}
	if v == nil {
		if msg.Text == "/start" {
			_ = b.SendMessage(ctx, msg.Chat.ID, "QuietVoice connected. Send a voice message any time; I'll route it to your coding agent.")
		}
		return
	}

	dest := filepath.Join(b.cfg.DownloadDir, fmt.Sprintf("in_%d_%d.ogg", msg.Chat.ID, msg.MessageID))
	if err := b.downloadFile(ctx, v.FileID, dest); err != nil {
		log.Printf("telegram: failed to download voice %s: %v", v.FileID, err)
		return
	}
	handler(ctx, IncomingVoice{
		ChatID:    msg.Chat.ID,
		UserID:    userID,
		MessageID: msg.MessageID,
		FileID:    v.FileID,
		LocalPath: dest,
		Duration:  v.Duration,
	})
}
