// Command quietvoice is the QuietVoice orchestrator (control plane): it serves
// the MCP endpoint, runs the Telegram transport, and keeps voice state. GPU work
// is delegated to an inference.Engine (local monolith or remote node) chosen by
// config. There is no local microphone or speaker — the user talks to their
// coding agent through Telegram voice messages.
package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"quietvoice/internal/config"
	"quietvoice/internal/engine"
	"quietvoice/internal/inference"
	"quietvoice/internal/mcp"
	"quietvoice/internal/store"
	"quietvoice/internal/telegram"
	"quietvoice/internal/voice"
)

func main() {
	log.SetOutput(os.Stdout)
	cfg := config.Load()

	if cfg.TelegramToken == "" {
		log.Fatal("TELEGRAM_BOT_TOKEN is required")
	}
	if cfg.DefaultChatID == 0 && len(cfg.AllowedChatIDs) == 1 {
		cfg.DefaultChatID = cfg.AllowedChatIDs[0]
	}
	if cfg.DefaultChatID == 0 {
		log.Fatal("TELEGRAM_DEFAULT_CHAT_ID is required (the chat that receives spoken updates)")
	}

	eng, err := engine.Build(cfg)
	if err != nil {
		log.Fatalf("inference: %v", err)
	}
	log.Printf("inference plane: mode=%s", cfg.InferenceMode)

	st, err := store.Open(cfg.StorePath)
	if err != nil {
		log.Fatalf("store: %v", err)
	}
	defer st.Close()

	bot, err := telegram.New(telegram.Config{
		Token:          cfg.TelegramToken,
		AllowedUserIDs: cfg.AllowedUserIDs,
		AllowedChatIDs: cfg.AllowedChatIDs,
		DownloadDir:    cfg.DownloadDir,
	})
	if err != nil {
		log.Fatalf("telegram: %v", err)
	}

	listenMode := inference.ParseListenMode(cfg.ListenMode)
	svc := voice.New(voice.Config{
		DefaultChatID:   cfg.DefaultChatID,
		WaitTimeout:     cfg.WaitTimeout,
		SayContextCount: cfg.SayContextCount,
		WorkDir:         cfg.WorkDir,
		ListenMode:      listenMode,
		EvalLogPath:     cfg.EvalLogPath,
	}, st, eng, bot)
	log.Printf("listen mode: %s", listenMode)

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	// Telegram poll loop: downloaded voice notes flow into the voice service.
	go func() {
		err := bot.Poll(ctx, func(pollCtx context.Context, in telegram.IncomingVoice) {
			svc.HandleIncomingVoice(pollCtx, voice.IncomingVoice{
				ChatID:    in.ChatID,
				UserID:    in.UserID,
				MessageID: in.MessageID,
				FileID:    in.FileID,
				LocalPath: in.LocalPath,
				Duration:  in.Duration,
			})
		})
		if err != nil && ctx.Err() == nil {
			log.Fatalf("telegram poll: %v", err)
		}
	}()

	mux := http.NewServeMux()
	mcp.New(svc, cfg.MCPToken).Routes(mux)
	srv := &http.Server{
		Addr:         cfg.MCPListen,
		Handler:      mux,
		ReadTimeout:  15 * time.Minute,
		WriteTimeout: 15 * time.Minute,
	}

	go func() {
		<-ctx.Done()
		shutCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = srv.Shutdown(shutCtx)
	}()

	log.Printf("QuietVoice MCP listening on %s (POST /rpc), chat=%d", cfg.MCPListen, cfg.DefaultChatID)
	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("http server: %v", err)
	}
	log.Printf("QuietVoice stopped")
}
