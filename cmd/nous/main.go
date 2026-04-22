// Command nous is the Nous Go memory service entry point.
//
// Subcommands:
//
//	serve    — start the HTTP API server (default if no subcommand given)
//	migrate  — apply pending database migrations and exit
//	health   — connect to the configured PostgreSQL + embedder, print status, exit
//
// All configuration comes from environment variables (see internal/config).
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/ologos-repos/nous/internal/api"
	"github.com/ologos-repos/nous/internal/config"
	"github.com/ologos-repos/nous/internal/embeddings"
	"github.com/ologos-repos/nous/internal/store"
)

// version is intentionally a literal — replaced at build time via -ldflags
// when nicer release builds are needed. The default is fine for dev.
const version = "0.1.0-dev"

func main() {
	logger := log.New(os.Stderr, "nous ", log.LstdFlags|log.Lmsgprefix)

	args := os.Args[1:]
	subcmd := "serve"
	if len(args) > 0 && !isFlag(args[0]) {
		subcmd = args[0]
		args = args[1:]
	}

	switch subcmd {
	case "serve":
		runServe(args, logger)
	case "migrate":
		runMigrate(args, logger)
	case "health":
		runHealth(args, logger)
	case "version", "-version", "--version":
		fmt.Printf("nous %s\n", version)
	case "help", "-h", "--help":
		printUsage(os.Stdout)
	default:
		fmt.Fprintf(os.Stderr, "unknown subcommand %q\n\n", subcmd)
		printUsage(os.Stderr)
		os.Exit(2)
	}
}

func isFlag(s string) bool {
	return len(s) > 0 && s[0] == '-'
}

func printUsage(w *os.File) {
	fmt.Fprintln(w, "Usage: nous <subcommand> [flags]")
	fmt.Fprintln(w, "")
	fmt.Fprintln(w, "Subcommands:")
	fmt.Fprintln(w, "  serve    Start the HTTP API server (default)")
	fmt.Fprintln(w, "  migrate  Apply pending database migrations and exit")
	fmt.Fprintln(w, "  health   Print health status (DB ping + embedder check) and exit")
	fmt.Fprintln(w, "  version  Print the build version")
	fmt.Fprintln(w, "")
	fmt.Fprintln(w, "Configuration is read from environment variables; see internal/config.")
	fmt.Fprintln(w, "Required: NOUS_POSTGRES_URL")
}

// ===== serve =====

func runServe(args []string, logger *log.Logger) {
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	addrOverride := fs.String("addr", "", "override NOUS_ADDR (e.g. :7474)")
	if err := fs.Parse(args); err != nil {
		logger.Fatalf("parse serve flags: %v", err)
	}

	cfg := mustLoadConfig(logger)
	if *addrOverride != "" {
		cfg.Addr = *addrOverride
	}

	store := mustConnectStore(cfg, logger)
	defer store.Close()
	embedder := mustBuildEmbedder(cfg, logger)
	defer embedder.Close()

	server := api.NewServer(api.ServerConfig{
		Addr:            cfg.Addr,
		ReadTimeout:     cfg.ReadTimeout,
		WriteTimeout:    cfg.WriteTimeout,
		ShutdownTimeout: cfg.ShutdownTimeout,
	}, api.DefaultsFromConfig(cfg), store, embedder, logger)

	// Graceful shutdown on SIGINT/SIGTERM. Server.Start blocks until the
	// underlying http.Server returns, so we run it in a goroutine and wait
	// for either a signal or a fatal startup error.
	errCh := make(chan error, 1)
	go func() {
		if err := server.Start(); err != nil {
			errCh <- err
		}
	}()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	select {
	case err := <-errCh:
		logger.Fatalf("server failed: %v", err)
	case sig := <-sigCh:
		logger.Printf("received signal %s; shutting down", sig)
		ctx, cancel := context.WithTimeout(context.Background(), cfg.ShutdownTimeout+5*time.Second)
		defer cancel()
		if err := server.Shutdown(ctx); err != nil {
			logger.Printf("graceful shutdown error: %v", err)
		}
	}
}

// ===== migrate =====

func runMigrate(args []string, logger *log.Logger) {
	fs := flag.NewFlagSet("migrate", flag.ExitOnError)
	if err := fs.Parse(args); err != nil {
		logger.Fatalf("parse migrate flags: %v", err)
	}

	cfg := mustLoadConfig(logger)
	cfg.RunMigrations = true // safety: even if NOUS_RUN_MIGRATIONS=false
	st := mustConnectStore(cfg, logger)
	defer st.Close()

	logger.Printf("migrate: complete (database is up to date)")
}

// ===== health =====

type healthReport struct {
	Status            string `json:"status"`
	PostgresReachable bool   `json:"postgres_reachable"`
	PostgresError     string `json:"postgres_error,omitempty"`
	EmbedderModel     string `json:"embedder_model"`
	EmbedderAvailable bool   `json:"embedder_available"`
	PoolConns         int32  `json:"pool_conns,omitempty"`
}

func runHealth(args []string, logger *log.Logger) {
	fs := flag.NewFlagSet("health", flag.ExitOnError)
	pretty := fs.Bool("pretty", true, "pretty-print JSON output")
	if err := fs.Parse(args); err != nil {
		logger.Fatalf("parse health flags: %v", err)
	}

	cfg, err := config.Load()
	if err != nil {
		report := healthReport{
			Status:        "config_error",
			PostgresError: err.Error(),
		}
		emit(report, *pretty)
		os.Exit(1)
	}

	report := healthReport{
		Status:        "ok",
		EmbedderModel: cfg.OllamaModel,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	st, err := store.Connect(ctx, store.Config{
		PostgresURL:   cfg.PostgresURL,
		ShellDir:      cfg.ShellDir,
		Embedder:      &embeddings.NullEmbedder{}, // health doesn't need a real embedder
		RunMigrations: false,
	})
	if err != nil {
		report.Status = "degraded"
		report.PostgresReachable = false
		report.PostgresError = err.Error()
		emit(report, *pretty)
		os.Exit(1)
	}
	defer st.Close()

	if err := st.Ping(ctx); err != nil {
		report.Status = "degraded"
		report.PostgresReachable = false
		report.PostgresError = err.Error()
	} else {
		report.PostgresReachable = true
		if pool := st.Pool(); pool != nil {
			report.PoolConns = pool.Stat().TotalConns()
		}
	}

	embedder := mustBuildEmbedder(cfg, logger)
	defer embedder.Close()
	report.EmbedderModel = embedder.ModelName()
	report.EmbedderAvailable = embedder.IsAvailable(ctx)
	if !report.EmbedderAvailable && report.Status == "ok" {
		report.Status = "degraded"
	}

	emit(report, *pretty)
	if report.Status != "ok" {
		os.Exit(1)
	}
}

func emit(r healthReport, pretty bool) {
	enc := json.NewEncoder(os.Stdout)
	if pretty {
		enc.SetIndent("", "  ")
	}
	_ = enc.Encode(r)
}

// ===== shared bootstrap helpers =====

func mustLoadConfig(logger *log.Logger) config.Config {
	cfg, err := config.Load()
	if err != nil {
		logger.Fatalf("config: %v", err)
	}
	return cfg
}

func mustConnectStore(cfg config.Config, logger *log.Logger) *store.MemoryStore {
	embedder := mustBuildEmbedder(cfg, logger)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	st, err := store.Connect(ctx, store.Config{
		PostgresURL:   cfg.PostgresURL,
		ShellDir:      cfg.ShellDir,
		Embedder:      embedder,
		RunMigrations: cfg.RunMigrations,
	})
	if err != nil {
		// Config-level error vs DB error are both fatal here.
		var connErr *connErrorWrapper
		if errors.As(err, &connErr) {
			logger.Fatalf("store connect: %v", err)
		}
		logger.Fatalf("store connect: %v", err)
	}
	return st
}

// connErrorWrapper exists only to satisfy errors.As patterns in the future
// without forcing an extra dependency on the storage package's internal
// error types.
type connErrorWrapper struct{ err error }

func (c *connErrorWrapper) Error() string { return c.err.Error() }
func (c *connErrorWrapper) Unwrap() error { return c.err }

func mustBuildEmbedder(cfg config.Config, logger *log.Logger) embeddings.EmbeddingProvider {
	switch cfg.EmbedProvider {
	case "ollama":
		return embeddings.NewOllamaEmbedder(cfg.OllamaModel, cfg.OllamaURL, cfg.EmbedDimensions, cfg.EmbedTimeout)
	case "null":
		return &embeddings.NullEmbedder{}
	default:
		logger.Fatalf("unsupported embedder provider %q", cfg.EmbedProvider)
		return nil // unreachable
	}
}
