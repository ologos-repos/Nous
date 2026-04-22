package api

import (
	"context"
	"errors"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/ologos-repos/nous/internal/config"
	"github.com/ologos-repos/nous/internal/embeddings"
)

// ServerConfig is the per-listener config block used by Server. It mirrors the
// values in config.Config but is decoupled so the api package doesn't depend
// on the full server config when a caller wants a custom subset (e.g. tests).
type ServerConfig struct {
	Addr            string
	ReadTimeout     time.Duration
	WriteTimeout    time.Duration
	ShutdownTimeout time.Duration
}

// Server wraps an http.Server with all the dependencies the handlers need.
// One Server per process. Use NewServer to construct it; call Start to listen
// and Shutdown to drain gracefully.
type Server struct {
	cfg      ServerConfig
	defaults Defaults
	store    HandlerStore
	embedder embeddings.EmbeddingProvider
	logger   *log.Logger
	srv      *http.Server
}

// Defaults holds the dendritic recall parameter defaults that handlers apply
// when a request leaves a field zero. Sourced from config.Config.
type Defaults struct {
	TopicK              int
	ActivationThreshold float64
	MemoryK             int
	Threshold           float64
	Hops                int
	GraphDiscount       float64
	Limit               int
}

// DefaultsFromConfig copies the recall defaults out of a loaded config.Config.
func DefaultsFromConfig(c config.Config) Defaults {
	return Defaults{
		TopicK:              c.DefaultTopicK,
		ActivationThreshold: c.DefaultActivationThreshold,
		MemoryK:             c.DefaultMemoryK,
		Threshold:           c.DefaultThreshold,
		Hops:                c.DefaultHops,
		GraphDiscount:       c.DefaultGraphDiscount,
		Limit:               c.DefaultLimit,
	}
}

// NewServer wires the dependencies into a Server. logger may be nil — the
// default *log.Logger is used in that case.
func NewServer(cfg ServerConfig, defaults Defaults, store HandlerStore, embedder embeddings.EmbeddingProvider, logger *log.Logger) *Server {
	if logger == nil {
		logger = log.Default()
	}
	if cfg.Addr == "" {
		cfg.Addr = "localhost:7474"
	}
	if cfg.ReadTimeout == 0 {
		cfg.ReadTimeout = 30 * time.Second
	}
	if cfg.WriteTimeout == 0 {
		cfg.WriteTimeout = 60 * time.Second
	}
	if cfg.ShutdownTimeout == 0 {
		cfg.ShutdownTimeout = 10 * time.Second
	}
	return &Server{
		cfg:      cfg,
		defaults: defaults,
		store:    store,
		embedder: embedder,
		logger:   logger,
	}
}

// Handler builds the http.Handler with all routes and middleware applied.
// Exposed separately from Start so tests can hit it via httptest.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	s.registerRoutes(mux)
	return chain(mux,
		recoverMiddleware(s.logger),
		loggingMiddleware(s.logger),
		jsonContentType,
	)
}

// Start binds the address and serves until the underlying server returns. It
// blocks. Use Shutdown to stop it.
func (s *Server) Start() error {
	s.srv = &http.Server{
		Addr:         s.cfg.Addr,
		Handler:      s.Handler(),
		ReadTimeout:  s.cfg.ReadTimeout,
		WriteTimeout: s.cfg.WriteTimeout,
	}
	s.logger.Printf("nous: listening on %s", s.cfg.Addr)
	err := s.srv.ListenAndServe()
	if err != nil && !errors.Is(err, http.ErrServerClosed) {
		return fmt.Errorf("listen %s: %w", s.cfg.Addr, err)
	}
	return nil
}

// Shutdown drains in-flight requests with the configured ShutdownTimeout.
// Safe to call before Start (no-op).
func (s *Server) Shutdown(ctx context.Context) error {
	if s.srv == nil {
		return nil
	}
	shutdownCtx, cancel := context.WithTimeout(ctx, s.cfg.ShutdownTimeout)
	defer cancel()
	s.logger.Printf("nous: shutting down (timeout=%s)", s.cfg.ShutdownTimeout)
	return s.srv.Shutdown(shutdownCtx)
}
