// Package config loads Nous Go server configuration from environment
// variables with the defaults specified in the design spec (section 11).
//
// All values are read once via Load() at startup. The returned Config is the
// authoritative server config — every subsystem (api.Server, store.Connect,
// embeddings.NewOllamaEmbedder, recall.DendriticRecall) reads from it.
package config

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// Config bundles all environment-driven server settings.
type Config struct {
	// Server
	Addr string

	// Database
	PostgresURL string
	ShellDir    string

	// Embeddings
	EmbedProvider   string // "ollama" | "null"
	OllamaModel     string
	OllamaURL       string
	EmbedDimensions int
	EmbedTimeout    time.Duration

	// Dendritic recall defaults — applied when a recall request leaves
	// fields zero. Per spec §11.
	DefaultTopicK              int
	DefaultActivationThreshold float64
	DefaultMemoryK             int
	DefaultThreshold           float64
	DefaultHops                int
	DefaultGraphDiscount       float64
	DefaultLimit               int

	// Migrations
	RunMigrations bool

	// HTTP server timeouts
	ReadTimeout     time.Duration
	WriteTimeout    time.Duration
	ShutdownTimeout time.Duration
}

// Defaults are the fallback values used when an env var is unset or empty.
// These mirror the values listed in spec §11.
var Defaults = Config{
	Addr:                       "localhost:7474",
	PostgresURL:                "",
	ShellDir:                   "", // resolved to $HOME/.nous/shells in Load
	EmbedProvider:              "ollama",
	OllamaModel:                "nomic-embed-text",
	OllamaURL:                  "http://localhost:11434",
	EmbedDimensions:            768,
	EmbedTimeout:               30 * time.Second,
	DefaultTopicK:              5,
	DefaultActivationThreshold: 0.3,
	DefaultMemoryK:             10,
	DefaultThreshold:           0.3,
	DefaultHops:                2,
	DefaultGraphDiscount:       0.6,
	DefaultLimit:               20,
	RunMigrations:              true,
	ReadTimeout:                30 * time.Second,
	WriteTimeout:               60 * time.Second,
	ShutdownTimeout:            10 * time.Second,
}

// Load reads configuration from process environment variables and applies
// defaults for anything unset. NOUS_POSTGRES_URL is REQUIRED — Load returns an
// error if it is missing or empty.
func Load() (Config, error) {
	c := Defaults

	c.Addr = envString("NOUS_ADDR", c.Addr)
	c.PostgresURL = envString("NOUS_POSTGRES_URL", c.PostgresURL)
	c.ShellDir = envString("NOUS_SHELL_DIR", c.ShellDir)
	if c.ShellDir == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return c, fmt.Errorf("resolve home dir for default shell dir: %w", err)
		}
		c.ShellDir = filepath.Join(home, ".nous", "shells")
	}

	c.EmbedProvider = strings.ToLower(envString("NOUS_EMBED_PROVIDER", c.EmbedProvider))
	c.OllamaModel = envString("NOUS_OLLAMA_MODEL", c.OllamaModel)
	c.OllamaURL = envString("NOUS_OLLAMA_URL", c.OllamaURL)
	c.EmbedDimensions = envInt("NOUS_EMBED_DIMENSIONS", c.EmbedDimensions)
	c.EmbedTimeout = envDuration("NOUS_EMBED_TIMEOUT", c.EmbedTimeout)

	c.DefaultTopicK = envInt("NOUS_DEFAULT_TOPIC_K", c.DefaultTopicK)
	c.DefaultActivationThreshold = envFloat("NOUS_DEFAULT_ACTIVATION_THRESHOLD", c.DefaultActivationThreshold)
	c.DefaultMemoryK = envInt("NOUS_DEFAULT_MEMORY_K", c.DefaultMemoryK)
	c.DefaultThreshold = envFloat("NOUS_DEFAULT_THRESHOLD", c.DefaultThreshold)
	c.DefaultHops = envInt("NOUS_DEFAULT_HOPS", c.DefaultHops)
	c.DefaultGraphDiscount = envFloat("NOUS_DEFAULT_GRAPH_DISCOUNT", c.DefaultGraphDiscount)
	c.DefaultLimit = envInt("NOUS_DEFAULT_LIMIT", c.DefaultLimit)

	c.RunMigrations = envBool("NOUS_RUN_MIGRATIONS", c.RunMigrations)

	c.ReadTimeout = envDuration("NOUS_READ_TIMEOUT", c.ReadTimeout)
	c.WriteTimeout = envDuration("NOUS_WRITE_TIMEOUT", c.WriteTimeout)
	c.ShutdownTimeout = envDuration("NOUS_SHUTDOWN_TIMEOUT", c.ShutdownTimeout)

	if c.PostgresURL == "" {
		return c, fmt.Errorf("NOUS_POSTGRES_URL is required (set the postgres connection string)")
	}
	switch c.EmbedProvider {
	case "ollama", "null":
		// ok
	default:
		return c, fmt.Errorf("unsupported NOUS_EMBED_PROVIDER %q (expected 'ollama' or 'null')", c.EmbedProvider)
	}

	return c, nil
}

// ===== env helpers =====

func envString(key, def string) string {
	v, ok := os.LookupEnv(key)
	if !ok || v == "" {
		return def
	}
	return v
}

func envInt(key string, def int) int {
	v, ok := os.LookupEnv(key)
	if !ok || v == "" {
		return def
	}
	parsed, err := strconv.Atoi(strings.TrimSpace(v))
	if err != nil {
		return def
	}
	return parsed
}

func envFloat(key string, def float64) float64 {
	v, ok := os.LookupEnv(key)
	if !ok || v == "" {
		return def
	}
	parsed, err := strconv.ParseFloat(strings.TrimSpace(v), 64)
	if err != nil {
		return def
	}
	return parsed
}

func envBool(key string, def bool) bool {
	v, ok := os.LookupEnv(key)
	if !ok || v == "" {
		return def
	}
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "1", "true", "t", "yes", "y", "on":
		return true
	case "0", "false", "f", "no", "n", "off":
		return false
	default:
		return def
	}
}

// envDuration accepts either a Go duration string ("30s", "5m") or a bare
// number interpreted as seconds (so legacy NOUS_EMBED_TIMEOUT=30 still works).
func envDuration(key string, def time.Duration) time.Duration {
	v, ok := os.LookupEnv(key)
	if !ok || v == "" {
		return def
	}
	v = strings.TrimSpace(v)
	if d, err := time.ParseDuration(v); err == nil {
		return d
	}
	if secs, err := strconv.Atoi(v); err == nil {
		return time.Duration(secs) * time.Second
	}
	return def
}
