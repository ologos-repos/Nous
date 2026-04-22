package embeddings

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// Default configuration for OllamaEmbedder. These match the spec (§4.1).
const (
	DefaultOllamaModel      = "nomic-embed-text"
	DefaultOllamaBaseURL    = "http://localhost:11434"
	DefaultOllamaDimensions = 768
	DefaultOllamaTimeout    = 30 * time.Second
)

// OllamaEmbedder implements EmbeddingProvider by calling a local Ollama
// daemon's /api/embed endpoint (Ollama 0.3+).
//
// The endpoint accepts both single strings and arrays via the "input"
// field and returns a uniform response shape:
//
//	{ "model": "...", "embeddings": [[...], [...]] }
//
// Older Ollama releases exposed /api/embeddings with a "prompt" field
// and no batch support — this implementation does NOT fall back to it,
// per the spec. Operators on older Ollama should upgrade.
//
// The struct is safe for concurrent use; *http.Client itself is
// goroutine-safe and the other fields are read-only after construction.
type OllamaEmbedder struct {
	model      string
	baseURL    string
	dimensions int
	timeout    time.Duration
	client     *http.Client
}

// NewOllamaEmbedder creates an embedder pointing at an Ollama instance.
//
// Empty or zero values fall back to sensible defaults (DefaultOllamaModel,
// DefaultOllamaBaseURL, DefaultOllamaDimensions, DefaultOllamaTimeout).
// Trailing slashes in baseURL are stripped so callers can pass either
// "http://host:port" or "http://host:port/".
func NewOllamaEmbedder(model, baseURL string, dimensions int, timeout time.Duration) *OllamaEmbedder {
	if model == "" {
		model = DefaultOllamaModel
	}
	if baseURL == "" {
		baseURL = DefaultOllamaBaseURL
	}
	baseURL = strings.TrimRight(baseURL, "/")
	if dimensions <= 0 {
		dimensions = DefaultOllamaDimensions
	}
	if timeout <= 0 {
		timeout = DefaultOllamaTimeout
	}
	return &OllamaEmbedder{
		model:      model,
		baseURL:    baseURL,
		dimensions: dimensions,
		timeout:    timeout,
		client: &http.Client{
			Timeout: timeout,
		},
	}
}

// ollamaEmbedRequest is the JSON body posted to /api/embed.
//
// Input can be either a string or a []string — Ollama handles both
// shapes. We preserve that flexibility here by typing it as `any`.
type ollamaEmbedRequest struct {
	Model string `json:"model"`
	Input any    `json:"input"`
}

// ollamaEmbedResponse is the JSON body returned by /api/embed.
// Embeddings is always a list of vectors even for single-string input
// (a one-element list in that case).
type ollamaEmbedResponse struct {
	Model      string      `json:"model"`
	Embeddings [][]float32 `json:"embeddings"`
}

// Embed calls POST /api/embed with input set to a single string.
//
// Returns the first embedding vector from the response. Returns (nil, err)
// on HTTP failure, decode failure, or empty-response. Does not return
// (nil, nil) — a successful call always yields a vector.
func (e *OllamaEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	resp, err := e.postEmbed(ctx, ollamaEmbedRequest{Model: e.model, Input: text})
	if err != nil {
		return nil, err
	}
	if len(resp.Embeddings) == 0 {
		return nil, fmt.Errorf("ollama: empty embeddings in response")
	}
	return resp.Embeddings[0], nil
}

// EmbedBatch calls POST /api/embed with input set to the full slice.
//
// If the batch call fails, falls back to sequential Embed calls so the
// caller receives a best-effort per-item result. A per-item failure
// during fallback is returned as an error (no partial results), which
// matches the store layer's expectation of all-or-nothing batch semantics.
func (e *OllamaEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	resp, err := e.postEmbed(ctx, ollamaEmbedRequest{Model: e.model, Input: texts})
	if err == nil && len(resp.Embeddings) == len(texts) {
		return resp.Embeddings, nil
	}

	// Fallback: sequential single-string calls. This path is hit when the
	// daemon rejects the batch shape (old Ollama versions) or returns a
	// mismatched count.
	out := make([][]float32, len(texts))
	for i, t := range texts {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		v, embedErr := e.Embed(ctx, t)
		if embedErr != nil {
			return nil, fmt.Errorf("ollama: batch fallback item %d: %w", i, embedErr)
		}
		out[i] = v
	}
	return out, nil
}

// Dimensions returns the configured output dimensionality.
func (e *OllamaEmbedder) Dimensions() int { return e.dimensions }

// ModelName returns the configured model identifier.
func (e *OllamaEmbedder) ModelName() string { return e.model }

// IsAvailable hits GET /api/tags and verifies the configured model is
// listed. Returns false on any HTTP error, decode failure, or if the
// model is absent from the daemon.
//
// The check is intentionally lightweight — it does not run a test
// embedding, just verifies the daemon is up and the model is loaded.
func (e *OllamaEmbedder) IsAvailable(ctx context.Context) bool {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, e.baseURL+"/api/tags", nil)
	if err != nil {
		return false
	}
	resp, err := e.client.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return false
	}
	var body struct {
		Models []struct {
			Name  string `json:"name"`
			Model string `json:"model"`
		} `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return false
	}
	// Ollama lists models under either "name" or "model"; accept either
	// field matching with or without the ":latest" tag suffix, since
	// callers typically configure bare model names.
	for _, m := range body.Models {
		if matchModelName(m.Name, e.model) || matchModelName(m.Model, e.model) {
			return true
		}
	}
	return false
}

// Close releases the underlying transport's idle connections.
// Returns nil — http.Client.CloseIdleConnections has no error channel.
func (e *OllamaEmbedder) Close() error {
	if e.client != nil {
		e.client.CloseIdleConnections()
	}
	return nil
}

// matchModelName reports whether listed == configured, ignoring a
// trailing ":latest" tag on the listed name (the default Ollama adds to
// newly pulled models).
func matchModelName(listed, configured string) bool {
	if listed == configured {
		return true
	}
	if strings.TrimSuffix(listed, ":latest") == configured {
		return true
	}
	return false
}

// postEmbed is the shared HTTP workhorse used by Embed and EmbedBatch.
// It marshals the request, posts it, and decodes the response — making
// the per-verb methods above trivial wrappers.
func (e *OllamaEmbedder) postEmbed(ctx context.Context, body ollamaEmbedRequest) (*ollamaEmbedResponse, error) {
	buf, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("ollama: marshal request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.baseURL+"/api/embed", bytes.NewReader(buf))
	if err != nil {
		return nil, fmt.Errorf("ollama: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("ollama: post /api/embed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		// Pull the body for diagnostic context (capped to keep errors bounded).
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, fmt.Errorf("ollama: status %d: %s", resp.StatusCode, strings.TrimSpace(string(raw)))
	}

	var out ollamaEmbedResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("ollama: decode response: %w", err)
	}
	return &out, nil
}

// Ensure the interface contract is satisfied at compile time — this is
// a zero-cost guarantee and catches accidental interface drift during
// refactors.
var _ EmbeddingProvider = (*OllamaEmbedder)(nil)
var _ EmbeddingProvider = (*NullEmbedder)(nil)
