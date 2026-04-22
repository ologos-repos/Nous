// Package embeddings defines the EmbeddingProvider interface and its
// implementations.
//
// The package is deliberately small. Implementations are free to talk to
// local daemons (Ollama), remote APIs, or act as a no-op. Everything in
// the store and recall layers depends only on the interface — swapping
// providers is a configuration change, not a code change.
package embeddings

import "context"

// EmbeddingProvider is the interface all embedding backends must satisfy.
//
// Implementations must be safe for concurrent use. Callers can and will
// invoke Embed/EmbedBatch from multiple goroutines (e.g. when a topic's
// representative set is being recomputed while a recall request is in
// flight).
//
// Methods are expected to honour ctx cancellation — long-running HTTP
// calls should pass ctx into their transports, and batch implementations
// should check ctx.Err() between items when falling back to sequential
// embedding.
type EmbeddingProvider interface {
	// Embed generates an embedding vector for a single text.
	// Returns (nil, err) on failure, or (nil, nil) for a no-op provider.
	Embed(ctx context.Context, text string) ([]float32, error)

	// EmbedBatch generates embeddings for multiple texts.
	// Implementations should use batch API calls when the underlying
	// service supports them (e.g. Ollama /api/embed with input:[]string).
	// The returned slice has len(texts) entries, index-aligned with the
	// input; nil entries indicate a per-item failure when the
	// implementation chooses to tolerate partial results.
	EmbedBatch(ctx context.Context, texts []string) ([][]float32, error)

	// Dimensions returns the number of dimensions in the output vectors.
	// Returns 0 for no-op providers (NullEmbedder).
	Dimensions() int

	// ModelName returns a stable identifier for the embedding model.
	// Used for provenance (stored alongside each embedding row).
	ModelName() string

	// IsAvailable checks if the embedding service is reachable and the
	// configured model is loadable. Intended for health checks; safe to
	// call on a hot path but implementations may cache a short TTL.
	IsAvailable(ctx context.Context) bool

	// Close releases any held resources (HTTP clients, connection pools).
	// Safe to call multiple times; subsequent calls return the same
	// result as the first.
	Close() error
}
