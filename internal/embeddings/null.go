package embeddings

import "context"

// NullEmbedder is a no-op EmbeddingProvider.
//
// It is the graceful-degradation path: when an operator does not want to
// run Ollama locally, or when the embedding service is temporarily
// unreachable, NullEmbedder lets the store layer continue serving
// keyword-only search while topic routing and semantic recall simply
// short-circuit to no-op.
//
// All methods are cheap and allocation-free; the embedder is safe to use
// as a singleton (see DefaultNullEmbedder) or to instantiate freely.
type NullEmbedder struct{}

// DefaultNullEmbedder is a convenience singleton for callers that want a
// ready-to-use NullEmbedder without constructing one themselves.
var DefaultNullEmbedder = &NullEmbedder{}

// NewNullEmbedder returns a new NullEmbedder. The zero value works too;
// this constructor exists for symmetry with the Ollama builder and to
// keep the spec's nine-method interface uniform across providers.
func NewNullEmbedder() *NullEmbedder {
	return &NullEmbedder{}
}

// Embed returns (nil, nil) for any input. Callers must tolerate a nil
// vector from Embed — this is the signal that semantic routing is
// disabled and they should fall back to keyword search.
func (n *NullEmbedder) Embed(_ context.Context, _ string) ([]float32, error) {
	return nil, nil
}

// EmbedBatch returns a len(texts) slice of nil vectors.
// The returned slice is never nil (even for an empty input) to keep
// callers' indexing code simple.
func (n *NullEmbedder) EmbedBatch(_ context.Context, texts []string) ([][]float32, error) {
	return make([][]float32, len(texts)), nil
}

// Dimensions returns 0 — NullEmbedder produces no vectors.
func (n *NullEmbedder) Dimensions() int { return 0 }

// ModelName returns a stable identifier usable in provenance columns.
func (n *NullEmbedder) ModelName() string { return "null" }

// IsAvailable always returns true. NullEmbedder has nothing to fail.
func (n *NullEmbedder) IsAvailable(_ context.Context) bool { return true }

// Close is a no-op; NullEmbedder owns no resources.
func (n *NullEmbedder) Close() error { return nil }
