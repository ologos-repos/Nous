package embeddings

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestNullEmbedderInterface(t *testing.T) {
	var e EmbeddingProvider = NewNullEmbedder()
	ctx := context.Background()

	v, err := e.Embed(ctx, "hello")
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if v != nil {
		t.Errorf("NullEmbedder.Embed = %v, want nil", v)
	}

	batch, err := e.EmbedBatch(ctx, []string{"a", "b", "c"})
	if err != nil {
		t.Fatalf("EmbedBatch: %v", err)
	}
	if len(batch) != 3 {
		t.Errorf("EmbedBatch len = %d, want 3", len(batch))
	}
	for i, v := range batch {
		if v != nil {
			t.Errorf("batch[%d] = %v, want nil", i, v)
		}
	}

	if e.Dimensions() != 0 {
		t.Errorf("Dimensions = %d, want 0", e.Dimensions())
	}
	if e.ModelName() != "null" {
		t.Errorf("ModelName = %q, want \"null\"", e.ModelName())
	}
	if !e.IsAvailable(ctx) {
		t.Error("IsAvailable = false, want true (NullEmbedder is always up)")
	}
	if err := e.Close(); err != nil {
		t.Errorf("Close returned %v, want nil", err)
	}
}

func TestOllamaEmbedderDefaults(t *testing.T) {
	// Empty constructor arguments should all pick up defaults.
	e := NewOllamaEmbedder("", "", 0, 0)
	if e.ModelName() != DefaultOllamaModel {
		t.Errorf("ModelName = %q, want %q", e.ModelName(), DefaultOllamaModel)
	}
	if e.Dimensions() != DefaultOllamaDimensions {
		t.Errorf("Dimensions = %d, want %d", e.Dimensions(), DefaultOllamaDimensions)
	}
	if e.baseURL != DefaultOllamaBaseURL {
		t.Errorf("baseURL = %q, want %q", e.baseURL, DefaultOllamaBaseURL)
	}
	if e.timeout != DefaultOllamaTimeout {
		t.Errorf("timeout = %v, want %v", e.timeout, DefaultOllamaTimeout)
	}
}

func TestOllamaEmbedderTrimsTrailingSlash(t *testing.T) {
	e := NewOllamaEmbedder("", "http://example.com:11434/", 0, 0)
	if strings.HasSuffix(e.baseURL, "/") {
		t.Errorf("baseURL has trailing slash: %q", e.baseURL)
	}
}

// fakeOllama spins up a test server that mimics /api/embed and /api/tags.
func fakeOllama(t *testing.T, handler func(w http.ResponseWriter, r *http.Request)) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(handler))
}

func TestOllamaEmbedSingle(t *testing.T) {
	ts := fakeOllama(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/embed" {
			t.Errorf("unexpected path %q", r.URL.Path)
		}
		if r.Method != http.MethodPost {
			t.Errorf("unexpected method %q", r.Method)
		}
		var req ollamaEmbedRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode: %v", err)
		}
		if req.Model != "nomic-embed-text" {
			t.Errorf("model = %q", req.Model)
		}
		if s, ok := req.Input.(string); !ok || s != "hello" {
			t.Errorf("Input = %v (%T), want \"hello\"", req.Input, req.Input)
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(ollamaEmbedResponse{
			Model:      "nomic-embed-text",
			Embeddings: [][]float32{{0.1, 0.2, 0.3}},
		})
	})
	defer ts.Close()

	e := NewOllamaEmbedder("nomic-embed-text", ts.URL, 768, time.Second)
	v, err := e.Embed(context.Background(), "hello")
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	want := []float32{0.1, 0.2, 0.3}
	if len(v) != len(want) {
		t.Fatalf("len = %d, want %d", len(v), len(want))
	}
	for i := range want {
		if v[i] != want[i] {
			t.Errorf("v[%d] = %v, want %v", i, v[i], want[i])
		}
	}
}

func TestOllamaEmbedBatch(t *testing.T) {
	ts := fakeOllama(t, func(w http.ResponseWriter, r *http.Request) {
		var req ollamaEmbedRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode: %v", err)
		}
		arr, ok := req.Input.([]any)
		if !ok {
			t.Fatalf("Input = %T, want []any", req.Input)
		}
		if len(arr) != 2 {
			t.Errorf("batch len = %d, want 2", len(arr))
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(ollamaEmbedResponse{
			Model:      "nomic-embed-text",
			Embeddings: [][]float32{{1, 0, 0}, {0, 1, 0}},
		})
	})
	defer ts.Close()

	e := NewOllamaEmbedder("", ts.URL, 0, time.Second)
	vs, err := e.EmbedBatch(context.Background(), []string{"a", "b"})
	if err != nil {
		t.Fatalf("EmbedBatch: %v", err)
	}
	if len(vs) != 2 {
		t.Fatalf("len(vs) = %d, want 2", len(vs))
	}
}

func TestOllamaEmbedBatchEmpty(t *testing.T) {
	e := NewOllamaEmbedder("", "http://localhost:11434", 0, time.Second)
	vs, err := e.EmbedBatch(context.Background(), nil)
	if err != nil {
		t.Fatalf("EmbedBatch(nil): %v", err)
	}
	if vs != nil {
		t.Errorf("want nil for empty input, got %v", vs)
	}
}

func TestOllamaEmbedErrorStatus(t *testing.T) {
	ts := fakeOllama(t, func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "model not found", http.StatusNotFound)
	})
	defer ts.Close()

	e := NewOllamaEmbedder("", ts.URL, 0, time.Second)
	if _, err := e.Embed(context.Background(), "anything"); err == nil {
		t.Error("Embed succeeded on 404; want error")
	}
}

func TestOllamaIsAvailable(t *testing.T) {
	cases := []struct {
		name   string
		models []string
		want   bool
	}{
		{"model present", []string{"nomic-embed-text"}, true},
		{"model present with tag", []string{"nomic-embed-text:latest"}, true},
		{"model absent", []string{"llama3", "mxbai-embed-large"}, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			ts := fakeOllama(t, func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != "/api/tags" {
					http.NotFound(w, r)
					return
				}
				type m struct {
					Name  string `json:"name"`
					Model string `json:"model"`
				}
				body := struct {
					Models []m `json:"models"`
				}{}
				for _, name := range c.models {
					body.Models = append(body.Models, m{Name: name, Model: name})
				}
				_ = json.NewEncoder(w).Encode(body)
			})
			defer ts.Close()

			e := NewOllamaEmbedder("nomic-embed-text", ts.URL, 0, time.Second)
			got := e.IsAvailable(context.Background())
			if got != c.want {
				t.Errorf("IsAvailable = %v, want %v", got, c.want)
			}
		})
	}
}

func TestOllamaIsAvailableUnreachable(t *testing.T) {
	// Point at an unreachable port — localhost on an unused port should
	// fail to connect within the client timeout.
	e := NewOllamaEmbedder("nomic-embed-text", "http://127.0.0.1:1", 0, 200*time.Millisecond)
	if e.IsAvailable(context.Background()) {
		t.Error("IsAvailable = true for unreachable server")
	}
}

func TestOllamaClose(t *testing.T) {
	e := NewOllamaEmbedder("", "", 0, 0)
	if err := e.Close(); err != nil {
		t.Errorf("Close: %v", err)
	}
	// Double-close must not panic or error.
	if err := e.Close(); err != nil {
		t.Errorf("double close: %v", err)
	}
}
