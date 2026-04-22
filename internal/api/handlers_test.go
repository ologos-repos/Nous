package api

// handlers_test.go — HTTP-layer tests for all Nous API endpoints.
//
// Each test uses net/http/httptest — no live PostgreSQL or Ollama required.
// The mockStore (mock_store_test.go) satisfies HandlerStore with configurable
// return values; the NullEmbedder from internal/embeddings handles /health.
//
// Coverage targets:
//   - All 20 route handlers (happy-path status codes + response shape)
//   - Malformed JSON → 400 with {error, code}
//   - Missing required fields → 400
//   - Content-Type: application/json set by middleware on every response
//   - Panic recovery middleware → 500 instead of crash

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/ologos-repos/nous/internal/embeddings"
	"github.com/ologos-repos/nous/internal/types"
)

// ─── test helpers ────────────────────────────────────────────────────────────

// newTestServer returns a Server wired to ms and the NullEmbedder.
func newTestServer(ms *mockStore) *Server {
	emb := embeddings.NewNullEmbedder()
	return NewServer(ServerConfig{}, Defaults{
		TopicK:              5,
		ActivationThreshold: 0.3,
		MemoryK:             10,
		Threshold:           0.3,
		Hops:                2,
		GraphDiscount:       0.6,
		Limit:               20,
	}, ms, emb, nil)
}

// do executes req against srv.Handler() and returns the recorder.
func do(t *testing.T, srv *Server, req *http.Request) *httptest.ResponseRecorder {
	t.Helper()
	rr := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rr, req)
	return rr
}

// mustJSON encodes v to JSON, failing the test on error.
func mustJSON(t *testing.T, v any) io.Reader {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("mustJSON: %v", err)
	}
	return bytes.NewReader(b)
}

// decodeBody unmarshals the response body into dst.
func decodeBody(t *testing.T, rr *httptest.ResponseRecorder, dst any) {
	t.Helper()
	if err := json.Unmarshal(rr.Body.Bytes(), dst); err != nil {
		t.Fatalf("decodeBody: %v (body=%s)", err, rr.Body.String())
	}
}

// assertStatus fails if the recorded status ≠ want.
func assertStatus(t *testing.T, rr *httptest.ResponseRecorder, want int) {
	t.Helper()
	if rr.Code != want {
		t.Fatalf("status: got %d, want %d (body=%s)", rr.Code, want, rr.Body.String())
	}
}

// assertContentType verifies the response has application/json content-type.
func assertContentType(t *testing.T, rr *httptest.ResponseRecorder) {
	t.Helper()
	ct := rr.Header().Get("Content-Type")
	if !strings.HasPrefix(ct, "application/json") {
		t.Errorf("Content-Type: got %q, want application/json", ct)
	}
}

// assertErrorBody verifies the error envelope fields are non-empty.
func assertErrorBody(t *testing.T, rr *httptest.ResponseRecorder) {
	t.Helper()
	var body struct {
		Error string `json:"error"`
		Code  string `json:"code"`
	}
	decodeBody(t, rr, &body)
	if body.Error == "" {
		t.Errorf("error body: missing 'error' field (body=%s)", rr.Body.String())
	}
}

// jsonPOST builds a POST request with a JSON body.
func jsonPOST(t *testing.T, path string, body any) *http.Request {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, path, mustJSON(t, body))
	req.Header.Set("Content-Type", "application/json")
	return req
}

// ─── GET /health ─────────────────────────────────────────────────────────────

func TestHealth_OK(t *testing.T) {
	srv := newTestServer(newMockStore()) // pingErr = nil → "ok"
	rr := do(t, srv, httptest.NewRequest(http.MethodGet, "/health", nil))

	assertStatus(t, rr, http.StatusOK)
	assertContentType(t, rr)

	var body struct {
		Status    string `json:"status"`
		Embedder  string `json:"embedder"`
		Timestamp string `json:"timestamp"`
	}
	decodeBody(t, rr, &body)
	if body.Status != "ok" {
		t.Errorf("status: got %q, want %q", body.Status, "ok")
	}
}

func TestHealth_Degraded(t *testing.T) {
	ms := newMockStore()
	ms.pingErr = errors.New("connection refused")
	srv := newTestServer(ms)
	rr := do(t, srv, httptest.NewRequest(http.MethodGet, "/health", nil))

	assertStatus(t, rr, http.StatusServiceUnavailable)
	assertContentType(t, rr)

	var body struct{ Status string `json:"status"` }
	decodeBody(t, rr, &body)
	if body.Status != "degraded" {
		t.Errorf("status: got %q, want %q", body.Status, "degraded")
	}
}

// ─── POST /remember ───────────────────────────────────────────────────────────

func TestRemember_OK(t *testing.T) {
	ms := newMockStore()
	ms.rememberResult = types.Memory{ID: 42, Content: "hello world", Category: "general"}
	srv := newTestServer(ms)

	req := jsonPOST(t, "/remember", map[string]any{
		"content":  "hello world",
		"category": "general",
	})
	rr := do(t, srv, req)

	assertStatus(t, rr, http.StatusCreated)
	assertContentType(t, rr)

	var body types.Memory
	decodeBody(t, rr, &body)
	if body.ID != 42 {
		t.Errorf("id: got %d, want 42", body.ID)
	}
}

func TestRemember_MissingContent(t *testing.T) {
	srv := newTestServer(newMockStore())
	rr := do(t, srv, jsonPOST(t, "/remember", map[string]any{"category": "fact"}))

	assertStatus(t, rr, http.StatusBadRequest)
	assertContentType(t, rr)
	assertErrorBody(t, rr)
}

func TestRemember_MalformedJSON(t *testing.T) {
	srv := newTestServer(newMockStore())
	req := httptest.NewRequest(http.MethodPost, "/remember", strings.NewReader(`{bad json`))
	req.Header.Set("Content-Type", "application/json")
	rr := do(t, srv, req)

	assertStatus(t, rr, http.StatusBadRequest)
	assertErrorBody(t, rr)
}

func TestRemember_WhitespaceOnlyContent(t *testing.T) {
	srv := newTestServer(newMockStore())
	rr := do(t, srv, jsonPOST(t, "/remember", map[string]any{"content": "   "}))
	assertStatus(t, rr, http.StatusBadRequest)
}

// ─── POST /recall ─────────────────────────────────────────────────────────────

func TestRecall_HybridMode(t *testing.T) {
	ms := newMockStore()
	ms.hybridRecallResult = []types.SearchResult{
		{Memory: types.Memory{ID: 1, Content: "relevant"}, Score: 0.9, MatchType: "hybrid"},
	}
	srv := newTestServer(ms)

	req := jsonPOST(t, "/recall", map[string]any{
		"query": "relevant topic",
		"mode":  "hybrid",
	})
	rr := do(t, srv, req)

	assertStatus(t, rr, http.StatusOK)
	assertContentType(t, rr)

	var body struct {
		Query   string               `json:"query"`
		Mode    string               `json:"mode"`
		Results []types.SearchResult `json:"results"`
	}
	decodeBody(t, rr, &body)
	if body.Mode != "hybrid" {
		t.Errorf("mode: got %q, want hybrid", body.Mode)
	}
	if len(body.Results) != 1 {
		t.Errorf("results: got %d, want 1", len(body.Results))
	}
}

func TestRecall_DendriticMode_NoTopics(t *testing.T) {
	// With NullEmbedder and no topics, dendritic falls back to hybrid recall.
	ms := newMockStore()
	ms.getAllTopicEmbeddingsResult = map[int64][][]float32{} // empty — no topics
	ms.listTopicsResult = []types.Topic{}
	ms.hybridRecallResult = []types.SearchResult{
		{Memory: types.Memory{ID: 99, Content: "fallback"}, Score: 0.5, MatchType: "keyword"},
	}
	srv := newTestServer(ms)

	req := jsonPOST(t, "/recall", map[string]any{
		"query": "something",
		"mode":  "dendritic",
	})
	rr := do(t, srv, req)

	assertStatus(t, rr, http.StatusOK)
	assertContentType(t, rr)

	var body struct {
		Mode    string               `json:"mode"`
		Results []types.ScoredMemory `json:"results"`
	}
	decodeBody(t, rr, &body)
	if body.Mode != "dendritic" {
		t.Errorf("mode: got %q, want dendritic", body.Mode)
	}
}

func TestRecall_DefaultModeDendritic(t *testing.T) {
	// mode="" defaults to "dendritic"
	ms := newMockStore()
	ms.getAllTopicEmbeddingsResult = map[int64][][]float32{}
	ms.listTopicsResult = []types.Topic{}
	ms.hybridRecallResult = []types.SearchResult{}
	srv := newTestServer(ms)

	req := jsonPOST(t, "/recall", map[string]any{"query": "test"})
	rr := do(t, srv, req)

	assertStatus(t, rr, http.StatusOK)

	var body struct{ Mode string `json:"mode"` }
	decodeBody(t, rr, &body)
	if body.Mode != "dendritic" {
		t.Errorf("default mode: got %q, want dendritic", body.Mode)
	}
}

func TestRecall_BadMode(t *testing.T) {
	srv := newTestServer(newMockStore())
	rr := do(t, srv, jsonPOST(t, "/recall", map[string]any{
		"query": "test",
		"mode":  "turbo",
	}))
	assertStatus(t, rr, http.StatusBadRequest)
	assertErrorBody(t, rr)
}

func TestRecall_MissingQuery(t *testing.T) {
	srv := newTestServer(newMockStore())
	rr := do(t, srv, jsonPOST(t, "/recall", map[string]any{"mode": "hybrid"}))
	assertStatus(t, rr, http.StatusBadRequest)
	assertErrorBody(t, rr)
}

// ─── DELETE /memory/{id} ─────────────────────────────────────────────────────

func TestForget_OK(t *testing.T) {
	ms := newMockStore()
	ms.forgetResult = true
	srv := newTestServer(ms)

	req := httptest.NewRequest(http.MethodDelete, "/memory/7", nil)
	rr := do(t, srv, req)

	assertStatus(t, rr, http.StatusOK)
	assertContentType(t, rr)

	var body struct {
		Deleted bool  `json:"deleted"`
		ID      int64 `json:"id"`
	}
	decodeBody(t, rr, &body)
	if !body.Deleted {
		t.Error("expected deleted=true")
	}
	if body.ID != 7 {
		t.Errorf("id: got %d, want 7", body.ID)
	}
}

func TestForget_NotFound(t *testing.T) {
	ms := newMockStore()
	ms.forgetResult = false
	srv := newTestServer(ms)

	rr := do(t, srv, httptest.NewRequest(http.MethodDelete, "/memory/999", nil))
	assertStatus(t, rr, http.StatusNotFound)
}

func TestForget_BadID(t *testing.T) {
	srv := newTestServer(newMockStore())
	rr := do(t, srv, httptest.NewRequest(http.MethodDelete, "/memory/abc", nil))
	assertStatus(t, rr, http.StatusBadRequest)
}

// ─── GET /topics ──────────────────────────────────────────────────────────────

func TestListTopics_OK(t *testing.T) {
	ms := newMockStore()
	ms.listTopicsResult = []types.Topic{
		{ID: 1, Name: "architecture", DisplayName: "Architecture", Source: types.TopicSourceCurated},
		{ID: 2, Name: "testing", DisplayName: "Testing", Source: types.TopicSourceEmergent},
	}
	srv := newTestServer(ms)

	rr := do(t, srv, httptest.NewRequest(http.MethodGet, "/topics", nil))

	assertStatus(t, rr, http.StatusOK)
	assertContentType(t, rr)

	var body struct {
		Topics []types.Topic `json:"topics"`
	}
	decodeBody(t, rr, &body)
	if len(body.Topics) != 2 {
		t.Errorf("topics count: got %d, want 2", len(body.Topics))
	}
}

func TestListTopics_Empty(t *testing.T) {
	ms := newMockStore()
	ms.listTopicsResult = []types.Topic{}
	srv := newTestServer(ms)

	rr := do(t, srv, httptest.NewRequest(http.MethodGet, "/topics", nil))

	assertStatus(t, rr, http.StatusOK)

	var body struct {
		Topics []types.Topic `json:"topics"`
	}
	decodeBody(t, rr, &body)
	if body.Topics == nil {
		t.Error("expected non-nil topics array")
	}
}

// ─── POST /topics ─────────────────────────────────────────────────────────────

func TestCreateTopic_OK(t *testing.T) {
	ms := newMockStore()
	ms.createTopicResult = types.Topic{ID: 5, Name: "architecture", Source: types.TopicSourceCurated}
	srv := newTestServer(ms)

	req := jsonPOST(t, "/topics", map[string]any{
		"name":        "architecture",
		"description": "System design memories",
	})
	rr := do(t, srv, req)

	assertStatus(t, rr, http.StatusCreated)
	assertContentType(t, rr)

	var body types.Topic
	decodeBody(t, rr, &body)
	if body.ID != 5 {
		t.Errorf("id: got %d, want 5", body.ID)
	}
}

func TestCreateTopic_MissingName(t *testing.T) {
	srv := newTestServer(newMockStore())
	rr := do(t, srv, jsonPOST(t, "/topics", map[string]any{"description": "no name"}))
	assertStatus(t, rr, http.StatusBadRequest)
	assertErrorBody(t, rr)
}

func TestCreateTopic_BadSource(t *testing.T) {
	srv := newTestServer(newMockStore())
	rr := do(t, srv, jsonPOST(t, "/topics", map[string]any{
		"name":   "x",
		"source": "invalid_source",
	}))
	assertStatus(t, rr, http.StatusBadRequest)
}

func TestCreateTopic_Duplicate(t *testing.T) {
	ms := newMockStore()
	ms.createTopicErr = errors.New("duplicate key value violates unique constraint")
	srv := newTestServer(ms)

	rr := do(t, srv, jsonPOST(t, "/topics", map[string]any{"name": "dup"}))
	assertStatus(t, rr, http.StatusConflict)
}

// ─── GET /topics/{id} ─────────────────────────────────────────────────────────

func TestGetTopic_OK(t *testing.T) {
	ms := newMockStore()
	ms.getTopicResult = types.Topic{ID: 3, Name: "deployment"}
	ms.getTopicFound = true
	srv := newTestServer(ms)

	rr := do(t, srv, httptest.NewRequest(http.MethodGet, "/topics/3", nil))
	assertStatus(t, rr, http.StatusOK)
	assertContentType(t, rr)

	var body types.Topic
	decodeBody(t, rr, &body)
	if body.ID != 3 {
		t.Errorf("id: got %d, want 3", body.ID)
	}
}

func TestGetTopic_NotFound(t *testing.T) {
	ms := newMockStore()
	ms.getTopicFound = false
	srv := newTestServer(ms)

	rr := do(t, srv, httptest.NewRequest(http.MethodGet, "/topics/9999", nil))
	assertStatus(t, rr, http.StatusNotFound)
	assertErrorBody(t, rr)
}

func TestGetTopic_BadID(t *testing.T) {
	srv := newTestServer(newMockStore())
	rr := do(t, srv, httptest.NewRequest(http.MethodGet, "/topics/notanid", nil))
	assertStatus(t, rr, http.StatusBadRequest)
}

// ─── POST /topics/{id}/assign ─────────────────────────────────────────────────

func TestAssignToTopic_OK(t *testing.T) {
	ms := newMockStore()
	// assignMemoryToTopicErr stays nil → success
	srv := newTestServer(ms)

	req := jsonPOST(t, "/topics/5/assign", map[string]any{"memory_id": 42})
	rr := do(t, srv, req)

	assertStatus(t, rr, http.StatusOK)
	assertContentType(t, rr)

	var body struct {
		TopicID    int64  `json:"topic_id"`
		MemoryID   int64  `json:"memory_id"`
		EmbedUpdate string `json:"embed_update"`
	}
	decodeBody(t, rr, &body)
	if body.TopicID != 5 {
		t.Errorf("topic_id: got %d, want 5", body.TopicID)
	}
	if body.MemoryID != 42 {
		t.Errorf("memory_id: got %d, want 42", body.MemoryID)
	}
}

func TestAssignToTopic_MissingMemoryID(t *testing.T) {
	srv := newTestServer(newMockStore())
	rr := do(t, srv, jsonPOST(t, "/topics/5/assign", map[string]any{"memory_id": 0}))
	assertStatus(t, rr, http.StatusBadRequest)
}

func TestAssignToTopic_BadTopicID(t *testing.T) {
	srv := newTestServer(newMockStore())
	req := jsonPOST(t, "/topics/nope/assign", map[string]any{"memory_id": 1})
	rr := do(t, srv, req)
	assertStatus(t, rr, http.StatusBadRequest)
}

// ─── GET /triplets ────────────────────────────────────────────────────────────

func TestGetTriplets_OK(t *testing.T) {
	ms := newMockStore()
	ms.getTripletsBySourceResult = []types.Triplet{
		{Subject: "Go", Predicate: "is", Object: "fast"},
	}
	srv := newTestServer(ms)

	req := httptest.NewRequest(http.MethodGet, "/triplets?source_type=memory&source_id=42", nil)
	rr := do(t, srv, req)

	assertStatus(t, rr, http.StatusOK)
	assertContentType(t, rr)

	var body struct {
		Triplets []types.Triplet `json:"triplets"`
	}
	decodeBody(t, rr, &body)
	if len(body.Triplets) != 1 {
		t.Errorf("triplets count: got %d, want 1", len(body.Triplets))
	}
}

func TestGetTriplets_MissingParams(t *testing.T) {
	srv := newTestServer(newMockStore())
	// Missing source_type and source_id → 400
	rr := do(t, srv, httptest.NewRequest(http.MethodGet, "/triplets", nil))
	assertStatus(t, rr, http.StatusBadRequest)
	assertErrorBody(t, rr)
}

func TestGetTriplets_PartialParams(t *testing.T) {
	srv := newTestServer(newMockStore())
	// source_type present but source_id missing → 400
	rr := do(t, srv, httptest.NewRequest(http.MethodGet, "/triplets?source_type=memory", nil))
	assertStatus(t, rr, http.StatusBadRequest)
}

// ─── POST /triplets ───────────────────────────────────────────────────────────

func TestStoreTriplets_OK(t *testing.T) {
	ms := newMockStore()
	ms.storeTripletsResult = 3
	srv := newTestServer(ms)

	req := jsonPOST(t, "/triplets", map[string]any{
		"triplets": [][3]string{
			{"Alice", "knows", "Bob"},
			{"Bob", "uses", "Go"},
			{"Go", "is", "fast"},
		},
		"source_type": "memory",
		"source_id":   "42",
	})
	rr := do(t, srv, req)

	assertStatus(t, rr, http.StatusCreated)
	assertContentType(t, rr)

	var body struct {
		Stored int `json:"stored"`
	}
	decodeBody(t, rr, &body)
	if body.Stored != 3 {
		t.Errorf("stored: got %d, want 3", body.Stored)
	}
}

func TestStoreTriplets_EmptyArray(t *testing.T) {
	srv := newTestServer(newMockStore())
	rr := do(t, srv, jsonPOST(t, "/triplets", map[string]any{
		"triplets": [][3]string{},
	}))
	assertStatus(t, rr, http.StatusBadRequest)
	assertErrorBody(t, rr)
}

func TestStoreTriplets_MalformedJSON(t *testing.T) {
	srv := newTestServer(newMockStore())
	req := httptest.NewRequest(http.MethodPost, "/triplets", strings.NewReader(`{"triplets": "not an array"}`))
	req.Header.Set("Content-Type", "application/json")
	rr := do(t, srv, req)
	assertStatus(t, rr, http.StatusBadRequest)
}

// ─── POST /walk ───────────────────────────────────────────────────────────────

func TestWalkGraph_OK(t *testing.T) {
	ms := newMockStore()
	ms.walkGraphResult = []types.Triplet{
		{Subject: "Alice", Predicate: "knows", Object: "Bob"},
		{Subject: "Bob", Predicate: "uses", Object: "Go"},
	}
	srv := newTestServer(ms)

	req := jsonPOST(t, "/walk", map[string]any{
		"entities": []string{"Alice"},
		"hops":     2,
		"limit":    50,
	})
	rr := do(t, srv, req)

	assertStatus(t, rr, http.StatusOK)
	assertContentType(t, rr)

	var body struct {
		Triplets           []types.Triplet `json:"triplets"`
		EntitiesDiscovered []string        `json:"entities_discovered"`
	}
	decodeBody(t, rr, &body)
	if len(body.Triplets) != 2 {
		t.Errorf("triplets: got %d, want 2", len(body.Triplets))
	}
	// seed entity "Alice" should be in entities_discovered
	found := false
	for _, e := range body.EntitiesDiscovered {
		if e == "Alice" {
			found = true
		}
	}
	if !found {
		t.Errorf("entities_discovered should include seed entity 'Alice', got %v", body.EntitiesDiscovered)
	}
}

func TestWalkGraph_EmptyEntities(t *testing.T) {
	srv := newTestServer(newMockStore())
	rr := do(t, srv, jsonPOST(t, "/walk", map[string]any{"entities": []string{}}))
	assertStatus(t, rr, http.StatusBadRequest)
	assertErrorBody(t, rr)
}

// ─── POST /worker/{name}/remember ─────────────────────────────────────────────

func TestWorkerRemember_OK(t *testing.T) {
	ms := newMockStore()
	ms.workerRememberResult = types.Memory{ID: 11, Content: "worker mem", WorkerName: "alpha"}
	srv := newTestServer(ms)

	req := jsonPOST(t, "/worker/alpha/remember", map[string]any{
		"content":  "worker mem",
		"category": "general",
	})
	rr := do(t, srv, req)

	assertStatus(t, rr, http.StatusCreated)
	assertContentType(t, rr)

	var body types.Memory
	decodeBody(t, rr, &body)
	if body.ID != 11 {
		t.Errorf("id: got %d, want 11", body.ID)
	}
}

func TestWorkerRemember_MissingContent(t *testing.T) {
	srv := newTestServer(newMockStore())
	rr := do(t, srv, jsonPOST(t, "/worker/alpha/remember", map[string]any{}))
	assertStatus(t, rr, http.StatusBadRequest)
}

func TestWorkerRemember_WhitespaceContent(t *testing.T) {
	srv := newTestServer(newMockStore())
	rr := do(t, srv, jsonPOST(t, "/worker/alpha/remember", map[string]any{"content": "  "}))
	assertStatus(t, rr, http.StatusBadRequest)
}

// ─── POST /worker/{name}/recall ──────────────────────────────────────────────

func TestWorkerRecall_OK(t *testing.T) {
	ms := newMockStore()
	ms.workerRecallResult = []types.SearchResult{
		{Memory: types.Memory{ID: 5, Content: "worker knows Go"}, Score: 0.8},
	}
	srv := newTestServer(ms)

	req := jsonPOST(t, "/worker/alpha/recall", map[string]any{
		"query": "Go",
		"limit": 5,
	})
	rr := do(t, srv, req)

	assertStatus(t, rr, http.StatusOK)
	assertContentType(t, rr)

	var body struct {
		WorkerName string               `json:"worker_name"`
		Query      string               `json:"query"`
		Results    []types.SearchResult `json:"results"`
	}
	decodeBody(t, rr, &body)
	if body.WorkerName != "alpha" {
		t.Errorf("worker_name: got %q, want alpha", body.WorkerName)
	}
	if len(body.Results) != 1 {
		t.Errorf("results: got %d, want 1", len(body.Results))
	}
}

func TestWorkerRecall_MissingQuery(t *testing.T) {
	srv := newTestServer(newMockStore())
	rr := do(t, srv, jsonPOST(t, "/worker/beta/recall", map[string]any{}))
	assertStatus(t, rr, http.StatusBadRequest)
	assertErrorBody(t, rr)
}

// ─── DELETE /worker/{name}/memory/{id} ───────────────────────────────────────

func TestWorkerForget_OK(t *testing.T) {
	ms := newMockStore()
	ms.workerForgetResult = true
	srv := newTestServer(ms)

	req := httptest.NewRequest(http.MethodDelete, "/worker/alpha/memory/5", nil)
	rr := do(t, srv, req)

	assertStatus(t, rr, http.StatusOK)

	var body struct {
		Deleted bool  `json:"deleted"`
		ID      int64 `json:"id"`
	}
	decodeBody(t, rr, &body)
	if !body.Deleted {
		t.Error("expected deleted=true")
	}
}

func TestWorkerForget_NotFound(t *testing.T) {
	ms := newMockStore()
	ms.workerForgetResult = false
	srv := newTestServer(ms)

	rr := do(t, srv, httptest.NewRequest(http.MethodDelete, "/worker/alpha/memory/9999", nil))
	assertStatus(t, rr, http.StatusNotFound)
}

func TestWorkerForget_BadID(t *testing.T) {
	srv := newTestServer(newMockStore())
	rr := do(t, srv, httptest.NewRequest(http.MethodDelete, "/worker/alpha/memory/notanid", nil))
	assertStatus(t, rr, http.StatusBadRequest)
}

// ─── GET /worker/{name}/memories ─────────────────────────────────────────────

func TestWorkerMemories_OK(t *testing.T) {
	ms := newMockStore()
	ms.workerRecallResult = []types.SearchResult{
		{Memory: types.Memory{ID: 1, Content: "mem1", WorkerName: "gamma"}, Score: 1.0},
		{Memory: types.Memory{ID: 2, Content: "mem2", WorkerName: "gamma"}, Score: 1.0},
	}
	srv := newTestServer(ms)

	rr := do(t, srv, httptest.NewRequest(http.MethodGet, "/worker/gamma/memories", nil))

	assertStatus(t, rr, http.StatusOK)
	assertContentType(t, rr)

	var body struct {
		WorkerName string         `json:"worker_name"`
		Memories   []types.Memory `json:"memories"`
	}
	decodeBody(t, rr, &body)
	if body.WorkerName != "gamma" {
		t.Errorf("worker_name: got %q, want gamma", body.WorkerName)
	}
	if len(body.Memories) != 2 {
		t.Errorf("memories: got %d, want 2", len(body.Memories))
	}
}

func TestWorkerMemories_Empty(t *testing.T) {
	ms := newMockStore()
	ms.workerRecallResult = []types.SearchResult{}
	srv := newTestServer(ms)

	rr := do(t, srv, httptest.NewRequest(http.MethodGet, "/worker/delta/memories", nil))
	assertStatus(t, rr, http.StatusOK)

	var body struct {
		Memories []types.Memory `json:"memories"`
	}
	decodeBody(t, rr, &body)
	if body.Memories == nil {
		t.Error("expected non-nil memories array")
	}
}

// ─── GET /worker/{name}/history ──────────────────────────────────────────────

func TestWorkerHistoryGet_OK(t *testing.T) {
	ms := newMockStore()
	now := time.Now()
	ms.getWorkerResumeResult = []types.WorkerResume{
		{TaskID: "t1", Description: "built API", Outcome: "completed", FinishedAt: &now},
	}
	srv := newTestServer(ms)

	rr := do(t, srv, httptest.NewRequest(http.MethodGet, "/worker/alpha/history", nil))

	assertStatus(t, rr, http.StatusOK)
	assertContentType(t, rr)

	var body struct {
		WorkerName string               `json:"worker_name"`
		History    []types.WorkerResume `json:"history"`
	}
	decodeBody(t, rr, &body)
	if body.WorkerName != "alpha" {
		t.Errorf("worker_name: got %q, want alpha", body.WorkerName)
	}
	if len(body.History) != 1 {
		t.Errorf("history: got %d, want 1", len(body.History))
	}
}

func TestWorkerHistoryGet_Empty(t *testing.T) {
	ms := newMockStore()
	ms.getWorkerResumeResult = []types.WorkerResume{}
	srv := newTestServer(ms)

	rr := do(t, srv, httptest.NewRequest(http.MethodGet, "/worker/newbie/history", nil))
	assertStatus(t, rr, http.StatusOK)

	var body struct{ History []types.WorkerResume `json:"history"` }
	decodeBody(t, rr, &body)
	if body.History == nil {
		t.Error("expected non-nil history array")
	}
}

// ─── POST /worker/{name}/history ─────────────────────────────────────────────

func TestWorkerHistoryPost_OK(t *testing.T) {
	ms := newMockStore()
	now := time.Now()
	ms.recordTaskCompletionResult = types.WorkerResume{
		TaskID:      "t42",
		Description: "built something",
		Outcome:     "completed",
		FinishedAt:  &now,
	}
	srv := newTestServer(ms)

	req := jsonPOST(t, "/worker/alpha/history", map[string]any{
		"task_id":     "t42",
		"description": "built something",
		"outcome":     "completed",
		"summary":     "it worked",
	})
	rr := do(t, srv, req)

	assertStatus(t, rr, http.StatusCreated)
	assertContentType(t, rr)

	var body types.WorkerResume
	decodeBody(t, rr, &body)
	if body.TaskID != "t42" {
		t.Errorf("task_id: got %q, want t42", body.TaskID)
	}
}

// ─── POST /conversation ───────────────────────────────────────────────────────

func TestConversationPost_OK(t *testing.T) {
	ms := newMockStore()
	ms.logConversationResult = types.ConversationTurn{ID: 7, Role: "user", Content: "hello"}
	srv := newTestServer(ms)

	req := jsonPOST(t, "/conversation", map[string]any{
		"role":    "user",
		"content": "hello",
	})
	rr := do(t, srv, req)

	assertStatus(t, rr, http.StatusCreated)
	assertContentType(t, rr)

	var body types.ConversationTurn
	decodeBody(t, rr, &body)
	if body.ID != 7 {
		t.Errorf("id: got %d, want 7", body.ID)
	}
	if body.Role != "user" {
		t.Errorf("role: got %q, want user", body.Role)
	}
}

func TestConversationPost_MissingRole(t *testing.T) {
	srv := newTestServer(newMockStore())
	rr := do(t, srv, jsonPOST(t, "/conversation", map[string]any{"content": "hello"}))
	assertStatus(t, rr, http.StatusBadRequest)
	assertErrorBody(t, rr)
}

func TestConversationPost_MissingContent(t *testing.T) {
	srv := newTestServer(newMockStore())
	rr := do(t, srv, jsonPOST(t, "/conversation", map[string]any{"role": "user"}))
	assertStatus(t, rr, http.StatusBadRequest)
	assertErrorBody(t, rr)
}

func TestConversationPost_MalformedJSON(t *testing.T) {
	srv := newTestServer(newMockStore())
	req := httptest.NewRequest(http.MethodPost, "/conversation", strings.NewReader(`not json`))
	rr := do(t, srv, req)
	assertStatus(t, rr, http.StatusBadRequest)
}

// ─── GET /conversation ────────────────────────────────────────────────────────

func TestConversationGet_OK(t *testing.T) {
	ms := newMockStore()
	ms.getRecentConversationsResult = []types.ConversationTurn{
		{ID: 1, Role: "user", Content: "hi"},
		{ID: 2, Role: "assistant", Content: "hello"},
	}
	srv := newTestServer(ms)

	rr := do(t, srv, httptest.NewRequest(http.MethodGet, "/conversation?limit=5", nil))

	assertStatus(t, rr, http.StatusOK)
	assertContentType(t, rr)

	var body struct {
		Turns []types.ConversationTurn `json:"turns"`
		Limit int64                    `json:"limit"`
	}
	decodeBody(t, rr, &body)
	if len(body.Turns) != 2 {
		t.Errorf("turns: got %d, want 2", len(body.Turns))
	}
}

func TestConversationGet_Empty(t *testing.T) {
	ms := newMockStore()
	ms.getRecentConversationsResult = []types.ConversationTurn{}
	srv := newTestServer(ms)

	rr := do(t, srv, httptest.NewRequest(http.MethodGet, "/conversation", nil))
	assertStatus(t, rr, http.StatusOK)

	var body struct{ Turns []types.ConversationTurn `json:"turns"` }
	decodeBody(t, rr, &body)
	if body.Turns == nil {
		t.Error("expected non-nil turns array")
	}
}

// ─── Middleware tests ─────────────────────────────────────────────────────────

func TestContentTypeMiddleware(t *testing.T) {
	// Every endpoint should respond with application/json — test a few.
	ms := newMockStore()
	ms.listTopicsResult = []types.Topic{}
	srv := newTestServer(ms)

	endpoints := []struct {
		method string
		path   string
	}{
		{http.MethodGet, "/health"},
		{http.MethodGet, "/topics"},
		{http.MethodGet, "/conversation"},
	}

	for _, ep := range endpoints {
		t.Run(ep.method+" "+ep.path, func(t *testing.T) {
			rr := do(t, srv, httptest.NewRequest(ep.method, ep.path, nil))
			ct := rr.Header().Get("Content-Type")
			if !strings.HasPrefix(ct, "application/json") {
				t.Errorf("%s %s: Content-Type=%q, want application/json", ep.method, ep.path, ct)
			}
		})
	}
}

func TestPanicRecoveryMiddleware(t *testing.T) {
	// Build a minimal handler chain around a panic-inducing handler to verify
	// the recover middleware returns 500 instead of crashing the process.
	panicHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		panic("intentional test panic")
	})
	wrapped := chain(panicHandler, recoverMiddleware(nil), jsonContentType)

	req := httptest.NewRequest(http.MethodGet, "/any", nil)
	rr := httptest.NewRecorder()
	wrapped.ServeHTTP(rr, req)

	if rr.Code != http.StatusInternalServerError {
		t.Errorf("panic recovery: got %d, want %d", rr.Code, http.StatusInternalServerError)
	}
	// Should still have an error body.
	var body struct {
		Error string `json:"error"`
	}
	if err := json.Unmarshal(rr.Body.Bytes(), &body); err != nil {
		t.Fatalf("could not decode panic response: %v", err)
	}
	if body.Error == "" {
		t.Error("expected non-empty error field after panic recovery")
	}
}

func TestPanicRecovery_ViaServer(t *testing.T) {
	// Verify that a real server's middleware chain catches panics gracefully.
	// We inject a mock store that panics on Ping().
	ms := newMockStore()
	ms.pingErr = nil // health check would normally succeed

	srv := newTestServer(ms)
	// Wrap the handler with a panic-inducing outer handler for /health.
	panicOnHealth := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/panic-test" {
			panic("boom")
		}
		srv.Handler().ServeHTTP(w, r)
	})
	wrapped := chain(panicOnHealth, recoverMiddleware(nil), jsonContentType)

	rr := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/panic-test", nil)
	wrapped.ServeHTTP(rr, req)

	if rr.Code != http.StatusInternalServerError {
		t.Errorf("server panic recovery: got %d, want 500", rr.Code)
	}
}

// ─── Error code tests ─────────────────────────────────────────────────────────

func TestErrorCodeField(t *testing.T) {
	// Verify that error responses include a machine-readable "code" field.
	srv := newTestServer(newMockStore())

	cases := []struct {
		name     string
		req      *http.Request
		wantCode string
	}{
		{
			name:     "remember missing content",
			req:      jsonPOST(t, "/remember", map[string]any{"category": "fact"}),
			wantCode: "missing_content",
		},
		{
			name:     "recall missing query",
			req:      jsonPOST(t, "/recall", map[string]any{"mode": "hybrid"}),
			wantCode: "missing_query",
		},
		{
			name:     "forget bad id",
			req:      httptest.NewRequest(http.MethodDelete, "/memory/abc", nil),
			wantCode: "bad_id",
		},
		{
			name:     "create topic missing name",
			req:      jsonPOST(t, "/topics", map[string]any{"description": "no name"}),
			wantCode: "missing_name",
		},
		{
			name:     "triplets empty array",
			req:      jsonPOST(t, "/triplets", map[string]any{"triplets": [][3]string{}}),
			wantCode: "no_triplets",
		},
		{
			name:     "walk empty entities",
			req:      jsonPOST(t, "/walk", map[string]any{"entities": []string{}}),
			wantCode: "no_entities",
		},
		{
			name:     "triplets missing params",
			req:      httptest.NewRequest(http.MethodGet, "/triplets", nil),
			wantCode: "missing_params",
		},
		{
			name:     "conversation missing role",
			req:      jsonPOST(t, "/conversation", map[string]any{"content": "hello"}),
			wantCode: "missing_role",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rr := do(t, srv, tc.req)
			var body errorResponse
			decodeBody(t, rr, &body)
			if body.Code != tc.wantCode {
				t.Errorf("code: got %q, want %q (body=%s)", body.Code, tc.wantCode, rr.Body.String())
			}
		})
	}
}

// ─── Store error propagation ──────────────────────────────────────────────────

func TestStoreErrors_Return500(t *testing.T) {
	storeErr := errors.New("simulated store failure")

	t.Run("remember store failure", func(t *testing.T) {
		ms := newMockStore()
		ms.rememberErr = storeErr
		srv := newTestServer(ms)
		rr := do(t, srv, jsonPOST(t, "/remember", map[string]any{"content": "test"}))
		assertStatus(t, rr, http.StatusInternalServerError)
		assertErrorBody(t, rr)
	})

	t.Run("list topics store failure", func(t *testing.T) {
		ms := newMockStore()
		ms.listTopicsErr = storeErr
		srv := newTestServer(ms)
		rr := do(t, srv, httptest.NewRequest(http.MethodGet, "/topics", nil))
		assertStatus(t, rr, http.StatusInternalServerError)
		assertErrorBody(t, rr)
	})

	t.Run("walk graph store failure", func(t *testing.T) {
		ms := newMockStore()
		ms.walkGraphErr = storeErr
		srv := newTestServer(ms)
		rr := do(t, srv, jsonPOST(t, "/walk", map[string]any{"entities": []string{"Alice"}}))
		assertStatus(t, rr, http.StatusInternalServerError)
		assertErrorBody(t, rr)
	})

	t.Run("log conversation store failure", func(t *testing.T) {
		ms := newMockStore()
		ms.logConversationErr = storeErr
		srv := newTestServer(ms)
		rr := do(t, srv, jsonPOST(t, "/conversation", map[string]any{
			"role": "user", "content": "test",
		}))
		assertStatus(t, rr, http.StatusInternalServerError)
		assertErrorBody(t, rr)
	})

	t.Run("worker remember store failure", func(t *testing.T) {
		ms := newMockStore()
		ms.workerRememberErr = storeErr
		srv := newTestServer(ms)
		rr := do(t, srv, jsonPOST(t, "/worker/alpha/remember", map[string]any{"content": "test"}))
		assertStatus(t, rr, http.StatusInternalServerError)
	})
}
