//go:build e2e

package api

// e2e_test.go — end-to-end smoke tests for the Nous HTTP server.
//
// These tests spin up a real net/http server via httptest.NewServer using the
// actual api.Server router, then drive it with a real HTTP client. No
// PostgreSQL or Ollama is required — a stateful in-memory store (e2eStore)
// satisfies HandlerStore and the NullEmbedder replaces Ollama.
//
// ─── Running the tests ───────────────────────────────────────────────────────
//
//	go test -tags e2e -v ./internal/api/ -run TestE2E
//
// The tests are gated behind the "e2e" build tag so they are excluded from the
// default `go test ./...` run (which would otherwise require no external deps
// but does start a TCP listener).
//
// ─── Coverage ────────────────────────────────────────────────────────────────
//  1. GET  /health              → 200, status "ok"
//  2. POST /remember            → 201, ID assigned, content echoed back
//  3. POST /recall (hybrid)     → 200, results slice present
//  4. GET  /topics              → 200, topics array present
//  5. POST /topics              → 201, topic created
//  6. Full cycle: remember → recall → verify content in results
//  7. POST /worker/{name}/remember → 201
//  8. POST /worker/{name}/recall   → 200
//  9. POST /conversation           → 201
// 10. GET  /conversation           → 200, turns returned
// 11. Error cases: missing fields  → 400
// 12. Error cases: bad mode        → 400

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/ologos-repos/nous/internal/embeddings"
	"github.com/ologos-repos/nous/internal/types"
)

// ─── e2eStore ────────────────────────────────────────────────────────────────

// e2eStore is a thread-safe, fully-stateful in-memory implementation of
// HandlerStore. Unlike the unit-test mockStore (which returns pre-canned
// values), e2eStore actually stores and retrieves data across calls, making it
// suitable for testing multi-step flows (remember → recall → forget).
type e2eStore struct {
	mu sync.Mutex

	memories    map[int64]types.Memory
	nextMemID   int64
	topicAssign map[int64]int64 // memoryID → topicID

	topics      map[int64]types.Topic
	nextTopicID int64

	triplets   []types.Triplet
	nextTripID int64

	workerMems    map[string]map[int64]types.Memory // workerName → id → memory
	nextWorkerIDs map[string]int64

	workerResumes map[string][]types.WorkerResume // workerName → []resume

	conversations []types.ConversationTurn
	nextConvID    int64
}

func newE2EStore() *e2eStore {
	return &e2eStore{
		memories:      make(map[int64]types.Memory),
		topics:        make(map[int64]types.Topic),
		workerMems:    make(map[string]map[int64]types.Memory),
		nextWorkerIDs: make(map[string]int64),
		workerResumes: make(map[string][]types.WorkerResume),
		topicAssign:   make(map[int64]int64),
		nextMemID:     1,
		nextTopicID:   1,
		nextTripID:    1,
		nextConvID:    1,
	}
}

// Compile-time guard: e2eStore must satisfy HandlerStore.
var _ HandlerStore = (*e2eStore)(nil)

// ── Connectivity ──

func (s *e2eStore) Ping(_ context.Context) error          { return nil }
func (s *e2eStore) Pool() *pgxpool.Pool                   { return nil }

// ── Director memory ──

func (s *e2eStore) Remember(_ context.Context, content, category string, topicID *int64, metadata map[string]any) (types.Memory, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	id := s.nextMemID
	s.nextMemID++
	now := time.Now().UTC()
	m := types.Memory{
		ID:        id,
		Content:   content,
		Category:  category,
		Tier:      types.TierDirector,
		TopicID:   topicID,
		Metadata:  metadata,
		CreatedAt: now,
		UpdatedAt: now,
	}
	s.memories[id] = m
	return m, nil
}

func (s *e2eStore) HybridRecall(_ context.Context, query, category string, limit int, threshold, _, _ float64) ([]types.SearchResult, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	q := strings.ToLower(strings.TrimSpace(query))
	var results []types.SearchResult
	for _, m := range s.memories {
		if category != "" && m.Category != category {
			continue
		}
		if q != "" && !strings.Contains(strings.ToLower(m.Content), q) {
			continue
		}
		results = append(results, types.SearchResult{
			Memory:    m,
			Score:     1.0,
			MatchType: "keyword",
		})
	}
	if limit > 0 && len(results) > limit {
		results = results[:limit]
	}
	return results, nil
}

func (s *e2eStore) HybridRecallScoped(_ context.Context, query string, topicID int64, limit int, _ float64) ([]types.SearchResult, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	q := strings.ToLower(strings.TrimSpace(query))
	var results []types.SearchResult
	for _, m := range s.memories {
		assignedTopic, ok := s.topicAssign[m.ID]
		if !ok || assignedTopic != topicID {
			continue
		}
		if q != "" && !strings.Contains(strings.ToLower(m.Content), q) {
			continue
		}
		results = append(results, types.SearchResult{
			Memory:    m,
			Score:     1.0,
			MatchType: "keyword",
		})
	}
	if limit > 0 && len(results) > limit {
		results = results[:limit]
	}
	return results, nil
}

func (s *e2eStore) Forget(_ context.Context, id int64) (bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.memories[id]; !ok {
		return false, nil
	}
	delete(s.memories, id)
	delete(s.topicAssign, id)
	return true, nil
}

func (s *e2eStore) GetMemoryByID(_ context.Context, id int64) (types.Memory, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	m, ok := s.memories[id]
	return m, ok, nil
}

// ── Topics ──

func (s *e2eStore) ListTopics(_ context.Context, source types.TopicSource) ([]types.Topic, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var out []types.Topic
	for _, t := range s.topics {
		if source != "" && t.Source != source {
			continue
		}
		out = append(out, t)
	}
	return out, nil
}

func (s *e2eStore) CreateTopic(_ context.Context, name, displayName, description string, source types.TopicSource) (types.Topic, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	// Unique-name check.
	for _, t := range s.topics {
		if t.Name == name {
			return types.Topic{}, fmt.Errorf("duplicate topic name %q", name)
		}
	}
	id := s.nextTopicID
	s.nextTopicID++
	now := time.Now().UTC()
	t := types.Topic{
		ID:          id,
		Name:        name,
		DisplayName: displayName,
		Description: description,
		Source:      source,
		CreatedAt:   now,
		UpdatedAt:   now,
	}
	s.topics[id] = t
	return t, nil
}

func (s *e2eStore) GetTopic(_ context.Context, id int64) (types.Topic, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	t, ok := s.topics[id]
	return t, ok, nil
}

func (s *e2eStore) AssignMemoryToTopic(_ context.Context, memoryID, topicID int64) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.topicAssign[memoryID] = topicID
	// Bump memory count on the topic.
	if t, ok := s.topics[topicID]; ok {
		t.MemoryCount++
		s.topics[topicID] = t
	}
	return nil
}

func (s *e2eStore) UpdateTopicEmbeddings(_ context.Context, _ int64) error {
	return nil // no-op: NullEmbedder produces no vectors
}

func (s *e2eStore) GetAllTopicEmbeddings(_ context.Context) (map[int64][][]float32, error) {
	return map[int64][][]float32{}, nil // empty → dendritic recall falls back to HybridRecall
}

// ── Graph ──

func (s *e2eStore) GetTripletsBySource(_ context.Context, sourceType, sourceID string) ([]types.Triplet, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var out []types.Triplet
	for _, t := range s.triplets {
		if t.SourceType == sourceType && t.SourceID == sourceID {
			out = append(out, t)
		}
	}
	return out, nil
}

func (s *e2eStore) StoreTriplets(_ context.Context, triplets [][3]string, sourceType, sourceID string, confidence float64) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	now := time.Now().UTC()
	for _, triple := range triplets {
		id := s.nextTripID
		s.nextTripID++
		s.triplets = append(s.triplets, types.Triplet{
			ID:         id,
			Subject:    triple[0],
			Predicate:  triple[1],
			Object:     triple[2],
			SourceType: sourceType,
			SourceID:   sourceID,
			Confidence: confidence,
			CreatedAt:  now,
		})
	}
	return len(triplets), nil
}

func (s *e2eStore) WalkGraph(_ context.Context, entities []string, _, _ int) ([]types.Triplet, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	entitySet := make(map[string]struct{}, len(entities))
	for _, e := range entities {
		entitySet[strings.ToLower(e)] = struct{}{}
	}
	var out []types.Triplet
	for _, t := range s.triplets {
		if _, ok := entitySet[strings.ToLower(t.Subject)]; ok {
			out = append(out, t)
		} else if _, ok := entitySet[strings.ToLower(t.Object)]; ok {
			out = append(out, t)
		}
	}
	return out, nil
}

// ── Worker memory ──

func (s *e2eStore) WorkerRemember(_ context.Context, workerName, content, category string, metadata map[string]any) (types.Memory, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.workerMems[workerName] == nil {
		s.workerMems[workerName] = make(map[int64]types.Memory)
	}
	id := s.nextWorkerIDs[workerName] + 1
	s.nextWorkerIDs[workerName] = id
	now := time.Now().UTC()
	m := types.Memory{
		ID:         id,
		Content:    content,
		Category:   category,
		Tier:       types.TierShared,
		WorkerName: workerName,
		Metadata:   metadata,
		CreatedAt:  now,
		UpdatedAt:  now,
	}
	s.workerMems[workerName][id] = m
	return m, nil
}

func (s *e2eStore) WorkerRecall(_ context.Context, workerName, query string, limit int) ([]types.SearchResult, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	q := strings.ToLower(strings.TrimSpace(query))
	var results []types.SearchResult
	for _, m := range s.workerMems[workerName] {
		if q != "" && !strings.Contains(strings.ToLower(m.Content), q) {
			continue
		}
		results = append(results, types.SearchResult{
			Memory:    m,
			Score:     1.0,
			MatchType: "keyword",
		})
	}
	if limit > 0 && len(results) > limit {
		results = results[:limit]
	}
	return results, nil
}

func (s *e2eStore) WorkerForget(_ context.Context, workerName string, id int64) (bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.workerMems[workerName] == nil {
		return false, nil
	}
	if _, ok := s.workerMems[workerName][id]; !ok {
		return false, nil
	}
	delete(s.workerMems[workerName], id)
	return true, nil
}

func (s *e2eStore) RecordTaskCompletion(_ context.Context, workerName, taskID, description, outcome, skillsUsed, summary string, startedAt *time.Time) (types.WorkerResume, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	now := time.Now().UTC()
	r := types.WorkerResume{
		TaskID:      taskID,
		Description: description,
		Outcome:     outcome,
		SkillsUsed:  skillsUsed,
		Summary:     summary,
		StartedAt:   startedAt,
		FinishedAt:  &now,
	}
	s.workerResumes[workerName] = append(s.workerResumes[workerName], r)
	return r, nil
}

func (s *e2eStore) GetWorkerResume(_ context.Context, workerName string, limit int) ([]types.WorkerResume, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	resumes := s.workerResumes[workerName]
	if limit > 0 && len(resumes) > limit {
		resumes = resumes[len(resumes)-limit:]
	}
	return resumes, nil
}

// ── Conversation ──

func (s *e2eStore) LogConversation(_ context.Context, role, content string) (types.ConversationTurn, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	id := s.nextConvID
	s.nextConvID++
	turn := types.ConversationTurn{
		ID:        id,
		Role:      role,
		Content:   content,
		CreatedAt: time.Now().UTC(),
	}
	s.conversations = append(s.conversations, turn)
	return turn, nil
}

func (s *e2eStore) GetRecentConversations(_ context.Context, limit int, _ float64) ([]types.ConversationTurn, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	turns := s.conversations
	if limit > 0 && len(turns) > limit {
		turns = turns[len(turns)-limit:]
	}
	return turns, nil
}

// ─── Server factory ──────────────────────────────────────────────────────────

// newE2EServer wires the e2eStore + NullEmbedder into a Server with sensible
// defaults and returns the test HTTP server (caller must defer ts.Close()).
func newE2EServer(t *testing.T) (*httptest.Server, *e2eStore) {
	t.Helper()
	st := newE2EStore()
	emb := embeddings.NewNullEmbedder()
	srv := NewServer(ServerConfig{}, Defaults{
		TopicK:              5,
		ActivationThreshold: 0.3,
		MemoryK:             10,
		Threshold:           0.3,
		Hops:                2,
		GraphDiscount:       0.6,
		Limit:               20,
	}, st, emb, nil)
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)
	return ts, st
}

// ─── HTTP helpers ─────────────────────────────────────────────────────────────

func e2ePost(t *testing.T, url string, body any) *http.Response {
	t.Helper()
	b, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("e2ePost marshal: %v", err)
	}
	resp, err := http.Post(url, "application/json", bytes.NewReader(b)) //nolint:noctx
	if err != nil {
		t.Fatalf("e2ePost %s: %v", url, err)
	}
	return resp
}

func e2eGet(t *testing.T, url string) *http.Response {
	t.Helper()
	resp, err := http.Get(url) //nolint:noctx
	if err != nil {
		t.Fatalf("e2eGet %s: %v", url, err)
	}
	return resp
}

func readBody(t *testing.T, resp *http.Response) []byte {
	t.Helper()
	defer resp.Body.Close()
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("readBody: %v", err)
	}
	return b
}

func mustUnmarshal(t *testing.T, data []byte, dst any) {
	t.Helper()
	if err := json.Unmarshal(data, dst); err != nil {
		t.Fatalf("mustUnmarshal: %v (data=%s)", err, data)
	}
}

func requireStatus(t *testing.T, resp *http.Response, want int, body []byte) {
	t.Helper()
	if resp.StatusCode != want {
		t.Fatalf("status: got %d, want %d (body=%s)", resp.StatusCode, want, body)
	}
}

// ─── Tests ───────────────────────────────────────────────────────────────────

// TestE2E_Health verifies the /health endpoint returns 200 with status "ok"
// and that the embedder is identified as "null".
func TestE2E_Health(t *testing.T) {
	ts, _ := newE2EServer(t)

	resp := e2eGet(t, ts.URL+"/health")
	body := readBody(t, resp)
	requireStatus(t, resp, http.StatusOK, body)

	var got map[string]any
	mustUnmarshal(t, body, &got)

	if got["status"] != "ok" {
		t.Errorf("health.status = %q, want %q", got["status"], "ok")
	}
	if got["embedder"] != "null" {
		t.Errorf("health.embedder = %q, want %q", got["embedder"], "null")
	}
	if got["embedder_available"] != true {
		t.Errorf("health.embedder_available = %v, want true", got["embedder_available"])
	}
}

// TestE2E_Remember verifies POST /remember stores a memory and returns 201
// with the created memory (ID set, content echoed).
func TestE2E_Remember(t *testing.T) {
	ts, _ := newE2EServer(t)

	resp := e2ePost(t, ts.URL+"/remember", map[string]any{
		"content":  "The Go proverb: clear is better than clever.",
		"category": "lesson",
	})
	body := readBody(t, resp)
	requireStatus(t, resp, http.StatusCreated, body)

	var got types.Memory
	mustUnmarshal(t, body, &got)

	if got.ID == 0 {
		t.Error("expected non-zero ID from /remember")
	}
	if got.Content != "The Go proverb: clear is better than clever." {
		t.Errorf("content mismatch: got %q", got.Content)
	}
	if got.Category != "lesson" {
		t.Errorf("category = %q, want %q", got.Category, "lesson")
	}
}

// TestE2E_Remember_MissingContent verifies that omitting content returns 400.
func TestE2E_Remember_MissingContent(t *testing.T) {
	ts, _ := newE2EServer(t)

	resp := e2ePost(t, ts.URL+"/remember", map[string]any{
		"category": "fact",
	})
	body := readBody(t, resp)
	requireStatus(t, resp, http.StatusBadRequest, body)

	var got map[string]any
	mustUnmarshal(t, body, &got)
	if got["code"] != "missing_content" {
		t.Errorf("error code = %q, want %q", got["code"], "missing_content")
	}
}

// TestE2E_Recall_Hybrid verifies POST /recall with mode=hybrid returns 200
// and a results array.
func TestE2E_Recall_Hybrid(t *testing.T) {
	ts, st := newE2EServer(t)

	// Pre-populate store directly so recall has something to find.
	ctx := context.Background()
	if _, err := st.Remember(ctx, "SQLite is an embedded database engine.", "fact", nil, nil); err != nil {
		t.Fatalf("pre-populate: %v", err)
	}
	if _, err := st.Remember(ctx, "PostgreSQL supports full-text search.", "fact", nil, nil); err != nil {
		t.Fatalf("pre-populate: %v", err)
	}

	resp := e2ePost(t, ts.URL+"/recall", map[string]any{
		"query": "embedded database",
		"mode":  "hybrid",
	})
	body := readBody(t, resp)
	requireStatus(t, resp, http.StatusOK, body)

	var got map[string]any
	mustUnmarshal(t, body, &got)

	if got["mode"] != "hybrid" {
		t.Errorf("mode = %q, want %q", got["mode"], "hybrid")
	}
	results, ok := got["results"].([]any)
	if !ok {
		t.Fatalf("results is not an array: %T", got["results"])
	}
	if len(results) == 0 {
		t.Error("expected at least one result for 'embedded database' query")
	}
}

// TestE2E_Recall_Dendritic verifies POST /recall with mode=dendritic falls
// back gracefully (NullEmbedder → no topic activations → HybridRecall) and
// returns 200 with the dendritic response envelope.
func TestE2E_Recall_Dendritic(t *testing.T) {
	ts, st := newE2EServer(t)

	ctx := context.Background()
	if _, err := st.Remember(ctx, "Dendritic recall routes through topic embeddings.", "lesson", nil, nil); err != nil {
		t.Fatalf("pre-populate: %v", err)
	}

	resp := e2ePost(t, ts.URL+"/recall", map[string]any{
		"query": "dendritic",
		"mode":  "dendritic",
	})
	body := readBody(t, resp)
	requireStatus(t, resp, http.StatusOK, body)

	var got map[string]any
	mustUnmarshal(t, body, &got)

	if got["mode"] != "dendritic" {
		t.Errorf("mode = %q, want %q", got["mode"], "dendritic")
	}
	// Response must contain the dendritic envelope fields.
	if _, ok := got["activated_topics"]; !ok {
		t.Error("response missing 'activated_topics'")
	}
	if _, ok := got["results"]; !ok {
		t.Error("response missing 'results'")
	}
}

// TestE2E_Recall_BadMode verifies an unknown mode returns 400.
func TestE2E_Recall_BadMode(t *testing.T) {
	ts, _ := newE2EServer(t)

	resp := e2ePost(t, ts.URL+"/recall", map[string]any{
		"query": "anything",
		"mode":  "turbo-boost",
	})
	body := readBody(t, resp)
	requireStatus(t, resp, http.StatusBadRequest, body)

	var got map[string]any
	mustUnmarshal(t, body, &got)
	if got["code"] != "bad_mode" {
		t.Errorf("error code = %q, want %q", got["code"], "bad_mode")
	}
}

// TestE2E_Recall_MissingQuery verifies that omitting the query returns 400.
func TestE2E_Recall_MissingQuery(t *testing.T) {
	ts, _ := newE2EServer(t)

	resp := e2ePost(t, ts.URL+"/recall", map[string]any{
		"mode": "hybrid",
	})
	body := readBody(t, resp)
	requireStatus(t, resp, http.StatusBadRequest, body)
}

// TestE2E_Topics_ListAndCreate verifies GET /topics and POST /topics.
func TestE2E_Topics_ListAndCreate(t *testing.T) {
	ts, _ := newE2EServer(t)

	// Initially empty.
	resp := e2eGet(t, ts.URL+"/topics")
	body := readBody(t, resp)
	requireStatus(t, resp, http.StatusOK, body)

	var listEmpty map[string]any
	mustUnmarshal(t, body, &listEmpty)
	topics := listEmpty["topics"].([]any)
	if len(topics) != 0 {
		t.Errorf("expected 0 topics initially, got %d", len(topics))
	}

	// Create a topic.
	resp2 := e2ePost(t, ts.URL+"/topics", map[string]any{
		"name":         "go-patterns",
		"display_name": "Go Patterns",
		"description":  "Idiomatic Go design patterns and best practices.",
		"source":       "curated",
	})
	body2 := readBody(t, resp2)
	requireStatus(t, resp2, http.StatusCreated, body2)

	var topic types.Topic
	mustUnmarshal(t, body2, &topic)
	if topic.ID == 0 {
		t.Error("expected non-zero topic ID")
	}
	if topic.Name != "go-patterns" {
		t.Errorf("topic.Name = %q, want %q", topic.Name, "go-patterns")
	}

	// List now returns 1.
	resp3 := e2eGet(t, ts.URL+"/topics")
	body3 := readBody(t, resp3)
	requireStatus(t, resp3, http.StatusOK, body3)

	var listOne map[string]any
	mustUnmarshal(t, body3, &listOne)
	after := listOne["topics"].([]any)
	if len(after) != 1 {
		t.Errorf("expected 1 topic after create, got %d", len(after))
	}
}

// TestE2E_Topics_GetByID verifies GET /topics/{id} returns 200 for existing
// topics and 404 for unknown IDs.
func TestE2E_Topics_GetByID(t *testing.T) {
	ts, _ := newE2EServer(t)

	// Create a topic first.
	createResp := e2ePost(t, ts.URL+"/topics", map[string]any{
		"name":   "testing",
		"source": "curated",
	})
	createBody := readBody(t, createResp)
	requireStatus(t, createResp, http.StatusCreated, createBody)

	var created types.Topic
	mustUnmarshal(t,createBody, &created)

	// Fetch it by ID.
	getURL := fmt.Sprintf("%s/topics/%d", ts.URL, created.ID)
	resp := e2eGet(t, getURL)
	body := readBody(t, resp)
	requireStatus(t, resp, http.StatusOK, body)

	// Non-existent ID → 404.
	resp404 := e2eGet(t, ts.URL+"/topics/99999")
	body404 := readBody(t, resp404)
	requireStatus(t, resp404, http.StatusNotFound, body404)
}

// TestE2E_Topics_DuplicateName verifies that creating two topics with the
// same name returns 409 Conflict on the second attempt.
func TestE2E_Topics_DuplicateName(t *testing.T) {
	ts, _ := newE2EServer(t)

	payload := map[string]any{"name": "duplicate-me", "source": "curated"}
	resp1 := e2ePost(t, ts.URL+"/topics", payload)
	readBody(t, resp1)
	requireStatus(t, resp1, http.StatusCreated, nil)

	resp2 := e2ePost(t, ts.URL+"/topics", payload)
	body2 := readBody(t, resp2)
	requireStatus(t, resp2, http.StatusConflict, body2)
}

// TestE2E_FullCycle is the key integration scenario:
//
//  1. POST /remember with distinct content
//  2. POST /recall (hybrid) querying for that content
//  3. Verify the ingested content appears in the recall results
func TestE2E_FullCycle(t *testing.T) {
	ts, _ := newE2EServer(t)

	uniquePhrase := "xyzzy-nous-e2e-canary-phrase"

	// Step 1: ingest.
	ingestResp := e2ePost(t, ts.URL+"/remember", map[string]any{
		"content":  fmt.Sprintf("End-to-end canary: %s is memorable.", uniquePhrase),
		"category": "fact",
	})
	ingestBody := readBody(t, ingestResp)
	requireStatus(t, ingestResp, http.StatusCreated, ingestBody)

	var ingested types.Memory
	mustUnmarshal(t,ingestBody, &ingested)
	if ingested.ID == 0 {
		t.Fatal("ingest: expected non-zero ID")
	}

	// Step 2: recall using the unique phrase.
	recallResp := e2ePost(t, ts.URL+"/recall", map[string]any{
		"query": uniquePhrase,
		"mode":  "hybrid",
	})
	recallBody := readBody(t, recallResp)
	requireStatus(t, recallResp, http.StatusOK, recallBody)

	var recalled map[string]any
	mustUnmarshal(t,recallBody, &recalled)

	results, ok := recalled["results"].([]any)
	if !ok || len(results) == 0 {
		t.Fatalf("recall returned no results for unique phrase %q", uniquePhrase)
	}

	// Step 3: verify the ingested content appears.
	found := false
	for _, r := range results {
		result, _ := r.(map[string]any)
		mem, _ := result["memory"].(map[string]any)
		content, _ := mem["content"].(string)
		if strings.Contains(content, uniquePhrase) {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("recall results do not contain the ingested content with phrase %q\nresults: %v", uniquePhrase, recalled["results"])
	}
}

// TestE2E_FullCycle_WithDelete extends the full cycle to also verify that
// DELETE /memory/{id} removes the memory and subsequent recall returns nothing.
func TestE2E_FullCycle_WithDelete(t *testing.T) {
	ts, _ := newE2EServer(t)

	phrase := "delete-me-canary-7f3a"

	// Ingest.
	ingestResp := e2ePost(t, ts.URL+"/remember", map[string]any{
		"content": fmt.Sprintf("Temporary memory: %s", phrase),
	})
	ingestBody := readBody(t, ingestResp)
	requireStatus(t, ingestResp, http.StatusCreated, ingestBody)

	var mem types.Memory
	mustUnmarshal(t,ingestBody, &mem)

	// Confirm it's recallable.
	recallBefore := e2ePost(t, ts.URL+"/recall", map[string]any{"query": phrase, "mode": "hybrid"})
	bodyBefore := readBody(t, recallBefore)
	requireStatus(t, recallBefore, http.StatusOK, bodyBefore)

	var rbf map[string]any
	mustUnmarshal(t,bodyBefore, &rbf)
	if len(rbf["results"].([]any)) == 0 {
		t.Fatal("expected memory to be recallable before delete")
	}

	// Delete via HTTP.
	deleteURL := fmt.Sprintf("%s/memory/%d", ts.URL, mem.ID)
	req, _ := http.NewRequest(http.MethodDelete, deleteURL, nil)
	deleteResp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("DELETE %s: %v", deleteURL, err)
	}
	deleteBody := readBody(t, deleteResp)
	requireStatus(t, deleteResp, http.StatusOK, deleteBody)

	// Recall again — should be empty now.
	recallAfter := e2ePost(t, ts.URL+"/recall", map[string]any{"query": phrase, "mode": "hybrid"})
	bodyAfter := readBody(t, recallAfter)
	requireStatus(t, recallAfter, http.StatusOK, bodyAfter)

	var raf map[string]any
	mustUnmarshal(t,bodyAfter, &raf)
	if len(raf["results"].([]any)) != 0 {
		t.Error("expected no results after memory was deleted")
	}
}

// TestE2E_WorkerMemory verifies the worker memory endpoints work end-to-end.
func TestE2E_WorkerMemory(t *testing.T) {
	ts, _ := newE2EServer(t)
	workerName := "alpha"

	// POST /worker/{name}/remember
	storeResp := e2ePost(t, ts.URL+"/worker/"+workerName+"/remember", map[string]any{
		"content":  "Worker Alpha remembers: always close channels.",
		"category": "lesson",
	})
	storeBody := readBody(t, storeResp)
	requireStatus(t, storeResp, http.StatusCreated, storeBody)

	var stored types.Memory
	mustUnmarshal(t,storeBody, &stored)
	if stored.ID == 0 {
		t.Error("expected non-zero ID from worker remember")
	}
	if stored.WorkerName != workerName {
		t.Errorf("worker_name = %q, want %q", stored.WorkerName, workerName)
	}

	// POST /worker/{name}/recall
	recallResp := e2ePost(t, ts.URL+"/worker/"+workerName+"/recall", map[string]any{
		"query": "channels",
	})
	recallBody := readBody(t, recallResp)
	requireStatus(t, recallResp, http.StatusOK, recallBody)

	var recalled map[string]any
	mustUnmarshal(t,recallBody, &recalled)
	results := recalled["results"].([]any)
	if len(results) == 0 {
		t.Error("expected at least one worker recall result")
	}

	// Verify worker scoping: a different worker gets no results.
	otherResp := e2ePost(t, ts.URL+"/worker/beta/recall", map[string]any{
		"query": "channels",
	})
	otherBody := readBody(t, otherResp)
	requireStatus(t, otherResp, http.StatusOK, otherBody)

	var other map[string]any
	mustUnmarshal(t,otherBody, &other)
	otherResults := other["results"].([]any)
	if len(otherResults) != 0 {
		t.Errorf("worker beta should not see alpha's memories, got %d results", len(otherResults))
	}
}

// TestE2E_Conversation verifies POST /conversation stores turns and
// GET /conversation returns them.
func TestE2E_Conversation(t *testing.T) {
	ts, _ := newE2EServer(t)

	turns := []map[string]any{
		{"role": "user", "content": "What is dendritic recall?"},
		{"role": "assistant", "content": "Dendritic recall is a three-tier topic-routed memory pipeline."},
	}
	for _, turn := range turns {
		resp := e2ePost(t, ts.URL+"/conversation", turn)
		body := readBody(t, resp)
		requireStatus(t, resp, http.StatusCreated, body)

		var stored types.ConversationTurn
		mustUnmarshal(t, body, &stored)
		if stored.ID == 0 {
			t.Error("expected non-zero turn ID")
		}
		wantContent := turn["content"].(string)
		if stored.Content != wantContent {
			t.Errorf("turn.content = %q, want %q", stored.Content, wantContent)
		}
	}

	// GET /conversation
	resp := e2eGet(t, ts.URL+"/conversation?limit=10")
	body := readBody(t, resp)
	requireStatus(t, resp, http.StatusOK, body)

	var got map[string]any
	mustUnmarshal(t, body, &got)
	storedTurns := got["turns"].([]any)
	if len(storedTurns) != 2 {
		t.Errorf("expected 2 conversation turns, got %d", len(storedTurns))
	}
}

// TestE2E_ContentTypeHeader verifies that every response carries the
// application/json Content-Type set by the middleware.
func TestE2E_ContentTypeHeader(t *testing.T) {
	ts, _ := newE2EServer(t)

	endpoints := []struct {
		method string
		path   string
		body   any
	}{
		{"GET", "/health", nil},
		{"GET", "/topics", nil},
		{"GET", "/conversation", nil},
	}

	for _, ep := range endpoints {
		t.Run(ep.method+" "+ep.path, func(t *testing.T) {
			var resp *http.Response
			if ep.method == "GET" {
				resp = e2eGet(t, ts.URL+ep.path)
			} else {
				resp = e2ePost(t, ts.URL+ep.path, ep.body)
			}
			readBody(t, resp)
			ct := resp.Header.Get("Content-Type")
			if !strings.HasPrefix(ct, "application/json") {
				t.Errorf("Content-Type = %q, want application/json", ct)
			}
		})
	}
}

// TestE2E_NotFoundMemory verifies DELETE /memory/{id} for an unknown ID
// returns 404 (not found, not a panic).
func TestE2E_NotFoundMemory(t *testing.T) {
	ts, _ := newE2EServer(t)

	req, _ := http.NewRequest(http.MethodDelete, ts.URL+"/memory/99999", nil)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("DELETE: %v", err)
	}
	body := readBody(t, resp)
	requireStatus(t, resp, http.StatusNotFound, body)
}
