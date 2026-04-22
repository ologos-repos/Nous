package recall

import (
	"context"
	"errors"
	"strconv"
	"testing"
	"time"

	"github.com/ologos-repos/nous/internal/types"
)

// ===== Mocks =====

type mockEmbedder struct {
	vec     []float32
	err     error
	calls   int
}

func (m *mockEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	m.calls++
	if m.err != nil {
		return nil, m.err
	}
	return m.vec, nil
}

func (m *mockEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i := range texts {
		out[i] = m.vec
	}
	return out, nil
}

func (m *mockEmbedder) Dimensions() int                   { return len(m.vec) }
func (m *mockEmbedder) ModelName() string                 { return "mock" }
func (m *mockEmbedder) IsAvailable(ctx context.Context) bool { return true }
func (m *mockEmbedder) Close() error                      { return nil }

type mockStore struct {
	topicEmbeds      map[int64][][]float32
	topics           map[int64]types.Topic
	hybridScoped     map[int64][]types.SearchResult
	hybridUnscoped   []types.SearchResult
	tripletsByEntity []types.Triplet
	memories         map[int64]*types.Memory

	hybridUnscopedErr error
	walkErr           error
	scopedCalls       []int64
	unscopedCalls     int
}

func (s *mockStore) GetAllTopicEmbeddings(ctx context.Context) (map[int64][][]float32, error) {
	if s.topicEmbeds == nil {
		return map[int64][][]float32{}, nil
	}
	return s.topicEmbeds, nil
}

func (s *mockStore) GetTopic(ctx context.Context, id int64) (types.Topic, error) {
	t, ok := s.topics[id]
	if !ok {
		return types.Topic{}, errors.New("topic not found")
	}
	return t, nil
}

func (s *mockStore) ListTopics(ctx context.Context, source types.TopicSource) ([]types.Topic, error) {
	out := make([]types.Topic, 0, len(s.topics))
	for _, t := range s.topics {
		if source != "" && t.Source != source {
			continue
		}
		out = append(out, t)
	}
	return out, nil
}

func (s *mockStore) HybridRecallScoped(ctx context.Context, query string, topicID int64, limit int, threshold float64) ([]types.SearchResult, error) {
	s.scopedCalls = append(s.scopedCalls, topicID)
	return s.hybridScoped[topicID], nil
}

func (s *mockStore) HybridRecall(ctx context.Context, query, category string, limit int, threshold, recencyBoostHours, recencyBoostValue float64) ([]types.SearchResult, error) {
	s.unscopedCalls++
	if s.hybridUnscopedErr != nil {
		return nil, s.hybridUnscopedErr
	}
	return s.hybridUnscoped, nil
}

func (s *mockStore) WalkGraph(ctx context.Context, entities []string, hops, limit int) ([]types.Triplet, error) {
	if s.walkErr != nil {
		return nil, s.walkErr
	}
	return s.tripletsByEntity, nil
}

func (s *mockStore) GetMemoryByID(ctx context.Context, id int64) (*types.Memory, error) {
	m, ok := s.memories[id]
	if !ok {
		return nil, nil
	}
	return m, nil
}

// ===== Helpers =====

func mustVec(scalars ...float32) []float32 { return scalars }

func makeMemory(id int64, content string) types.Memory {
	return types.Memory{
		ID:        id,
		Content:   content,
		Category:  "general",
		Tier:      types.TierDirector,
		CreatedAt: time.Now().UTC(),
		UpdatedAt: time.Now().UTC(),
	}
}

func makeTopic(id int64, name string) types.Topic {
	return types.Topic{
		ID:          id,
		Name:        name,
		DisplayName: name,
		Source:      types.TopicSourceCurated,
	}
}

// ===== ExtractEntities =====

func TestExtractEntities_BasicCapitalizedPhrases(t *testing.T) {
	// Note: the canonical regex \b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b is a
	// pure-uppercase-followed-by-lowercase pattern. Names like "PostgreSQL"
	// (with internal capitals) DO NOT match the heuristic — that's a
	// documented limitation in spec §7.8. This test uses well-formed
	// capitalized words so the heuristic behaves as the spec describes.
	memories := []types.Memory{
		{Content: "Rhode and Hermetic both rely on Postgres."},
		{Content: "Go uses the Postgres library pgx."},
	}
	got := ExtractEntities(memories)

	want := map[string]bool{
		"Rhode":    true,
		"Hermetic": true,
		"Postgres": true,
		"Go":       true,
	}
	for _, e := range got {
		if !want[e] {
			t.Errorf("unexpected entity %q in result %v", e, got)
		}
	}
	for w := range want {
		found := false
		for _, e := range got {
			if e == w {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("missing expected entity %q in result %v", w, got)
		}
	}
}

func TestExtractEntities_FiltersBlacklist(t *testing.T) {
	memories := []types.Memory{
		{Content: "The cake is a lie. They lied about it. We all knew."},
	}
	got := ExtractEntities(memories)
	for _, e := range got {
		switch e {
		case "The", "They", "We":
			t.Errorf("blacklisted entity slipped through: %q", e)
		}
	}
}

func TestExtractEntities_DeduplicatesCaseInsensitive(t *testing.T) {
	memories := []types.Memory{
		{Content: "Postgres rocks. Postgres also has json. POSTGRES is a typo."},
	}
	got := ExtractEntities(memories)
	count := 0
	for _, e := range got {
		if e == "Postgres" {
			count++
		}
	}
	if count != 1 {
		t.Errorf("expected exactly one Postgres entity, got %d (full list %v)", count, got)
	}
}

func TestExtractEntities_MultiWordPhrase(t *testing.T) {
	memories := []types.Memory{
		{Content: "Working on Go Memory Store this week."},
	}
	got := ExtractEntities(memories)
	found := false
	for _, e := range got {
		if e == "Go Memory Store" {
			found = true
		}
	}
	if !found {
		t.Errorf("expected multi-word phrase 'Go Memory Store' in %v", got)
	}
}

// ===== RouteTopics =====

func TestRouteTopics_NullEmbedderReturnsEmpty(t *testing.T) {
	embedder := &mockEmbedder{vec: nil}
	store := &mockStore{}

	acts, vec, err := RouteTopics(context.Background(), "anything", embedder, store, 5, 0.3)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(acts) != 0 {
		t.Errorf("expected no activations, got %d", len(acts))
	}
	if len(vec) != 0 {
		t.Errorf("expected empty queryVec, got %v", vec)
	}
}

func TestRouteTopics_PicksTopKAboveThreshold(t *testing.T) {
	q := mustVec(1, 0, 0)
	store := &mockStore{
		topicEmbeds: map[int64][][]float32{
			1: {mustVec(1, 0, 0)},      // perfect match  -> 1.0
			2: {mustVec(0.9, 0.1, 0)},  // strong match
			3: {mustVec(0, 1, 0)},      // orthogonal     -> 0.0 (below threshold)
		},
		topics: map[int64]types.Topic{
			1: makeTopic(1, "alpha"),
			2: makeTopic(2, "beta"),
			3: makeTopic(3, "gamma"),
		},
	}
	embedder := &mockEmbedder{vec: q}

	acts, _, err := RouteTopics(context.Background(), "q", embedder, store, 5, 0.3)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(acts) != 2 {
		t.Fatalf("expected 2 activations (above threshold), got %d", len(acts))
	}
	if acts[0].Topic.ID != 1 || acts[1].Topic.ID != 2 {
		t.Errorf("expected topic order [1,2], got [%d,%d]", acts[0].Topic.ID, acts[1].Topic.ID)
	}
	if acts[0].Score < acts[1].Score {
		t.Errorf("expected descending sort, got %v vs %v", acts[0].Score, acts[1].Score)
	}
}

func TestRouteTopics_RespectsTopK(t *testing.T) {
	store := &mockStore{
		topicEmbeds: map[int64][][]float32{
			1: {mustVec(1, 0)},
			2: {mustVec(1, 0)},
			3: {mustVec(1, 0)},
		},
		topics: map[int64]types.Topic{
			1: makeTopic(1, "a"),
			2: makeTopic(2, "b"),
			3: makeTopic(3, "c"),
		},
	}
	embedder := &mockEmbedder{vec: mustVec(1, 0)}

	acts, _, err := RouteTopics(context.Background(), "q", embedder, store, 2, 0.3)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(acts) != 2 {
		t.Errorf("expected exactly topK=2 activations, got %d", len(acts))
	}
}

func TestRouteTopics_NoTopicsReturnsEmptyButKeepsQueryVec(t *testing.T) {
	store := &mockStore{topicEmbeds: map[int64][][]float32{}}
	embedder := &mockEmbedder{vec: mustVec(1, 2, 3)}

	acts, vec, err := RouteTopics(context.Background(), "q", embedder, store, 5, 0.3)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(acts) != 0 {
		t.Errorf("expected no activations, got %d", len(acts))
	}
	if len(vec) != 3 {
		t.Errorf("expected query vec to be returned for fallback, got len=%d", len(vec))
	}
}

// ===== SearchActivatedTopics =====

func TestSearchActivatedTopics_SumsMultiTopicHits(t *testing.T) {
	mem := makeMemory(42, "Shared memory")
	store := &mockStore{
		hybridScoped: map[int64][]types.SearchResult{
			1: {{Memory: mem, Score: 0.8, MatchType: "semantic"}},
			2: {{Memory: mem, Score: 0.6, MatchType: "semantic"}},
		},
	}
	activations := []types.TopicActivation{
		{Topic: makeTopic(1, "alpha"), Score: 0.9},
		{Topic: makeTopic(2, "beta"), Score: 0.5},
	}

	got, err := SearchActivatedTopics(context.Background(), "q", nil, activations, store, 5, 0.3)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("expected 1 deduped memory, got %d", len(got))
	}
	wantScore := 0.9*0.8 + 0.5*0.6
	if got[0].DendriticScore < wantScore-1e-9 || got[0].DendriticScore > wantScore+1e-9 {
		t.Errorf("expected dendritic score %v, got %v", wantScore, got[0].DendriticScore)
	}
	if len(got[0].ViaTopic) != 2 {
		t.Errorf("expected ViaTopic to have both topics, got %v", got[0].ViaTopic)
	}
	if len(got[0].TopicScores) != 2 {
		t.Errorf("expected TopicScores breakdown of 2, got %d", len(got[0].TopicScores))
	}
}

func TestSearchActivatedTopics_SortsDescending(t *testing.T) {
	store := &mockStore{
		hybridScoped: map[int64][]types.SearchResult{
			1: {
				{Memory: makeMemory(1, "low"), Score: 0.2},
				{Memory: makeMemory(2, "high"), Score: 0.9},
			},
		},
	}
	activations := []types.TopicActivation{
		{Topic: makeTopic(1, "alpha"), Score: 1.0},
	}
	got, err := SearchActivatedTopics(context.Background(), "q", nil, activations, store, 5, 0.0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(got) != 2 || got[0].Memory.ID != 2 {
		t.Errorf("expected high-score memory first, got %+v", got)
	}
}

// ===== ExpandViaGraph =====

func TestExpandViaGraph_FindsCrossTopicMemories(t *testing.T) {
	tier2 := []types.ScoredMemory{
		{Memory: makeMemory(1, "Rhode uses Nous"), DendriticScore: 0.8},
	}
	store := &mockStore{
		tripletsByEntity: []types.Triplet{
			{Subject: "Rhode", Predicate: "uses", Object: "Nous", SourceType: "memory", SourceID: "99"},
		},
		memories: map[int64]*types.Memory{
			99: {ID: 99, Content: "deep cut from another topic"},
		},
	}

	triplets, cross, err := ExpandViaGraph(context.Background(), tier2, store, 2, 0.6)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(triplets) != 1 {
		t.Errorf("expected 1 triplet, got %d", len(triplets))
	}
	if len(cross) != 1 {
		t.Fatalf("expected 1 cross-topic memory, got %d", len(cross))
	}
	if cross[0].MatchType != "graph" {
		t.Errorf("expected match_type=graph, got %s", cross[0].MatchType)
	}
	wantScore := 0.6 * 0.5
	if cross[0].DendriticScore < wantScore-1e-9 || cross[0].DendriticScore > wantScore+1e-9 {
		t.Errorf("expected score %v, got %v", wantScore, cross[0].DendriticScore)
	}
}

func TestExpandViaGraph_SkipsMemoriesAlreadyInTier2(t *testing.T) {
	tier2 := []types.ScoredMemory{
		{Memory: makeMemory(99, "Already known"), DendriticScore: 0.8},
	}
	store := &mockStore{
		tripletsByEntity: []types.Triplet{
			{Subject: "Already", Predicate: "is", Object: "known", SourceType: "memory", SourceID: "99"},
		},
		memories: map[int64]*types.Memory{99: {ID: 99}},
	}

	_, cross, err := ExpandViaGraph(context.Background(), tier2, store, 2, 0.6)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(cross) != 0 {
		t.Errorf("expected no cross-topic hits (memory already in Tier 2), got %d", len(cross))
	}
}

func TestExpandViaGraph_NoEntitiesReturnsEmpty(t *testing.T) {
	tier2 := []types.ScoredMemory{
		{Memory: makeMemory(1, "lowercase only, no entities here"), DendriticScore: 0.5},
	}
	store := &mockStore{}

	triplets, cross, err := ExpandViaGraph(context.Background(), tier2, store, 2, 0.6)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(triplets) != 0 || len(cross) != 0 {
		t.Errorf("expected empty results, got %d triplets / %d cross", len(triplets), len(cross))
	}
}

// ===== DendriticRecall (orchestration) =====

func TestDendriticRecall_FullPipeline(t *testing.T) {
	q := mustVec(1, 0, 0)
	mem1 := makeMemory(1, "Go memory pattern: Rhode logs to Nous")
	mem99 := makeMemory(99, "Cross-topic graph hit")

	store := &mockStore{
		topicEmbeds: map[int64][][]float32{1: {mustVec(1, 0, 0)}},
		topics:      map[int64]types.Topic{1: makeTopic(1, "patterns")},
		hybridScoped: map[int64][]types.SearchResult{
			1: {{Memory: mem1, Score: 0.7, MatchType: "semantic"}},
		},
		tripletsByEntity: []types.Triplet{
			{Subject: "Rhode", Predicate: "logs", Object: "Nous", SourceType: "memory", SourceID: "99"},
		},
		memories: map[int64]*types.Memory{99: &mem99},
	}
	embedder := &mockEmbedder{vec: q}

	res, err := DendriticRecall(context.Background(), types.DendriticRecallRequest{Query: "go memory"}, store, embedder)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(res.ActivatedTopics) != 1 {
		t.Errorf("expected 1 activated topic, got %d", len(res.ActivatedTopics))
	}
	if len(res.Results) < 1 {
		t.Errorf("expected merged results, got %d", len(res.Results))
	}
	if len(res.GraphTriplets) != 1 {
		t.Errorf("expected 1 graph triplet, got %d", len(res.GraphTriplets))
	}
	if len(res.CrossTopicHits) != 1 {
		t.Errorf("expected 1 cross-topic hit, got %d", len(res.CrossTopicHits))
	}
}

func TestDendriticRecall_FallbackWhenNullEmbedder(t *testing.T) {
	store := &mockStore{
		hybridUnscoped: []types.SearchResult{
			{Memory: makeMemory(1, "fallback hit"), Score: 0.7, MatchType: "keyword"},
		},
	}
	embedder := &mockEmbedder{vec: nil}

	res, err := DendriticRecall(context.Background(), types.DendriticRecallRequest{Query: "anything"}, store, embedder)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if store.unscopedCalls != 1 {
		t.Errorf("expected unscoped HybridRecall to be called once, got %d", store.unscopedCalls)
	}
	if len(res.Results) != 1 {
		t.Fatalf("expected 1 fallback result, got %d", len(res.Results))
	}
	if res.Results[0].Memory.ID != 1 {
		t.Errorf("expected fallback memory id 1, got %d", res.Results[0].Memory.ID)
	}
	if len(res.ActivatedTopics) != 0 {
		t.Errorf("expected no activated topics in fallback, got %d", len(res.ActivatedTopics))
	}
}

func TestDendriticRecall_FallbackWhenNoTopics(t *testing.T) {
	store := &mockStore{
		topicEmbeds: map[int64][][]float32{}, // no topics
		hybridUnscoped: []types.SearchResult{
			{Memory: makeMemory(2, "via fallback path"), Score: 0.5, MatchType: "hybrid"},
		},
	}
	embedder := &mockEmbedder{vec: mustVec(1, 0, 0)}

	res, err := DendriticRecall(context.Background(), types.DendriticRecallRequest{Query: "x"}, store, embedder)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if store.unscopedCalls != 1 {
		t.Errorf("expected unscoped fallback, got calls=%d", store.unscopedCalls)
	}
	if len(res.Results) != 1 {
		t.Errorf("expected 1 fallback result, got %d", len(res.Results))
	}
}

func TestDendriticRecall_RespectsLimit(t *testing.T) {
	results := make([]types.SearchResult, 30)
	for i := 0; i < 30; i++ {
		id := int64(i + 1)
		results[i] = types.SearchResult{
			Memory: makeMemory(id, "mem "+strconv.FormatInt(id, 10)),
			Score:  1.0 - float64(i)/100,
		}
	}
	store := &mockStore{
		topicEmbeds:  map[int64][][]float32{1: {mustVec(1, 0)}},
		topics:       map[int64]types.Topic{1: makeTopic(1, "all")},
		hybridScoped: map[int64][]types.SearchResult{1: results},
	}
	embedder := &mockEmbedder{vec: mustVec(1, 0)}

	res, err := DendriticRecall(context.Background(), types.DendriticRecallRequest{Query: "x", Limit: 5}, store, embedder)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(res.Results) != 5 {
		t.Errorf("expected limit=5 results, got %d", len(res.Results))
	}
}

func TestDendriticRecall_AppliesDefaults(t *testing.T) {
	embedder := &mockEmbedder{vec: nil}
	store := &mockStore{}
	// Empty request — every field zero.
	_, err := DendriticRecall(context.Background(), types.DendriticRecallRequest{Query: "x"}, store, embedder)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Just ensure it doesn't crash and the unscoped path was exercised.
	if store.unscopedCalls != 1 {
		t.Errorf("expected unscoped fallback to be called once, got %d", store.unscopedCalls)
	}
}
