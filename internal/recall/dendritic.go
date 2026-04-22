// Package recall implements the dendritic recall pipeline — the three-tier
// topic-routed memory search that is the heart of Nous Go.
//
// Pipeline:
//
//	Tier 1 (RouteTopics)         — query → topic activations via embedding array cosine
//	Tier 2 (SearchActivatedTopics) — per-topic hybrid search, summed for multi-topic hits
//	Tier 3 (ExpandViaGraph)       — entity extraction → BFS walk → cross-topic memories
//
// DendriticRecall orchestrates all three tiers. When no topics exist or the
// embedder is a NullEmbedder, it falls back to unscoped HybridRecall so the
// system degrades gracefully to Python-Nous-equivalent behavior.
package recall

import (
	"context"
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/ologos-repos/nous/internal/embeddings"
	"github.com/ologos-repos/nous/internal/types"
	"github.com/ologos-repos/nous/internal/vectors"
)

// Store is the minimal interface DendriticRecall requires from the storage
// layer. It is satisfied by *store.MemoryStore but kept narrow so the recall
// package is easy to test with mocks.
//
// All "lookup-or-miss" methods follow the (value, found, error) pattern used
// by the storage package: a not-found row returns (zero, false, nil) so
// callers don't have to inspect error strings.
type Store interface {
	// Topic routing
	GetAllTopicEmbeddings(ctx context.Context) (map[int64][][]float32, error)
	GetTopic(ctx context.Context, id int64) (types.Topic, bool, error)
	ListTopics(ctx context.Context, source types.TopicSource) ([]types.Topic, error)

	// Within-topic and global hybrid search
	HybridRecallScoped(ctx context.Context, query string, topicID int64, limit int, threshold float64) ([]types.SearchResult, error)
	HybridRecall(ctx context.Context, query, category string, limit int, threshold, recencyBoostHours, recencyBoostValue float64) ([]types.SearchResult, error)

	// Graph walk + memory lookup
	WalkGraph(ctx context.Context, entities []string, hops, limit int) ([]types.Triplet, error)
	GetMemoryByID(ctx context.Context, id int64) (types.Memory, bool, error)
}

// Defaults applied when a DendriticRecallRequest leaves a field at its zero
// value. They match spec section 7.6.
const (
	defaultTopicK              = 5
	defaultActivationThreshold = 0.3
	defaultMemoryK             = 10
	defaultMemoryThreshold     = 0.3
	defaultHops                = 2
	defaultGraphDiscount       = 0.6
	defaultLimit               = 20

	// Tier 3 seed score for graph-discovered memories: graph_discount * 0.5.
	// The 0.5 multiplier is a flat "base relevance" for graph hits — spec §7.4.
	graphSeedRelevance = 0.5
)

// applyDefaults fills zero-valued fields in the request with the spec defaults.
func applyDefaults(req types.DendriticRecallRequest) types.DendriticRecallRequest {
	if req.TopicK <= 0 {
		req.TopicK = defaultTopicK
	}
	if req.ActivationThreshold <= 0 {
		req.ActivationThreshold = defaultActivationThreshold
	}
	if req.MemoryK <= 0 {
		req.MemoryK = defaultMemoryK
	}
	if req.Threshold <= 0 {
		req.Threshold = defaultMemoryThreshold
	}
	if req.Hops <= 0 {
		req.Hops = defaultHops
	}
	if req.GraphDiscount <= 0 {
		req.GraphDiscount = defaultGraphDiscount
	}
	if req.Limit <= 0 {
		req.Limit = defaultLimit
	}
	return req
}

// RouteTopics embeds the query, scores each topic by max cosine similarity
// against its representative embedding array, and returns the top-K topics
// above activationThreshold (Tier 1 dendritic fan-out).
//
// When the embedder produces a nil/empty vector (NullEmbedder) or no topics
// have embeddings, an empty slice is returned along with the (possibly nil)
// query vector — the caller must then fall back to unscoped search.
func RouteTopics(
	ctx context.Context,
	query string,
	embedder embeddings.EmbeddingProvider,
	store Store,
	topicK int,
	activationThreshold float64,
) ([]types.TopicActivation, []float32, error) {
	if topicK <= 0 {
		topicK = defaultTopicK
	}
	if activationThreshold <= 0 {
		activationThreshold = defaultActivationThreshold
	}

	queryVec, err := embedder.Embed(ctx, query)
	if err != nil {
		// Embedding failure — caller should fall back to keyword-only hybrid recall.
		return nil, nil, fmt.Errorf("embed query: %w", err)
	}
	if len(queryVec) == 0 {
		// NullEmbedder path — skip Tier 1 entirely.
		return nil, queryVec, nil
	}

	topicEmbeds, err := store.GetAllTopicEmbeddings(ctx)
	if err != nil {
		return nil, queryVec, fmt.Errorf("load topic embeddings: %w", err)
	}
	if len(topicEmbeds) == 0 {
		return nil, queryVec, nil
	}

	activations := make([]types.TopicActivation, 0, len(topicEmbeds))
	for topicID, embedArray := range topicEmbeds {
		if len(embedArray) == 0 {
			continue
		}

		// Max cosine similarity across the representative set — one strong
		// representative is enough to activate the topic.
		var maxScore float64
		for _, rep := range embedArray {
			s := vectors.CosineSimilarity(queryVec, rep)
			if s > maxScore {
				maxScore = s
			}
		}
		if maxScore < activationThreshold {
			continue
		}

		topic, found, err := store.GetTopic(ctx, topicID)
		if err != nil {
			return nil, queryVec, fmt.Errorf("get topic %d: %w", topicID, err)
		}
		if !found {
			// Topic was deleted between GetAllTopicEmbeddings and now — skip it.
			continue
		}

		activations = append(activations, types.TopicActivation{
			Topic: topic,
			Score: maxScore,
		})
	}

	// Descending by score, tie-break by topic ID for determinism.
	sort.Slice(activations, func(i, j int) bool {
		if activations[i].Score != activations[j].Score {
			return activations[i].Score > activations[j].Score
		}
		return activations[i].Topic.ID < activations[j].Topic.ID
	})

	if len(activations) > topicK {
		activations = activations[:topicK]
	}

	return activations, queryVec, nil
}

// SearchActivatedTopics runs hybrid search scoped to each activated topic and
// merges the results. Multi-topic hits sum their (activation × relevance)
// contributions (dendritic integration). Returns results sorted by
// DendriticScore descending (Tier 2).
func SearchActivatedTopics(
	ctx context.Context,
	query string,
	_ []float32, // queryVec kept in signature for symmetry / future semantic scoring tweaks
	activations []types.TopicActivation,
	store Store,
	memoryK int,
	threshold float64,
) ([]types.ScoredMemory, error) {
	if memoryK <= 0 {
		memoryK = defaultMemoryK
	}
	if threshold <= 0 {
		threshold = defaultMemoryThreshold
	}

	accumulated := make(map[int64]*types.ScoredMemory)

	for _, activation := range activations {
		results, err := store.HybridRecallScoped(ctx, query, activation.Topic.ID, memoryK, threshold)
		if err != nil {
			return nil, fmt.Errorf("hybrid recall scoped topic=%d: %w", activation.Topic.ID, err)
		}

		for _, result := range results {
			combined := activation.Score * result.Score
			ts := types.TopicScore{
				TopicName:       activation.Topic.Name,
				ActivationScore: activation.Score,
				MemoryScore:     result.Score,
				Combined:        combined,
			}

			if existing, ok := accumulated[result.Memory.ID]; ok {
				existing.DendriticScore += combined
				existing.TopicScores = append(existing.TopicScores, ts)
				existing.ViaTopic = append(existing.ViaTopic, activation.Topic.Name)
				continue
			}

			mem := result.Memory // copy, we'll mutate only via the pointer
			accumulated[result.Memory.ID] = &types.ScoredMemory{
				Memory:         mem,
				DendriticScore: combined,
				TopicScores:    []types.TopicScore{ts},
				MatchType:      result.MatchType,
				ViaTopic:       []string{activation.Topic.Name},
			}
		}
	}

	out := make([]types.ScoredMemory, 0, len(accumulated))
	for _, sm := range accumulated {
		out = append(out, *sm)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].DendriticScore != out[j].DendriticScore {
			return out[i].DendriticScore > out[j].DendriticScore
		}
		return out[i].Memory.ID < out[j].Memory.ID
	})
	return out, nil
}

// ExpandViaGraph extracts capitalized-noun entities from the Tier 2 memories,
// walks the triplet graph from those seeds, then looks up any source memories
// that weren't already captured by Tier 2. Returns the walked triplets and the
// cross-topic discovered memories (Tier 3).
//
// Cross-topic scoring (spec §7.4): graphDiscount × graphSeedRelevance (0.5).
// Memories whose ID already appears in tier2Results are skipped.
func ExpandViaGraph(
	ctx context.Context,
	tier2Results []types.ScoredMemory,
	store Store,
	hops int,
	graphDiscount float64,
) ([]types.Triplet, []types.ScoredMemory, error) {
	if hops <= 0 {
		hops = defaultHops
	}
	if graphDiscount <= 0 {
		graphDiscount = defaultGraphDiscount
	}

	memories := make([]types.Memory, 0, len(tier2Results))
	alreadyIn := make(map[int64]struct{}, len(tier2Results))
	for _, sm := range tier2Results {
		memories = append(memories, sm.Memory)
		alreadyIn[sm.Memory.ID] = struct{}{}
	}

	entities := ExtractEntities(memories)
	if len(entities) == 0 {
		return nil, nil, nil
	}

	// Cap the triplet walk at a generous limit — results are re-filtered anyway.
	triplets, err := store.WalkGraph(ctx, entities, hops, 50)
	if err != nil {
		return nil, nil, fmt.Errorf("walk graph: %w", err)
	}
	if len(triplets) == 0 {
		return triplets, nil, nil
	}

	graphScore := graphDiscount * graphSeedRelevance
	seenMem := make(map[int64]struct{})
	crossTopic := make([]types.ScoredMemory, 0)

	for _, t := range triplets {
		if t.SourceType != "memory" {
			continue
		}
		memID, err := strconv.ParseInt(strings.TrimSpace(t.SourceID), 10, 64)
		if err != nil || memID <= 0 {
			continue
		}
		if _, ok := alreadyIn[memID]; ok {
			continue
		}
		if _, ok := seenMem[memID]; ok {
			continue
		}
		seenMem[memID] = struct{}{}

		mem, found, err := store.GetMemoryByID(ctx, memID)
		if err != nil || !found {
			continue
		}

		crossTopic = append(crossTopic, types.ScoredMemory{
			Memory:         mem,
			DendriticScore: graphScore,
			MatchType:      "graph",
			ViaTopic:       []string{},
		})
	}

	return triplets, crossTopic, nil
}

// DendriticRecall runs the full three-tier pipeline end to end.
//
// Falls back to unscoped HybridRecall whenever Tier 1 cannot activate topics
// (NullEmbedder, no topics in the registry, or no topic has embeddings yet).
// Graph expansion (Tier 3) always runs against the Tier 2 results, regardless
// of whether the fallback path was taken — the triplet graph is topic-agnostic.
func DendriticRecall(
	ctx context.Context,
	req types.DendriticRecallRequest,
	store Store,
	embedder embeddings.EmbeddingProvider,
) (types.DendriticResult, error) {
	req = applyDefaults(req)

	result := types.DendriticResult{
		Query:           req.Query,
		ActivatedTopics: []types.TopicActivation{},
		Results:         []types.ScoredMemory{},
		GraphTriplets:   []types.Triplet{},
		CrossTopicHits:  []types.ScoredMemory{},
	}

	// Tier 1 — topic routing.
	activations, queryVec, err := RouteTopics(ctx, req.Query, embedder, store, req.TopicK, req.ActivationThreshold)
	if err != nil {
		// Soft-fail: embedder errors shouldn't kill the request, just skip Tier 1.
		activations = nil
		queryVec = nil
	}
	result.ActivatedTopics = activations

	// Tier 2 — within-topic search, or unscoped fallback.
	var tier2 []types.ScoredMemory
	if len(activations) > 0 {
		tier2, err = SearchActivatedTopics(ctx, req.Query, queryVec, activations, store, req.MemoryK, req.Threshold)
		if err != nil {
			return result, err
		}
	} else {
		// Fallback: unscoped hybrid recall with a modest recency boost (match
		// Python Nous defaults — 24h window, +0.1 score bump).
		fallback, err := store.HybridRecall(ctx, req.Query, "", req.MemoryK, req.Threshold, 24.0, 0.1)
		if err != nil {
			return result, fmt.Errorf("hybrid recall fallback: %w", err)
		}
		tier2 = make([]types.ScoredMemory, 0, len(fallback))
		for _, r := range fallback {
			tier2 = append(tier2, types.ScoredMemory{
				Memory:         r.Memory,
				DendriticScore: r.Score,
				TopicScores:    []types.TopicScore{},
				MatchType:      r.MatchType,
				ViaTopic:       []string{},
			})
		}
	}

	// Normalize DendriticScores to [0,1] if any exceed 1 (multi-topic summation).
	var maxScore float64
	for _, sm := range tier2 {
		if sm.DendriticScore > maxScore {
			maxScore = sm.DendriticScore
		}
	}
	if maxScore > 1.0 {
		for i := range tier2 {
			tier2[i].DendriticScore /= maxScore
			for j := range tier2[i].TopicScores {
				// Keep Combined visible; only normalize the headline DendriticScore.
				_ = tier2[i].TopicScores[j]
			}
		}
	}

	// Tier 3 — graph expansion.
	triplets, crossTopic, err := ExpandViaGraph(ctx, tier2, store, req.Hops, req.GraphDiscount)
	if err != nil {
		// Soft-fail: graph errors shouldn't drop Tier 2 results.
		triplets = nil
		crossTopic = nil
	}
	result.GraphTriplets = triplets
	result.CrossTopicHits = crossTopic

	// Merge Tier 2 + cross-topic hits, sort, cap at Limit.
	merged := make([]types.ScoredMemory, 0, len(tier2)+len(crossTopic))
	merged = append(merged, tier2...)
	merged = append(merged, crossTopic...)

	sort.Slice(merged, func(i, j int) bool {
		if merged[i].DendriticScore != merged[j].DendriticScore {
			return merged[i].DendriticScore > merged[j].DendriticScore
		}
		return merged[i].Memory.ID < merged[j].Memory.ID
	})

	if len(merged) > req.Limit {
		merged = merged[:req.Limit]
	}
	result.Results = merged

	return result, nil
}

// entityPattern matches capitalized noun-phrase sequences: single capitalized
// words and multi-word phrases like "Go Memory Store". Same heuristic as the
// Python _heuristic_extract in store.py.
var entityPattern = regexp.MustCompile(`\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b`)

// entityBlacklist is the canonical Appendix B list. Kept lowercase for fast
// case-insensitive lookup.
var entityBlacklist = func() map[string]struct{} {
	raw := []string{
		"This", "It", "That", "Here", "There", "Everything", "Nothing",
		"Something", "Always", "Never", "Each", "Every", "These", "Those",
		"The", "A", "An", "We", "They", "You", "He", "She", "My", "Your",
		"His", "Her", "Its", "Our", "Their", "What", "Which", "Who", "Whom",
		"How", "When", "Where", "Why", "Just", "Also", "Since", "Because",
		"Although", "However", "Therefore", "Thus", "Then", "Now", "Still",
		"And", "But", "Or", "So", "Yet",
	}
	set := make(map[string]struct{}, len(raw))
	for _, w := range raw {
		set[strings.ToLower(w)] = struct{}{}
	}
	return set
}()

// ExtractEntities pulls capitalized noun phrases out of the supplied memory
// content, filters the stopword blacklist, and deduplicates by lowercase form.
// Input order is preserved as much as possible (first-seen wins).
func ExtractEntities(memories []types.Memory) []string {
	seen := make(map[string]struct{})
	out := make([]string, 0)

	for _, m := range memories {
		matches := entityPattern.FindAllStringSubmatch(m.Content, -1)
		for _, match := range matches {
			if len(match) < 2 {
				continue
			}
			candidate := strings.TrimSpace(match[1])
			if len(candidate) < 2 {
				continue
			}
			key := strings.ToLower(candidate)
			if _, bad := entityBlacklist[key]; bad {
				continue
			}
			if _, dupe := seen[key]; dupe {
				continue
			}
			seen[key] = struct{}{}
			out = append(out, candidate)
		}
	}
	return out
}
