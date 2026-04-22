// Package types defines the core type system for the Nous memory service.
//
// All memory/topic/graph/worker shapes in the system flow through this package.
// See the dendritic recall specification (§2) for the full type hierarchy.
package types

import (
	"math"
	"time"
)

// ---------------------------------------------------------------------------
// 2.1 Core Memory Types
// ---------------------------------------------------------------------------

// MemoryTier identifies which storage tier a memory belongs to.
type MemoryTier string

const (
	// TierDirector is Tier 1: curated, embedded, topic-routed memories.
	TierDirector MemoryTier = "director"
	// TierShared is Tier 2: worker shared memories, name-scoped in PostgreSQL.
	TierShared MemoryTier = "shared"
	// TierPrivate is Tier 3: per-worker SQLite shell memories, importance-weighted.
	TierPrivate MemoryTier = "private"
)

// MemoryCategory is the standard taxonomy for memory categorization.
// Custom category strings are also valid — these are recommended defaults.
type MemoryCategory string

const (
	CategoryPreference MemoryCategory = "preference"
	CategoryLesson     MemoryCategory = "lesson"
	CategoryFact       MemoryCategory = "fact"
	CategoryDecision   MemoryCategory = "decision"
	CategoryProject    MemoryCategory = "project"
	CategoryPerson     MemoryCategory = "person"
	CategoryGeneral    MemoryCategory = "general"
)

// Memory is a single memory entry.
//
// The Embedding field is NOT serialized to JSON — embeddings are stored as
// binary blobs in PostgreSQL (see vectors.SerializeVector). JSON consumers
// of this service never need the raw vector.
type Memory struct {
	ID         int64          `json:"id"`
	Content    string         `json:"content"`
	Category   string         `json:"category"`
	Tier       MemoryTier     `json:"tier"`
	Importance float64        `json:"importance"`  // 0.0–1.0, used for Tier 3 retention
	WorkerName string         `json:"worker_name"` // Empty for Tier 1 (director)
	TopicID    *int64         `json:"topic_id"`    // Set for Tier 1 memories with topics
	Metadata   map[string]any `json:"metadata"`
	Embedding  []float32      `json:"-"` // Not serialized to JSON
	CreatedAt  time.Time      `json:"created_at"`
	UpdatedAt  time.Time      `json:"updated_at"`
}

// AgeDays returns fractional days since the memory was created.
func (m *Memory) AgeDays() float64 {
	return time.Since(m.CreatedAt).Hours() / 24.0
}

// SearchResult is a memory paired with a relevance score from a search operation.
//
// MatchType identifies how the memory matched: "semantic" (vector similarity),
// "keyword" (token/ILIKE match), "hybrid" (combined), or "graph" (walked from
// a triplet neighborhood).
type SearchResult struct {
	Memory    Memory  `json:"memory"`
	Score     float64 `json:"score"`      // 0.0–1.0
	MatchType string  `json:"match_type"` // "semantic", "keyword", "hybrid", "graph"
}

// ConversationTurn is a single logged turn from the conversation log.
type ConversationTurn struct {
	ID        int64     `json:"id"`
	Role      string    `json:"role"` // "user", "assistant", "system"
	Content   string    `json:"content"`
	CreatedAt time.Time `json:"created_at"`
}

// WorkerResume is a task-completion record for a named worker.
//
// Populated from the worker_history table. StartedAt and FinishedAt are
// pointers because started_at may be NULL for legacy rows.
type WorkerResume struct {
	TaskID      string     `json:"task_id"`
	Description string     `json:"description"`
	Outcome     string     `json:"outcome"` // "completed", "failed"
	SkillsUsed  string     `json:"skills_used"`
	Summary     string     `json:"summary"`
	StartedAt   *time.Time `json:"started_at"`
	FinishedAt  *time.Time `json:"finished_at"`
}

// WorkerShell represents a worker's private SQLite shell metadata.
//
// The actual shell database lives at {shell_dir}/{worker_name}.db. This
// struct is returned from the HTTP API as a summary, not the full DB contents.
type WorkerShell struct {
	WorkerName       string `json:"worker_name"`
	DBPath           string `json:"db_path"`
	MemoriesCount    int    `json:"memories_count"`
	KnowledgeCount   int    `json:"knowledge_count"`
	InstructionCount int    `json:"instruction_count"`
	TasksCompleted   int    `json:"tasks_completed"`
}

// RetentionPolicy controls importance-weighted pruning for Tier 3 memories.
//
// Memories in PreserveCategories are never pruned. Otherwise the age cutoff
// is selected by importance:
//   - importance < LowThreshold     → LowRetentionDays
//   - importance < MediumThreshold  → MediumRetentionDays
//   - importance >= MediumThreshold → HighRetentionDays
type RetentionPolicy struct {
	LowThreshold        float64  `json:"low_threshold"`         // default 0.3
	MediumThreshold     float64  `json:"medium_threshold"`      // default 0.7
	LowRetentionDays    int      `json:"low_retention_days"`    // default 30
	MediumRetentionDays int      `json:"medium_retention_days"` // default 90
	HighRetentionDays   int      `json:"high_retention_days"`   // default 180
	PreserveCategories  []string `json:"preserve_categories"`   // default ["decision","lesson"]
}

// DefaultRetentionPolicy returns the standard retention policy.
func DefaultRetentionPolicy() RetentionPolicy {
	return RetentionPolicy{
		LowThreshold:        0.3,
		MediumThreshold:     0.7,
		LowRetentionDays:    30,
		MediumRetentionDays: 90,
		HighRetentionDays:   180,
		PreserveCategories:  []string{"decision", "lesson"},
	}
}

// ShouldPrune returns true if the memory should be pruned under this policy.
//
// Category is checked first: if the memory's category is in PreserveCategories,
// it is never pruned regardless of age or importance.
func (rp RetentionPolicy) ShouldPrune(m Memory) bool {
	for _, cat := range rp.PreserveCategories {
		if m.Category == cat {
			return false
		}
	}
	age := m.AgeDays()
	switch {
	case m.Importance < rp.LowThreshold:
		return age > float64(rp.LowRetentionDays)
	case m.Importance < rp.MediumThreshold:
		return age > float64(rp.MediumRetentionDays)
	default:
		return age > float64(rp.HighRetentionDays)
	}
}

// ---------------------------------------------------------------------------
// 2.2 Knowledge Graph Types
// ---------------------------------------------------------------------------

// Triplet is a knowledge graph node: (subject, predicate, object).
//
// SourceType identifies the origin system (e.g. "memory", "conversation",
// "task", "note") and SourceID is the ID within that system — e.g. a triplet
// extracted from memory #123 has SourceType="memory" and SourceID="123".
type Triplet struct {
	ID         int64     `json:"id"`
	Subject    string    `json:"subject"`
	Predicate  string    `json:"predicate"`
	Object     string    `json:"object"`
	SourceType string    `json:"source_type"` // "conversation", "memory", "task", "note", etc.
	SourceID   string    `json:"source_id"`
	Confidence float64   `json:"confidence"` // 0.0–1.0
	CreatedAt  time.Time `json:"created_at"`
}

// GraphContext is the result of a graph-enhanced recall.
//
// Entities holds every entity name encountered during the graph walk —
// useful for downstream display and for subsequent walks.
type GraphContext struct {
	RAGResults      []SearchResult     `json:"rag_results"`
	GraphTriplets   []Triplet          `json:"graph_triplets"`
	DiscoveredTurns []ConversationTurn `json:"discovered_turns"`
	Entities        []string           `json:"entities"`
}

// TripletContext is a lightweight entity-neighbourhood view for downstream
// consumers that don't need the full graph walk trace.
type TripletContext struct {
	Query            string                     `json:"query"`
	RAGResults       []SearchResult             `json:"rag_results"`
	Triplets         []Triplet                  `json:"triplets"`
	Entities         []string                   `json:"entities"`
	EntityPredicates map[string][]PredicatePair `json:"entity_predicates"`
}

// PredicatePair is a (predicate, related_entity) pair used in entity
// neighborhood summaries.
type PredicatePair struct {
	Predicate string `json:"predicate"`
	Related   string `json:"related"`
}

// ---------------------------------------------------------------------------
// 2.3 Dendritic Topic Types
// ---------------------------------------------------------------------------

// TopicSource indicates how a topic was created.
type TopicSource string

const (
	// TopicSourceCurated means the topic was created deliberately by an agent.
	TopicSourceCurated TopicSource = "curated"
	// TopicSourceEmergent means the topic was created automatically because
	// incoming memories did not fit any existing topic.
	TopicSourceEmergent TopicSource = "emergent"
)

// Topic is a first-class semantic bucket for organizing director memories.
//
// Topics are agent-curated (like Legate categories), not auto-clusters. They
// hold a name (slug), a display name, a description that seeds routing, and
// a count of representative embeddings stored in the topic_embeddings table.
type Topic struct {
	ID          int64       `json:"id"`
	Name        string      `json:"name"`         // slug-style, e.g. "golang-patterns"
	DisplayName string      `json:"display_name"` // human-readable, e.g. "Go Patterns"
	Description string      `json:"description"`  // semantic description used for routing
	Source      TopicSource `json:"source"`       // "curated" or "emergent"
	MemoryCount int         `json:"memory_count"` // denormalized count of assigned memories
	EmbedCount  int         `json:"embed_count"`  // number of representative embeddings
	CreatedAt   time.Time   `json:"created_at"`
	UpdatedAt   time.Time   `json:"updated_at"`
}

// TopicEmbedding is one representative embedding in a topic's embedding array.
//
// Topics have N representative embeddings (NOT one centroid) because
// embedding models have chunk-length limits — a topic with 200 memories
// cannot collapse to a single vector without losing semantic coverage.
//
// EmbeddingData is the raw binary (little-endian float32, 4 bytes per dim)
// — the same format as Memory.Embedding when serialized for storage.
type TopicEmbedding struct {
	ID            int64     `json:"id"`
	TopicID       int64     `json:"topic_id"`
	MemoryID      int64     `json:"memory_id"`
	EmbeddingData []byte    `json:"-"` // Raw binary, not JSON-serialized
	EmbedModel    string    `json:"embed_model"`
	CreatedAt     time.Time `json:"created_at"`
}

// TopicActivation is a topic that was activated by the query during Tier 1
// routing, paired with the max cosine-similarity score observed against
// that topic's representative embedding array.
type TopicActivation struct {
	Topic Topic   `json:"topic"`
	Score float64 `json:"score"` // 0.0–1.0
}

// DendriticResult is the full output of the three-tier dendritic recall
// pipeline: activated topics, final ranked memories, graph triplets that
// were walked, and cross-topic memories surfaced via graph expansion.
type DendriticResult struct {
	Query           string            `json:"query"`
	ActivatedTopics []TopicActivation `json:"activated_topics"`
	Results         []ScoredMemory    `json:"results"`
	GraphTriplets   []Triplet         `json:"graph_triplets"`
	CrossTopicHits  []ScoredMemory    `json:"cross_topic_hits"`
}

// ScoredMemory is a memory with its dendritic score and per-topic breakdown.
//
// DendriticScore is the combined score, possibly > 1.0 when the memory was
// reached via multiple activated topics (dendritic integration sums the
// contributions). The final recall step normalises this to [0.0, 1.0] before
// returning to the caller.
//
// ViaTopic lists the topic names that activated this memory, in order of
// first contribution.
type ScoredMemory struct {
	Memory         Memory       `json:"memory"`
	DendriticScore float64      `json:"dendritic_score"`
	TopicScores    []TopicScore `json:"topic_scores"`
	MatchType      string       `json:"match_type"` // "semantic", "keyword", "hybrid", "graph"
	ViaTopic       []string     `json:"via_topic"`
}

// TopicScore is a per-topic contribution to a memory's dendritic score.
//
//	Combined = ActivationScore × MemoryScore
//
// ActivationScore is the Tier 1 topic activation (cosine vs the topic's
// embedding array). MemoryScore is the Tier 2 within-topic relevance.
type TopicScore struct {
	TopicName       string  `json:"topic_name"`
	ActivationScore float64 `json:"activation_score"`
	MemoryScore     float64 `json:"memory_score"`
	Combined        float64 `json:"combined"`
}

// DendriticRecallRequest is the full parameter set for dendritic recall.
// All fields have spec-defined defaults; callers should apply them before
// running the pipeline (see recall package).
type DendriticRecallRequest struct {
	Query               string  `json:"query"`
	TopicK              int     `json:"topic_k"`              // max activated topics, default 5
	ActivationThreshold float64 `json:"activation_threshold"` // min topic score, default 0.3
	MemoryK             int     `json:"memory_k"`             // memories per topic, default 10
	Threshold           float64 `json:"threshold"`            // min memory relevance, default 0.3
	Hops                int     `json:"hops"`                 // graph walk depth, default 2
	GraphDiscount       float64 `json:"graph_discount"`       // Tier 3 score multiplier, default 0.6
	Limit               int     `json:"limit"`                // final result cap, default 20
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// TopicEmbedK computes the number of representative embeddings to keep for
// a topic given its memory count.
//
//	K = min(ceil(sqrt(N)), 50)
//
// Returns 0 when memoryCount <= 0 so callers can short-circuit on empty
// topics.
func TopicEmbedK(memoryCount int) int {
	if memoryCount <= 0 {
		return 0
	}
	k := int(math.Ceil(math.Sqrt(float64(memoryCount))))
	if k > 50 {
		k = 50
	}
	return k
}
