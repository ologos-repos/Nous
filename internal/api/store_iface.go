package api

// HandlerStore is the minimal interface the HTTP handlers and recall pipeline
// require from the storage layer. *store.MemoryStore satisfies this interface.
//
// Keeping it narrow (no pgxpool, no SQLite, no embeddings) makes the api
// package easy to test with lightweight mocks — the store_iface_test.go file
// verifies *store.MemoryStore still satisfies it at compile time.
//
// Includes all methods needed by:
//   - handlers directly (Remember, Forget, ListTopics, …)
//   - recall.DendriticRecall (GetAllTopicEmbeddings, HybridRecallScoped, …)

import (
	"context"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/ologos-repos/nous/internal/types"
)

// HandlerStore is the storage interface used by the HTTP layer.
type HandlerStore interface {
	// Connectivity
	Ping(ctx context.Context) error
	// Pool returns the underlying pgxpool.Pool for stat reporting; may be nil in
	// non-Postgres deployments (e.g. tests). Handlers guard all Pool() calls.
	Pool() *pgxpool.Pool

	// Director memory
	Remember(ctx context.Context, content, category string, topicID *int64, metadata map[string]any) (types.Memory, error)
	HybridRecall(ctx context.Context, query, category string, limit int, threshold, recencyBoostHours, recencyBoostValue float64) ([]types.SearchResult, error)
	HybridRecallScoped(ctx context.Context, query string, topicID int64, limit int, threshold float64) ([]types.SearchResult, error)
	Forget(ctx context.Context, id int64) (bool, error)
	GetMemoryByID(ctx context.Context, id int64) (types.Memory, bool, error)

	// Topics
	ListTopics(ctx context.Context, source types.TopicSource) ([]types.Topic, error)
	CreateTopic(ctx context.Context, name, displayName, description string, source types.TopicSource) (types.Topic, error)
	GetTopic(ctx context.Context, id int64) (types.Topic, bool, error)
	AssignMemoryToTopic(ctx context.Context, memoryID, topicID int64) error
	UpdateTopicEmbeddings(ctx context.Context, topicID int64) error
	GetAllTopicEmbeddings(ctx context.Context) (map[int64][][]float32, error)

	// Graph
	GetTripletsBySource(ctx context.Context, sourceType, sourceID string) ([]types.Triplet, error)
	StoreTriplets(ctx context.Context, triplets [][3]string, sourceType, sourceID string, confidence float64) (int, error)
	WalkGraph(ctx context.Context, entities []string, hops, limit int) ([]types.Triplet, error)

	// Worker memory (shared PostgreSQL tier)
	WorkerRemember(ctx context.Context, workerName, content, category string, metadata map[string]any) (types.Memory, error)
	WorkerRecall(ctx context.Context, workerName, query string, limit int) ([]types.SearchResult, error)
	WorkerForget(ctx context.Context, workerName string, id int64) (bool, error)
	RecordTaskCompletion(ctx context.Context, workerName, taskID, description, outcome, skillsUsed, summary string, startedAt *time.Time) (types.WorkerResume, error)
	GetWorkerResume(ctx context.Context, workerName string, limit int) ([]types.WorkerResume, error)

	// Conversation
	LogConversation(ctx context.Context, role, content string) (types.ConversationTurn, error)
	GetRecentConversations(ctx context.Context, limit int, hoursWindow float64) ([]types.ConversationTurn, error)
}
