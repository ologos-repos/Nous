package api

// mockStore is a configurable stub that satisfies HandlerStore without
// requiring a live PostgreSQL instance. Each method has a corresponding
// "On*" field that the test sets before calling the handler. Un-configured
// methods return safe zero values so tests only need to configure what they
// actually care about.

import (
	"context"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/ologos-repos/nous/internal/types"
)

type mockStore struct {
	// Connectivity
	pingErr error

	// Director memory
	rememberResult types.Memory
	rememberErr    error

	hybridRecallResult []types.SearchResult
	hybridRecallErr    error

	hybridRecallScopedResult []types.SearchResult
	hybridRecallScopedErr    error

	forgetResult bool
	forgetErr    error

	getMemoryByIDResult types.Memory
	getMemoryByIDFound  bool
	getMemoryByIDErr    error

	// Topics
	listTopicsResult []types.Topic
	listTopicsErr    error

	createTopicResult types.Topic
	createTopicErr    error

	getTopicResult types.Topic
	getTopicFound  bool
	getTopicErr    error

	assignMemoryToTopicErr    error
	updateTopicEmbeddingsErr  error
	getAllTopicEmbeddingsResult map[int64][][]float32
	getAllTopicEmbeddingsErr    error

	// Graph
	getTripletsBySourceResult []types.Triplet
	getTripletsBySourceErr    error

	storeTripletsResult int
	storeTripletsErr    error

	walkGraphResult []types.Triplet
	walkGraphErr    error

	// Worker memory
	workerRememberResult types.Memory
	workerRememberErr    error

	workerRecallResult []types.SearchResult
	workerRecallErr    error

	workerForgetResult bool
	workerForgetErr    error

	recordTaskCompletionResult types.WorkerResume
	recordTaskCompletionErr    error

	getWorkerResumeResult []types.WorkerResume
	getWorkerResumeErr    error

	// Conversation
	logConversationResult types.ConversationTurn
	logConversationErr    error

	getRecentConversationsResult []types.ConversationTurn
	getRecentConversationsErr    error
}

// Verify at compile time that *mockStore implements HandlerStore.
var _ HandlerStore = (*mockStore)(nil)

func newMockStore() *mockStore { return &mockStore{} }

// ---- Connectivity ----

func (m *mockStore) Ping(_ context.Context) error { return m.pingErr }

func (m *mockStore) Pool() *pgxpool.Pool { return nil } // tests don't need real pool

// ---- Director memory ----

func (m *mockStore) Remember(_ context.Context, _, _ string, _ *int64, _ map[string]any) (types.Memory, error) {
	return m.rememberResult, m.rememberErr
}

func (m *mockStore) HybridRecall(_ context.Context, _, _ string, _ int, _, _, _ float64) ([]types.SearchResult, error) {
	return m.hybridRecallResult, m.hybridRecallErr
}

func (m *mockStore) HybridRecallScoped(_ context.Context, _ string, _ int64, _ int, _ float64) ([]types.SearchResult, error) {
	return m.hybridRecallScopedResult, m.hybridRecallScopedErr
}

func (m *mockStore) Forget(_ context.Context, _ int64) (bool, error) {
	return m.forgetResult, m.forgetErr
}

func (m *mockStore) GetMemoryByID(_ context.Context, _ int64) (types.Memory, bool, error) {
	return m.getMemoryByIDResult, m.getMemoryByIDFound, m.getMemoryByIDErr
}

// ---- Topics ----

func (m *mockStore) ListTopics(_ context.Context, _ types.TopicSource) ([]types.Topic, error) {
	return m.listTopicsResult, m.listTopicsErr
}

func (m *mockStore) CreateTopic(_ context.Context, _, _, _ string, _ types.TopicSource) (types.Topic, error) {
	return m.createTopicResult, m.createTopicErr
}

func (m *mockStore) GetTopic(_ context.Context, _ int64) (types.Topic, bool, error) {
	return m.getTopicResult, m.getTopicFound, m.getTopicErr
}

func (m *mockStore) AssignMemoryToTopic(_ context.Context, _, _ int64) error {
	return m.assignMemoryToTopicErr
}

func (m *mockStore) UpdateTopicEmbeddings(_ context.Context, _ int64) error {
	return m.updateTopicEmbeddingsErr
}

func (m *mockStore) GetAllTopicEmbeddings(_ context.Context) (map[int64][][]float32, error) {
	return m.getAllTopicEmbeddingsResult, m.getAllTopicEmbeddingsErr
}

// ---- Graph ----

func (m *mockStore) GetTripletsBySource(_ context.Context, _, _ string) ([]types.Triplet, error) {
	return m.getTripletsBySourceResult, m.getTripletsBySourceErr
}

func (m *mockStore) StoreTriplets(_ context.Context, _ [][3]string, _, _ string, _ float64) (int, error) {
	return m.storeTripletsResult, m.storeTripletsErr
}

func (m *mockStore) WalkGraph(_ context.Context, _ []string, _, _ int) ([]types.Triplet, error) {
	return m.walkGraphResult, m.walkGraphErr
}

// ---- Worker memory ----

func (m *mockStore) WorkerRemember(_ context.Context, _, _, _ string, _ map[string]any) (types.Memory, error) {
	return m.workerRememberResult, m.workerRememberErr
}

func (m *mockStore) WorkerRecall(_ context.Context, _, _ string, _ int) ([]types.SearchResult, error) {
	return m.workerRecallResult, m.workerRecallErr
}

func (m *mockStore) WorkerForget(_ context.Context, _ string, _ int64) (bool, error) {
	return m.workerForgetResult, m.workerForgetErr
}

func (m *mockStore) RecordTaskCompletion(_ context.Context, _, _, _, _, _, _ string, _ *time.Time) (types.WorkerResume, error) {
	return m.recordTaskCompletionResult, m.recordTaskCompletionErr
}

func (m *mockStore) GetWorkerResume(_ context.Context, _ string, _ int) ([]types.WorkerResume, error) {
	return m.getWorkerResumeResult, m.getWorkerResumeErr
}

// ---- Conversation ----

func (m *mockStore) LogConversation(_ context.Context, _, _ string) (types.ConversationTurn, error) {
	return m.logConversationResult, m.logConversationErr
}

func (m *mockStore) GetRecentConversations(_ context.Context, _ int, _ float64) ([]types.ConversationTurn, error) {
	return m.getRecentConversationsResult, m.getRecentConversationsErr
}
