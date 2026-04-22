// Package store implements Nous's PostgreSQL-backed memory store and per-worker
// SQLite shells.
//
// The store exposes three memory tiers:
//
//   - Tier 1 (Director): curated, embedded, topic-routable memories that back
//     the dendritic recall pipeline. See Remember/Recall/HybridRecall.
//   - Tier 2 (Worker Shared): worker-namespaced memory in PostgreSQL with
//     keyword-only search. See WorkerRemember/WorkerRecall/WorkerForget.
//   - Tier 3 (Worker Private): per-worker SQLite databases managed as
//     ShellStore handles. See Shell / ShellFor.
//
// In addition the store owns:
//
//   - The conversation log (LogConversation / GetRecentConversations)
//   - Worker history / resumes (RecordTaskCompletion / GetWorkerResume)
//   - The topic registry (see topics.go)
//   - The knowledge graph triplets table (see graph.go)
//
// All vector math — cosine similarity, MMR, binary (de)serialization — lives
// in internal/vectors and is called from this package. Embeddings are always
// stored as little-endian float32 bytes (4 bytes per dim) so Python Nous and
// Go Nous can share the same database without reformatting.
package store

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/ologos-repos/nous/internal/embeddings"
	"github.com/ologos-repos/nous/internal/types"
	"github.com/ologos-repos/nous/internal/vectors"
)

// Config holds connection and runtime parameters for a MemoryStore.
//
// PostgresURL is required. ShellDir defaults to "./shells" when empty.
// Embedder may be nil — the store then falls back to keyword-only retrieval.
type Config struct {
	PostgresURL   string
	ShellDir      string
	Embedder      embeddings.EmbeddingProvider
	MinPoolConns  int32 // default 2
	MaxPoolConns  int32 // default 10
	RunMigrations bool  // default true when zero value would be surprising; see Connect
}

// MemoryStore is the central interface for all memory operations. It owns the
// PostgreSQL pool, a lazily-populated map of per-worker SQLite shells, and the
// active embedding provider.
//
// MemoryStore is safe for concurrent use. The shells map is protected by an
// internal mutex; the PostgreSQL pool is inherently concurrent via pgxpool.
type MemoryStore struct {
	pool     *pgxpool.Pool
	shellDir string
	embedder embeddings.EmbeddingProvider

	mu     sync.RWMutex
	shells map[string]*ShellStore
}

// Connect creates a new MemoryStore connected to PostgreSQL. When
// cfg.RunMigrations is true (the default for Connect callers) the embedded
// SQL migrations are applied before returning.
//
// Callers should defer Close() to release the pool and all open shell handles.
func Connect(ctx context.Context, cfg Config) (*MemoryStore, error) {
	if strings.TrimSpace(cfg.PostgresURL) == "" {
		return nil, errors.New("store.Connect: PostgresURL is required")
	}
	if cfg.ShellDir == "" {
		cfg.ShellDir = "./shells"
	}
	if cfg.MinPoolConns <= 0 {
		cfg.MinPoolConns = 2
	}
	if cfg.MaxPoolConns <= 0 {
		cfg.MaxPoolConns = 10
	}

	poolCfg, err := pgxpool.ParseConfig(cfg.PostgresURL)
	if err != nil {
		return nil, fmt.Errorf("parse postgres url: %w", err)
	}
	poolCfg.MinConns = cfg.MinPoolConns
	poolCfg.MaxConns = cfg.MaxPoolConns

	pool, err := pgxpool.NewWithConfig(ctx, poolCfg)
	if err != nil {
		return nil, fmt.Errorf("open postgres pool: %w", err)
	}
	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("ping postgres: %w", err)
	}

	s := &MemoryStore{
		pool:     pool,
		shellDir: cfg.ShellDir,
		embedder: cfg.Embedder,
		shells:   make(map[string]*ShellStore),
	}

	if cfg.RunMigrations {
		runner := NewMigrationRunner(pool)
		if _, err := runner.Run(ctx); err != nil {
			pool.Close()
			return nil, fmt.Errorf("run migrations: %w", err)
		}
	}
	return s, nil
}

// Pool returns the underlying pgxpool. Primarily for tests and health checks.
func (s *MemoryStore) Pool() *pgxpool.Pool { return s.pool }

// Embedder returns the active embedding provider, which may be nil.
func (s *MemoryStore) Embedder() embeddings.EmbeddingProvider { return s.embedder }

// ShellDir returns the root directory for per-worker SQLite shells.
func (s *MemoryStore) ShellDir() string { return s.shellDir }

// Close releases the PostgreSQL pool and all open worker shells.
func (s *MemoryStore) Close() error {
	s.mu.Lock()
	for name, sh := range s.shells {
		_ = sh.Close()
		delete(s.shells, name)
	}
	s.mu.Unlock()

	if s.pool != nil {
		s.pool.Close()
	}
	return nil
}

// Ping verifies the PostgreSQL connection is alive.
func (s *MemoryStore) Ping(ctx context.Context) error {
	return s.pool.Ping(ctx)
}

// =============================================================================
// Tier 1 — Director Memory
// =============================================================================

// Remember stores a director memory. If the store has an embedder, the content
// is embedded and stored as little-endian float32 bytes. If topicID is non-nil,
// the memory is assigned to that topic and the topic's representative embedding
// set is recomputed (via UpdateTopicEmbeddings).
func (s *MemoryStore) Remember(
	ctx context.Context,
	content, category string,
	topicID *int64,
	metadata map[string]any,
) (types.Memory, error) {
	if category == "" {
		category = string(types.CategoryGeneral)
	}
	if metadata == nil {
		metadata = map[string]any{}
	}
	metaJSON, err := json.Marshal(metadata)
	if err != nil {
		return types.Memory{}, fmt.Errorf("marshal metadata: %w", err)
	}

	var (
		embedBytes []byte
		embedModel string
	)
	if s.embedder != nil {
		vec, err := s.embedder.Embed(ctx, content)
		if err == nil && len(vec) > 0 {
			embedBytes = vectors.SerializeVector(vec)
			embedModel = s.embedder.ModelName()
		}
	}

	var m types.Memory
	var tid *int64
	row := s.pool.QueryRow(ctx, `
		INSERT INTO director_memory
			(content, category, topic_id, embedding, embedding_model, importance, metadata)
		VALUES ($1, $2, $3, $4, $5, $6, $7)
		RETURNING id, content, category, topic_id, embedding_model, importance, metadata, created_at, updated_at
	`, content, category, topicID, embedBytes, nullableString(embedModel), 0.5, metaJSON)

	var (
		mdJSON     []byte
		modelStr   *string
		importance float64
	)
	if err := row.Scan(&m.ID, &m.Content, &m.Category, &tid, &modelStr, &importance, &mdJSON, &m.CreatedAt, &m.UpdatedAt); err != nil {
		return types.Memory{}, fmt.Errorf("insert director_memory: %w", err)
	}
	m.Tier = types.TierDirector
	m.Importance = importance
	m.TopicID = tid
	if len(mdJSON) > 0 {
		_ = json.Unmarshal(mdJSON, &m.Metadata)
	}
	if m.Metadata == nil {
		m.Metadata = map[string]any{}
	}

	if topicID != nil {
		// Best-effort: keep topic denormalized counts + representative set fresh.
		if err := s.bumpTopicMemoryCount(ctx, *topicID, 1); err != nil {
			// Non-fatal: the memory is already inserted. Surface nothing — the
			// caller would have no meaningful recovery path. Callers that care
			// should call AssignMemoryToTopic explicitly.
			_ = err
		}
		if err := s.UpdateTopicEmbeddings(ctx, *topicID); err != nil {
			_ = err
		}
	}
	return m, nil
}

// Recall runs multi-query hybrid search against director_memory. Currently it
// is a thin wrapper around HybridRecall with conservative defaults that match
// the Python Nous behavior for single-query callers.
func (s *MemoryStore) Recall(
	ctx context.Context,
	query, category string,
	limit int,
	threshold float64,
) ([]types.SearchResult, error) {
	return s.HybridRecall(ctx, query, category, limit, threshold, 0, 0)
}

// HybridRecall combines semantic similarity (via embeddings) and OR-keyword
// search. Each match is scored; when a memory appears in both result sets
// the higher of the two scores is kept. Optional recency boost adds
// recencyBoostValue to memories created within recencyBoostHours.
//
//	final_score = max(semantic_score, keyword_score) + recency_boost
//
// If no embedder is configured, semantic scoring is skipped and the method
// degrades gracefully to keyword-only retrieval.
func (s *MemoryStore) HybridRecall(
	ctx context.Context,
	query, category string,
	limit int,
	threshold, recencyBoostHours, recencyBoostValue float64,
) ([]types.SearchResult, error) {
	return s.hybridRecall(ctx, query, category, nil, limit, threshold, recencyBoostHours, recencyBoostValue)
}

// HybridRecallScoped restricts hybrid recall to memories assigned to a given
// topic. Used in Tier 2 of the dendritic recall pipeline.
func (s *MemoryStore) HybridRecallScoped(
	ctx context.Context,
	query string,
	topicID int64,
	limit int,
	threshold float64,
) ([]types.SearchResult, error) {
	tid := topicID
	return s.hybridRecall(ctx, query, "", &tid, limit, threshold, 0, 0)
}

// hybridRecall is the shared implementation backing Recall / HybridRecall /
// HybridRecallScoped. topicID, when non-nil, restricts the candidate set to a
// single topic.
func (s *MemoryStore) hybridRecall(
	ctx context.Context,
	query, category string,
	topicID *int64,
	limit int,
	threshold, recencyBoostHours, recencyBoostValue float64,
) ([]types.SearchResult, error) {
	if limit <= 0 {
		limit = 20
	}

	// Step 1: optional semantic scoring.
	semanticScores := map[int64]float64{}
	var queryVec []float32
	if s.embedder != nil {
		if v, err := s.embedder.Embed(ctx, query); err == nil && len(v) > 0 {
			queryVec = v
			candidates, err := s.fetchEmbeddedCandidates(ctx, category, topicID)
			if err != nil {
				return nil, err
			}
			for _, c := range candidates {
				sim := vectors.CosineSimilarity(queryVec, c.vec)
				if sim >= threshold {
					semanticScores[c.id] = sim
				}
			}
		}
	}

	// Step 2: OR-keyword scoring.
	keywordScores, err := s.keywordRecall(ctx, query, category, topicID)
	if err != nil {
		return nil, err
	}

	// Step 3: merge — keep max score per ID.
	merged := make(map[int64]struct {
		score     float64
		matchType string
	}, len(semanticScores)+len(keywordScores))
	for id, s := range semanticScores {
		merged[id] = struct {
			score     float64
			matchType string
		}{s, "semantic"}
	}
	for id, s := range keywordScores {
		cur, ok := merged[id]
		switch {
		case !ok:
			merged[id] = struct {
				score     float64
				matchType string
			}{s, "keyword"}
		case s > cur.score:
			merged[id] = struct {
				score     float64
				matchType string
			}{s, "hybrid"}
		default:
			merged[id] = struct {
				score     float64
				matchType string
			}{cur.score, "hybrid"}
		}
	}

	if len(merged) == 0 {
		return nil, nil
	}

	// Step 4: hydrate memories.
	ids := make([]int64, 0, len(merged))
	for id := range merged {
		ids = append(ids, id)
	}
	memoriesByID, err := s.fetchMemoriesByIDs(ctx, ids)
	if err != nil {
		return nil, err
	}

	// Step 5: apply recency boost + assemble results.
	results := make([]types.SearchResult, 0, len(merged))
	now := time.Now()
	for id, hit := range merged {
		m, ok := memoriesByID[id]
		if !ok {
			continue
		}
		score := hit.score
		if recencyBoostHours > 0 && recencyBoostValue > 0 {
			age := now.Sub(m.CreatedAt).Hours()
			if age <= recencyBoostHours {
				score += recencyBoostValue
			}
		}
		results = append(results, types.SearchResult{
			Memory:    m,
			Score:     score,
			MatchType: hit.matchType,
		})
	}

	// Step 6: sort descending and cap.
	sortSearchResults(results)
	if len(results) > limit {
		results = results[:limit]
	}
	return results, nil
}

// Forget deletes a director memory by ID. Returns false if no row matched.
// If the memory was assigned to a topic, the topic's memory_count and
// representative embeddings are refreshed.
func (s *MemoryStore) Forget(ctx context.Context, id int64) (bool, error) {
	// Grab the topic_id before deleting so we can refresh denormalized counts.
	var topicID *int64
	err := s.pool.QueryRow(ctx, `SELECT topic_id FROM director_memory WHERE id=$1`, id).Scan(&topicID)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return false, nil
		}
		return false, fmt.Errorf("lookup memory: %w", err)
	}

	tag, err := s.pool.Exec(ctx, `DELETE FROM director_memory WHERE id=$1`, id)
	if err != nil {
		return false, fmt.Errorf("delete memory: %w", err)
	}
	if tag.RowsAffected() == 0 {
		return false, nil
	}

	if topicID != nil {
		_ = s.bumpTopicMemoryCount(ctx, *topicID, -1)
		_ = s.UpdateTopicEmbeddings(ctx, *topicID)
	}
	return true, nil
}

// GetAllMemories returns director memories, optionally filtered by category.
// Results are ordered by created_at DESC.
func (s *MemoryStore) GetAllMemories(ctx context.Context, category string, limit int) ([]types.Memory, error) {
	if limit <= 0 {
		limit = 100
	}

	var (
		rows pgx.Rows
		err  error
	)
	if strings.TrimSpace(category) == "" {
		rows, err = s.pool.Query(ctx, `
			SELECT id, content, category, topic_id, embedding_model, importance, metadata, created_at, updated_at
			FROM director_memory
			ORDER BY created_at DESC
			LIMIT $1
		`, limit)
	} else {
		rows, err = s.pool.Query(ctx, `
			SELECT id, content, category, topic_id, embedding_model, importance, metadata, created_at, updated_at
			FROM director_memory
			WHERE category = $1
			ORDER BY created_at DESC
			LIMIT $2
		`, category, limit)
	}
	if err != nil {
		return nil, fmt.Errorf("query director_memory: %w", err)
	}
	defer rows.Close()

	out := []types.Memory{}
	for rows.Next() {
		m, err := scanDirectorMemory(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, m)
	}
	return out, rows.Err()
}

// GetMemoryByID fetches a single director memory. Returns (zero, false, nil)
// if not found.
func (s *MemoryStore) GetMemoryByID(ctx context.Context, id int64) (types.Memory, bool, error) {
	row := s.pool.QueryRow(ctx, `
		SELECT id, content, category, topic_id, embedding_model, importance, metadata, created_at, updated_at
		FROM director_memory
		WHERE id = $1
	`, id)
	m, err := scanDirectorMemory(row)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return types.Memory{}, false, nil
		}
		return types.Memory{}, false, err
	}
	return m, true, nil
}

// =============================================================================
// Tier 2 — Worker Shared Memory
// =============================================================================

// WorkerRemember stores a worker-scoped shared memory in PostgreSQL.
func (s *MemoryStore) WorkerRemember(
	ctx context.Context,
	workerName, content, category string,
	metadata map[string]any,
) (types.Memory, error) {
	if strings.TrimSpace(workerName) == "" {
		return types.Memory{}, errors.New("worker name required")
	}
	if category == "" {
		category = string(types.CategoryGeneral)
	}
	if metadata == nil {
		metadata = map[string]any{}
	}
	metaJSON, err := json.Marshal(metadata)
	if err != nil {
		return types.Memory{}, fmt.Errorf("marshal metadata: %w", err)
	}

	var (
		m      types.Memory
		mdJSON []byte
	)
	err = s.pool.QueryRow(ctx, `
		INSERT INTO worker_memory (worker_name, content, category, metadata)
		VALUES ($1, $2, $3, $4)
		RETURNING id, worker_name, content, category, metadata, created_at
	`, workerName, content, category, metaJSON).
		Scan(&m.ID, &m.WorkerName, &m.Content, &m.Category, &mdJSON, &m.CreatedAt)
	if err != nil {
		return types.Memory{}, fmt.Errorf("insert worker_memory: %w", err)
	}
	m.Tier = types.TierShared
	m.UpdatedAt = m.CreatedAt
	if len(mdJSON) > 0 {
		_ = json.Unmarshal(mdJSON, &m.Metadata)
	}
	if m.Metadata == nil {
		m.Metadata = map[string]any{}
	}
	return m, nil
}

// WorkerRecall searches a worker's shared memories by OR-keyword.
// Keyword match score is a simple term hit count normalized to [0,1].
func (s *MemoryStore) WorkerRecall(
	ctx context.Context,
	workerName, query string,
	limit int,
) ([]types.SearchResult, error) {
	if strings.TrimSpace(workerName) == "" {
		return nil, errors.New("worker name required")
	}
	if limit <= 0 {
		limit = 20
	}

	terms := keywordTerms(query)
	if len(terms) == 0 {
		// Return most-recent without scoring when the query has no usable terms.
		rows, err := s.pool.Query(ctx, `
			SELECT id, worker_name, content, category, metadata, created_at
			FROM worker_memory
			WHERE worker_name = $1
			ORDER BY created_at DESC
			LIMIT $2
		`, workerName, limit)
		if err != nil {
			return nil, fmt.Errorf("query worker_memory: %w", err)
		}
		defer rows.Close()
		out := []types.SearchResult{}
		for rows.Next() {
			m, err := scanWorkerMemory(rows)
			if err != nil {
				return nil, err
			}
			out = append(out, types.SearchResult{Memory: m, Score: 0.0, MatchType: "recent"})
		}
		return out, rows.Err()
	}

	// OR-keyword search against content using ILIKE.
	ilikes, args := buildILikeClause("content", terms, 2)
	args = append([]any{workerName}, args...)
	sql := fmt.Sprintf(`
		SELECT id, worker_name, content, category, metadata, created_at
		FROM worker_memory
		WHERE worker_name = $1 AND (%s)
		ORDER BY created_at DESC
		LIMIT %d
	`, ilikes, limit)
	rows, err := s.pool.Query(ctx, sql, args...)
	if err != nil {
		return nil, fmt.Errorf("query worker_memory: %w", err)
	}
	defer rows.Close()

	out := []types.SearchResult{}
	for rows.Next() {
		m, err := scanWorkerMemory(rows)
		if err != nil {
			return nil, err
		}
		score := termHitRatio(m.Content, terms)
		out = append(out, types.SearchResult{Memory: m, Score: score, MatchType: "keyword"})
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	sortSearchResults(out)
	return out, nil
}

// WorkerForget deletes a worker-scoped memory, enforcing name-scoping. Returns
// false if no row matched (either missing or owned by a different worker).
func (s *MemoryStore) WorkerForget(ctx context.Context, workerName string, id int64) (bool, error) {
	if strings.TrimSpace(workerName) == "" {
		return false, errors.New("worker name required")
	}
	tag, err := s.pool.Exec(ctx, `
		DELETE FROM worker_memory WHERE id = $1 AND worker_name = $2
	`, id, workerName)
	if err != nil {
		return false, fmt.Errorf("delete worker_memory: %w", err)
	}
	return tag.RowsAffected() > 0, nil
}

// =============================================================================
// Conversation Log
// =============================================================================

// LogConversation appends a turn to the conversation log.
func (s *MemoryStore) LogConversation(ctx context.Context, role, content string) (types.ConversationTurn, error) {
	var t types.ConversationTurn
	err := s.pool.QueryRow(ctx, `
		INSERT INTO conversations (role, content) VALUES ($1, $2)
		RETURNING id, role, content, created_at
	`, role, content).Scan(&t.ID, &t.Role, &t.Content, &t.CreatedAt)
	if err != nil {
		return types.ConversationTurn{}, fmt.Errorf("insert conversation: %w", err)
	}
	return t, nil
}

// GetRecentConversations returns turns in chronological order. When
// hoursWindow > 0, only turns within that window are returned.
func (s *MemoryStore) GetRecentConversations(
	ctx context.Context,
	limit int,
	hoursWindow float64,
) ([]types.ConversationTurn, error) {
	if limit <= 0 {
		limit = 50
	}

	var (
		rows pgx.Rows
		err  error
	)
	if hoursWindow > 0 {
		cutoff := time.Now().Add(-time.Duration(hoursWindow * float64(time.Hour)))
		rows, err = s.pool.Query(ctx, `
			SELECT id, role, content, created_at
			FROM conversations
			WHERE created_at >= $1
			ORDER BY created_at DESC
			LIMIT $2
		`, cutoff, limit)
	} else {
		rows, err = s.pool.Query(ctx, `
			SELECT id, role, content, created_at
			FROM conversations
			ORDER BY created_at DESC
			LIMIT $1
		`, limit)
	}
	if err != nil {
		return nil, fmt.Errorf("query conversations: %w", err)
	}
	defer rows.Close()

	out := []types.ConversationTurn{}
	for rows.Next() {
		var t types.ConversationTurn
		if err := rows.Scan(&t.ID, &t.Role, &t.Content, &t.CreatedAt); err != nil {
			return nil, err
		}
		out = append(out, t)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	// Reverse to chronological order (oldest first).
	for i, j := 0, len(out)-1; i < j; i, j = i+1, j-1 {
		out[i], out[j] = out[j], out[i]
	}
	return out, nil
}

// =============================================================================
// Worker History (Resume)
// =============================================================================

// RecordTaskCompletion appends a resume entry for a named worker.
func (s *MemoryStore) RecordTaskCompletion(
	ctx context.Context,
	workerName, taskID, description, outcome, skillsUsed, summary string,
	startedAt *time.Time,
) (types.WorkerResume, error) {
	if strings.TrimSpace(workerName) == "" {
		return types.WorkerResume{}, errors.New("worker name required")
	}

	var (
		r          types.WorkerResume
		finishedAt time.Time
	)
	var startOut *time.Time
	err := s.pool.QueryRow(ctx, `
		INSERT INTO worker_history
			(worker_name, task_id, task_description, outcome, skills_used, summary, started_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7)
		RETURNING task_id, task_description, outcome, skills_used, summary, started_at, finished_at
	`, workerName, taskID, description, outcome, skillsUsed, summary, startedAt).
		Scan(&r.TaskID, &r.Description, &r.Outcome, &r.SkillsUsed, &r.Summary, &startOut, &finishedAt)
	if err != nil {
		return types.WorkerResume{}, fmt.Errorf("insert worker_history: %w", err)
	}
	r.StartedAt = startOut
	fa := finishedAt
	r.FinishedAt = &fa
	return r, nil
}

// GetWorkerResume returns a worker's task history, most-recent first.
func (s *MemoryStore) GetWorkerResume(ctx context.Context, workerName string, limit int) ([]types.WorkerResume, error) {
	if strings.TrimSpace(workerName) == "" {
		return nil, errors.New("worker name required")
	}
	if limit <= 0 {
		limit = 20
	}

	rows, err := s.pool.Query(ctx, `
		SELECT task_id, task_description, outcome, skills_used, summary, started_at, finished_at
		FROM worker_history
		WHERE worker_name = $1
		ORDER BY finished_at DESC NULLS LAST
		LIMIT $2
	`, workerName, limit)
	if err != nil {
		return nil, fmt.Errorf("query worker_history: %w", err)
	}
	defer rows.Close()

	out := []types.WorkerResume{}
	for rows.Next() {
		var r types.WorkerResume
		var started, finished *time.Time
		var taskID, desc, outcome, skills, summary *string
		if err := rows.Scan(&taskID, &desc, &outcome, &skills, &summary, &started, &finished); err != nil {
			return nil, err
		}
		r.TaskID = deref(taskID)
		r.Description = deref(desc)
		r.Outcome = deref(outcome)
		r.SkillsUsed = deref(skills)
		r.Summary = deref(summary)
		r.StartedAt = started
		r.FinishedAt = finished
		out = append(out, r)
	}
	return out, rows.Err()
}

// =============================================================================
// Helpers
// =============================================================================

// candidate is an internal id/vector pair used during hybrid recall.
type candidate struct {
	id  int64
	vec []float32
}

// fetchEmbeddedCandidates returns all memory IDs + deserialized embeddings
// filtered by optional category and topic. Used to compute semantic scores.
func (s *MemoryStore) fetchEmbeddedCandidates(
	ctx context.Context,
	category string,
	topicID *int64,
) ([]candidate, error) {
	var (
		rows pgx.Rows
		err  error
	)
	switch {
	case topicID != nil:
		rows, err = s.pool.Query(ctx, `
			SELECT id, embedding FROM director_memory
			WHERE embedding IS NOT NULL AND topic_id = $1
		`, *topicID)
	case strings.TrimSpace(category) != "":
		rows, err = s.pool.Query(ctx, `
			SELECT id, embedding FROM director_memory
			WHERE embedding IS NOT NULL AND category = $1
		`, category)
	default:
		rows, err = s.pool.Query(ctx, `
			SELECT id, embedding FROM director_memory
			WHERE embedding IS NOT NULL
		`)
	}
	if err != nil {
		return nil, fmt.Errorf("fetch embedded candidates: %w", err)
	}
	defer rows.Close()

	out := []candidate{}
	for rows.Next() {
		var id int64
		var b []byte
		if err := rows.Scan(&id, &b); err != nil {
			return nil, err
		}
		vec := vectors.DeserializeVector(b)
		if vec == nil {
			continue
		}
		out = append(out, candidate{id: id, vec: vec})
	}
	return out, rows.Err()
}

// keywordRecall runs an OR-keyword ILIKE search. Returns a map of memory ID
// to normalized hit ratio in [0.0, 1.0].
func (s *MemoryStore) keywordRecall(
	ctx context.Context,
	query, category string,
	topicID *int64,
) (map[int64]float64, error) {
	terms := keywordTerms(query)
	if len(terms) == 0 {
		return map[int64]float64{}, nil
	}

	// Build WHERE clauses
	conds := make([]string, 0, 3)
	args := make([]any, 0, len(terms)+2)

	ilikes, ilikeArgs := buildILikeClause("content", terms, len(args)+1)
	conds = append(conds, "("+ilikes+")")
	args = append(args, ilikeArgs...)

	if strings.TrimSpace(category) != "" {
		args = append(args, category)
		conds = append(conds, fmt.Sprintf("category = $%d", len(args)))
	}
	if topicID != nil {
		args = append(args, *topicID)
		conds = append(conds, fmt.Sprintf("topic_id = $%d", len(args)))
	}

	sql := `SELECT id, content FROM director_memory`
	if len(conds) > 0 {
		sql += " WHERE " + strings.Join(conds, " AND ")
	}
	sql += " LIMIT 500"

	rows, err := s.pool.Query(ctx, sql, args...)
	if err != nil {
		return nil, fmt.Errorf("keyword recall query: %w", err)
	}
	defer rows.Close()

	out := map[int64]float64{}
	for rows.Next() {
		var id int64
		var content string
		if err := rows.Scan(&id, &content); err != nil {
			return nil, err
		}
		out[id] = termHitRatio(content, terms)
	}
	return out, rows.Err()
}

// fetchMemoriesByIDs hydrates a batch of director memories by ID.
func (s *MemoryStore) fetchMemoriesByIDs(ctx context.Context, ids []int64) (map[int64]types.Memory, error) {
	if len(ids) == 0 {
		return map[int64]types.Memory{}, nil
	}
	rows, err := s.pool.Query(ctx, `
		SELECT id, content, category, topic_id, embedding_model, importance, metadata, created_at, updated_at
		FROM director_memory
		WHERE id = ANY($1::bigint[])
	`, ids)
	if err != nil {
		return nil, fmt.Errorf("fetch memories by ids: %w", err)
	}
	defer rows.Close()

	out := make(map[int64]types.Memory, len(ids))
	for rows.Next() {
		m, err := scanDirectorMemory(rows)
		if err != nil {
			return nil, err
		}
		out[m.ID] = m
	}
	return out, rows.Err()
}

// scanDirectorMemory scans a single director_memory row into types.Memory.
// Accepts both pgx.Row (QueryRow) and pgx.Rows (Query).
func scanDirectorMemory(row pgx.Row) (types.Memory, error) {
	var (
		m         types.Memory
		tid       *int64
		modelStr  *string
		mdJSON    []byte
		impVal    float64
	)
	err := row.Scan(&m.ID, &m.Content, &m.Category, &tid, &modelStr, &impVal, &mdJSON, &m.CreatedAt, &m.UpdatedAt)
	if err != nil {
		return types.Memory{}, err
	}
	m.Tier = types.TierDirector
	m.Importance = impVal
	m.TopicID = tid
	if len(mdJSON) > 0 {
		_ = json.Unmarshal(mdJSON, &m.Metadata)
	}
	if m.Metadata == nil {
		m.Metadata = map[string]any{}
	}
	return m, nil
}

// scanWorkerMemory scans a worker_memory row into types.Memory.
func scanWorkerMemory(row pgx.Row) (types.Memory, error) {
	var m types.Memory
	var mdJSON []byte
	if err := row.Scan(&m.ID, &m.WorkerName, &m.Content, &m.Category, &mdJSON, &m.CreatedAt); err != nil {
		return types.Memory{}, err
	}
	m.Tier = types.TierShared
	m.UpdatedAt = m.CreatedAt
	m.Importance = 0.5
	if len(mdJSON) > 0 {
		_ = json.Unmarshal(mdJSON, &m.Metadata)
	}
	if m.Metadata == nil {
		m.Metadata = map[string]any{}
	}
	return m, nil
}

// stopWords is the small default English stop list used to filter keyword terms.
// Matches spec Appendix A.
var stopWords = map[string]struct{}{
	"a": {}, "an": {}, "and": {}, "are": {}, "as": {}, "at": {}, "be": {}, "by": {},
	"for": {}, "from": {}, "has": {}, "have": {}, "he": {}, "in": {}, "is": {},
	"it": {}, "its": {}, "of": {}, "on": {}, "or": {}, "that": {}, "the": {},
	"to": {}, "was": {}, "were": {}, "will": {}, "with": {}, "i": {}, "you": {},
	"your": {}, "my": {}, "me": {}, "we": {}, "our": {}, "they": {}, "them": {},
	"their": {}, "this": {}, "these": {}, "those": {}, "but": {}, "not": {},
	"no": {}, "if": {}, "then": {}, "so": {}, "do": {}, "did": {}, "does": {},
	"been": {}, "being": {}, "am": {}, "what": {}, "how": {}, "why": {}, "when": {},
	"where": {}, "who": {},
}

// keywordTerms splits a query into lowercase tokens, filters stop words and
// short tokens, and deduplicates.
func keywordTerms(q string) []string {
	if strings.TrimSpace(q) == "" {
		return nil
	}
	// Split on non-letter/digit.
	fields := strings.FieldsFunc(q, func(r rune) bool {
		return !isWordRune(r)
	})
	seen := map[string]struct{}{}
	out := make([]string, 0, len(fields))
	for _, f := range fields {
		t := strings.ToLower(strings.TrimSpace(f))
		if len(t) < 3 {
			continue
		}
		if _, stop := stopWords[t]; stop {
			continue
		}
		if _, ok := seen[t]; ok {
			continue
		}
		seen[t] = struct{}{}
		out = append(out, t)
	}
	return out
}

func isWordRune(r rune) bool {
	return (r >= 'a' && r <= 'z') ||
		(r >= 'A' && r <= 'Z') ||
		(r >= '0' && r <= '9') ||
		r == '_' || r == '-'
}

// buildILikeClause returns a SQL fragment and args for an OR-keyword ILIKE
// search. argOffset tells us where the first $N placeholder begins.
func buildILikeClause(column string, terms []string, argOffset int) (string, []any) {
	parts := make([]string, 0, len(terms))
	args := make([]any, 0, len(terms))
	for i, t := range terms {
		parts = append(parts, fmt.Sprintf("%s ILIKE $%d", column, argOffset+i))
		args = append(args, "%"+t+"%")
	}
	return strings.Join(parts, " OR "), args
}

// termHitRatio returns the fraction of terms present in content (case-insensitive).
// Range: [0, 1]. Used as the keyword score for hybrid merging.
func termHitRatio(content string, terms []string) float64 {
	if len(terms) == 0 {
		return 0.0
	}
	low := strings.ToLower(content)
	hits := 0
	for _, t := range terms {
		if strings.Contains(low, t) {
			hits++
		}
	}
	return float64(hits) / float64(len(terms))
}

// sortSearchResults sorts in place by Score descending.
func sortSearchResults(rs []types.SearchResult) {
	// Simple insertion sort — result sets are small (≤ limit).
	for i := 1; i < len(rs); i++ {
		j := i
		for j > 0 && rs[j-1].Score < rs[j].Score {
			rs[j-1], rs[j] = rs[j], rs[j-1]
			j--
		}
	}
}

// nullableString converts an empty string to nil for use with nullable SQL cols.
func nullableString(s string) any {
	if s == "" {
		return nil
	}
	return s
}

// deref safely dereferences a *string, returning "" for nil.
func deref(s *string) string {
	if s == nil {
		return ""
	}
	return *s
}
