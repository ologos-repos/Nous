package store

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/jackc/pgx/v5"

	"github.com/ologos-repos/nous/internal/types"
	"github.com/ologos-repos/nous/internal/vectors"
)

// =============================================================================
// Topic Registry CRUD
// =============================================================================

// CreateTopic creates a new topic. Returns an error if the slug is already taken.
// If source is empty, defaults to TopicSourceCurated.
func (s *MemoryStore) CreateTopic(
	ctx context.Context,
	name, displayName, description string,
	source types.TopicSource,
) (types.Topic, error) {
	name = strings.TrimSpace(name)
	if name == "" {
		return types.Topic{}, errors.New("topic name required")
	}
	if displayName == "" {
		displayName = name
	}
	if source == "" {
		source = types.TopicSourceCurated
	}

	var t types.Topic
	err := s.pool.QueryRow(ctx, `
		INSERT INTO topic_registry (name, display_name, description, source)
		VALUES ($1, $2, $3, $4)
		RETURNING id, name, display_name, description, source, memory_count, embed_count, created_at, updated_at
	`, name, displayName, description, string(source)).
		Scan(&t.ID, &t.Name, &t.DisplayName, &t.Description, &t.Source,
			&t.MemoryCount, &t.EmbedCount, &t.CreatedAt, &t.UpdatedAt)
	if err != nil {
		return types.Topic{}, fmt.Errorf("insert topic_registry: %w", err)
	}
	return t, nil
}

// GetTopic returns a topic by ID. (zero, false, nil) when missing.
func (s *MemoryStore) GetTopic(ctx context.Context, id int64) (types.Topic, bool, error) {
	row := s.pool.QueryRow(ctx, `
		SELECT id, name, display_name, description, source, memory_count, embed_count, created_at, updated_at
		FROM topic_registry WHERE id = $1
	`, id)
	t, err := scanTopic(row)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return types.Topic{}, false, nil
		}
		return types.Topic{}, false, err
	}
	return t, true, nil
}

// GetTopicByName returns a topic by slug name. (zero, false, nil) when missing.
func (s *MemoryStore) GetTopicByName(ctx context.Context, name string) (types.Topic, bool, error) {
	row := s.pool.QueryRow(ctx, `
		SELECT id, name, display_name, description, source, memory_count, embed_count, created_at, updated_at
		FROM topic_registry WHERE name = $1
	`, name)
	t, err := scanTopic(row)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return types.Topic{}, false, nil
		}
		return types.Topic{}, false, err
	}
	return t, true, nil
}

// ListTopics returns all topics, optionally filtered by source.
// Pass an empty source ("") for no filter.
func (s *MemoryStore) ListTopics(ctx context.Context, source types.TopicSource) ([]types.Topic, error) {
	var (
		rows pgx.Rows
		err  error
	)
	if source == "" {
		rows, err = s.pool.Query(ctx, `
			SELECT id, name, display_name, description, source, memory_count, embed_count, created_at, updated_at
			FROM topic_registry
			ORDER BY name ASC
		`)
	} else {
		rows, err = s.pool.Query(ctx, `
			SELECT id, name, display_name, description, source, memory_count, embed_count, created_at, updated_at
			FROM topic_registry
			WHERE source = $1
			ORDER BY name ASC
		`, string(source))
	}
	if err != nil {
		return nil, fmt.Errorf("list topics: %w", err)
	}
	defer rows.Close()

	out := []types.Topic{}
	for rows.Next() {
		t, err := scanTopic(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, t)
	}
	return out, rows.Err()
}

// UpdateTopic updates a topic's display_name and description. Returns the
// updated row. Returns (zero, false, nil) when the topic does not exist.
func (s *MemoryStore) UpdateTopic(
	ctx context.Context,
	id int64,
	displayName, description string,
) (types.Topic, bool, error) {
	row := s.pool.QueryRow(ctx, `
		UPDATE topic_registry
		SET display_name = $2,
		    description  = $3,
		    updated_at   = NOW()
		WHERE id = $1
		RETURNING id, name, display_name, description, source, memory_count, embed_count, created_at, updated_at
	`, id, displayName, description)
	t, err := scanTopic(row)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return types.Topic{}, false, nil
		}
		return types.Topic{}, false, fmt.Errorf("update topic: %w", err)
	}
	return t, true, nil
}

// DeleteTopic removes a topic. Associated director_memory rows have topic_id
// set to NULL via ON DELETE SET NULL; topic_embeddings rows cascade-delete.
func (s *MemoryStore) DeleteTopic(ctx context.Context, id int64) error {
	_, err := s.pool.Exec(ctx, `DELETE FROM topic_registry WHERE id = $1`, id)
	if err != nil {
		return fmt.Errorf("delete topic: %w", err)
	}
	return nil
}

// AssignMemoryToTopic assigns a director memory to a topic and recomputes
// the topic's representative embedding set.
//
// The operation runs in a transaction so topic_id + denormalized counts stay
// consistent; UpdateTopicEmbeddings runs after commit because it may touch
// many rows and does not need to be atomic with the assignment.
func (s *MemoryStore) AssignMemoryToTopic(ctx context.Context, memoryID, topicID int64) error {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("begin tx: %w", err)
	}
	defer func() { _ = tx.Rollback(ctx) }()

	// Capture previous topic_id so we can adjust both old + new counts.
	var prev *int64
	err = tx.QueryRow(ctx, `SELECT topic_id FROM director_memory WHERE id = $1`, memoryID).Scan(&prev)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return fmt.Errorf("memory %d not found", memoryID)
		}
		return fmt.Errorf("lookup memory: %w", err)
	}

	if prev != nil && *prev == topicID {
		return tx.Commit(ctx)
	}

	_, err = tx.Exec(ctx, `
		UPDATE director_memory
		SET topic_id = $2, updated_at = NOW()
		WHERE id = $1
	`, memoryID, topicID)
	if err != nil {
		return fmt.Errorf("assign memory topic: %w", err)
	}

	// Denormalized counters.
	if prev != nil {
		if _, err := tx.Exec(ctx, `
			UPDATE topic_registry SET memory_count = GREATEST(memory_count - 1, 0), updated_at = NOW()
			WHERE id = $1
		`, *prev); err != nil {
			return fmt.Errorf("decrement old topic count: %w", err)
		}
	}
	if _, err := tx.Exec(ctx, `
		UPDATE topic_registry SET memory_count = memory_count + 1, updated_at = NOW()
		WHERE id = $1
	`, topicID); err != nil {
		return fmt.Errorf("increment new topic count: %w", err)
	}

	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("commit tx: %w", err)
	}

	// Refresh embeddings for the new topic. If the old topic had embeddings,
	// refresh it too so its representative set doesn't reference a memory
	// that no longer belongs.
	if prev != nil {
		_ = s.UpdateTopicEmbeddings(ctx, *prev)
	}
	return s.UpdateTopicEmbeddings(ctx, topicID)
}

// GetTopicMemories returns memories currently assigned to the given topic,
// ordered by created_at DESC.
func (s *MemoryStore) GetTopicMemories(ctx context.Context, topicID int64, limit int) ([]types.Memory, error) {
	if limit <= 0 {
		limit = 100
	}
	rows, err := s.pool.Query(ctx, `
		SELECT id, content, category, topic_id, embedding_model, importance, metadata, created_at, updated_at
		FROM director_memory
		WHERE topic_id = $1
		ORDER BY created_at DESC
		LIMIT $2
	`, topicID, limit)
	if err != nil {
		return nil, fmt.Errorf("get topic memories: %w", err)
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

// =============================================================================
// Topic Embedding Array Management
// =============================================================================

// UpdateTopicEmbeddings recomputes the representative embedding set for a topic.
//
// Algorithm (spec §6.2):
//  1. Fetch all director_memory rows assigned to this topic that have embeddings.
//  2. Compute K = min(ceil(sqrt(N)), 50) via types.TopicEmbedK.
//  3. Use MMR to select K diverse embeddings (λ=0.5). The "query" vector is
//     the mean of all candidate embeddings (approximate centroid).
//  4. Replace the topic's topic_embeddings rows atomically inside a transaction.
//  5. Update topic_registry.embed_count = len(selected).
//
// Runs synchronously and is safe to call from the hot path for small topics.
// For large topics (N >> 100) callers should invoke it in a goroutine.
func (s *MemoryStore) UpdateTopicEmbeddings(ctx context.Context, topicID int64) error {
	rows, err := s.pool.Query(ctx, `
		SELECT id, embedding, embedding_model
		FROM director_memory
		WHERE topic_id = $1 AND embedding IS NOT NULL
	`, topicID)
	if err != nil {
		return fmt.Errorf("fetch topic embeddings source: %w", err)
	}
	type candidateMem struct {
		id     int64
		vec    []float32
		model  string
	}
	cands := []candidateMem{}
	for rows.Next() {
		var (
			id        int64
			b         []byte
			modelStr  *string
		)
		if err := rows.Scan(&id, &b, &modelStr); err != nil {
			rows.Close()
			return fmt.Errorf("scan candidate: %w", err)
		}
		v := vectors.DeserializeVector(b)
		if v == nil {
			continue
		}
		cands = append(cands, candidateMem{id: id, vec: v, model: deref(modelStr)})
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return err
	}

	n := len(cands)
	// Always refresh counts + wipe stale embeddings even when n == 0.
	k := types.TopicEmbedK(n)

	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("begin tx: %w", err)
	}
	defer func() { _ = tx.Rollback(ctx) }()

	if _, err := tx.Exec(ctx, `DELETE FROM topic_embeddings WHERE topic_id = $1`, topicID); err != nil {
		return fmt.Errorf("clear old topic_embeddings: %w", err)
	}

	if n > 0 && k > 0 {
		// Centroid of the candidate set (dim-matched, ignore mismatched vectors).
		dim := len(cands[0].vec)
		mean := make([]float32, dim)
		kept := 0
		for _, c := range cands {
			if len(c.vec) != dim {
				continue
			}
			for i, v := range c.vec {
				mean[i] += v
			}
			kept++
		}
		if kept > 0 {
			inv := float32(1.0 / float64(kept))
			for i := range mean {
				mean[i] *= inv
			}
		}

		vecs := make([][]float32, len(cands))
		for i, c := range cands {
			vecs[i] = c.vec
		}
		selected := vectors.MMR(mean, vecs, k, 0.5)

		for _, idx := range selected {
			c := cands[idx]
			_, err := tx.Exec(ctx, `
				INSERT INTO topic_embeddings (topic_id, memory_id, embedding, embed_model)
				VALUES ($1, $2, $3, $4)
				ON CONFLICT (topic_id, memory_id) DO UPDATE
					SET embedding   = EXCLUDED.embedding,
					    embed_model = EXCLUDED.embed_model
			`, topicID, c.id, vectors.SerializeVector(c.vec), c.model)
			if err != nil {
				return fmt.Errorf("insert topic_embedding: %w", err)
			}
		}
	}

	// embed_count = number of representatives actually stored (not k, since
	// MMR may stop early if candidates exhausted).
	var embedCount int
	if err := tx.QueryRow(ctx, `
		SELECT COUNT(*) FROM topic_embeddings WHERE topic_id = $1
	`, topicID).Scan(&embedCount); err != nil {
		return fmt.Errorf("count topic_embeddings: %w", err)
	}

	if _, err := tx.Exec(ctx, `
		UPDATE topic_registry
		SET embed_count  = $2,
		    memory_count = (SELECT COUNT(*) FROM director_memory WHERE topic_id = $1),
		    updated_at   = NOW()
		WHERE id = $1
	`, topicID, embedCount); err != nil {
		return fmt.Errorf("update topic counts: %w", err)
	}

	return tx.Commit(ctx)
}

// GetTopicEmbeddings returns the representative embedding array for a topic
// as []float32 vectors.
func (s *MemoryStore) GetTopicEmbeddings(ctx context.Context, topicID int64) ([][]float32, error) {
	rows, err := s.pool.Query(ctx, `
		SELECT embedding FROM topic_embeddings WHERE topic_id = $1
	`, topicID)
	if err != nil {
		return nil, fmt.Errorf("query topic embeddings: %w", err)
	}
	defer rows.Close()

	out := [][]float32{}
	for rows.Next() {
		var b []byte
		if err := rows.Scan(&b); err != nil {
			return nil, err
		}
		v := vectors.DeserializeVector(b)
		if v == nil {
			continue
		}
		out = append(out, v)
	}
	return out, rows.Err()
}

// GetAllTopicEmbeddings fetches embedding arrays for every topic in a single
// query. The result map is keyed by topic_id. Suitable for Tier 1 routing
// where the query must be compared against every topic in one scan.
func (s *MemoryStore) GetAllTopicEmbeddings(ctx context.Context) (map[int64][][]float32, error) {
	rows, err := s.pool.Query(ctx, `
		SELECT topic_id, embedding FROM topic_embeddings
	`)
	if err != nil {
		return nil, fmt.Errorf("query all topic embeddings: %w", err)
	}
	defer rows.Close()

	out := map[int64][][]float32{}
	for rows.Next() {
		var (
			topicID int64
			b       []byte
		)
		if err := rows.Scan(&topicID, &b); err != nil {
			return nil, err
		}
		v := vectors.DeserializeVector(b)
		if v == nil {
			continue
		}
		out[topicID] = append(out[topicID], v)
	}
	return out, rows.Err()
}

// bumpTopicMemoryCount adjusts topic_registry.memory_count by delta.
// Never goes negative.
func (s *MemoryStore) bumpTopicMemoryCount(ctx context.Context, topicID int64, delta int) error {
	_, err := s.pool.Exec(ctx, `
		UPDATE topic_registry
		SET memory_count = GREATEST(memory_count + $2, 0),
		    updated_at   = NOW()
		WHERE id = $1
	`, topicID, delta)
	if err != nil {
		return fmt.Errorf("bump topic memory_count: %w", err)
	}
	return nil
}

// scanTopic hydrates a topic row.
func scanTopic(row pgx.Row) (types.Topic, error) {
	var t types.Topic
	var source string
	if err := row.Scan(
		&t.ID, &t.Name, &t.DisplayName, &t.Description, &source,
		&t.MemoryCount, &t.EmbedCount, &t.CreatedAt, &t.UpdatedAt,
	); err != nil {
		return types.Topic{}, err
	}
	t.Source = types.TopicSource(source)
	return t, nil
}
