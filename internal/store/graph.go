package store

import (
	"context"
	"fmt"
	"strings"

	"github.com/ologos-repos/nous/internal/types"
)

// =============================================================================
// Knowledge Graph (Triplets)
// =============================================================================
//
// The graph tier stores (subject, predicate, object) triplets tagged with a
// source (e.g. "conversation:<id>", "memory:<id>"). WalkGraph performs a
// breadth-first traversal from a seed entity set, used by the dendritic recall
// pipeline to discover memories in non-activated topics.

// StoreTriplets batch-inserts (subject, predicate, object) triplets. Entries
// where any of subject/predicate/object are empty after trimming are silently
// skipped. Returns the count actually inserted.
//
// confidence defaults to 1.0 when non-positive.
func (s *MemoryStore) StoreTriplets(
	ctx context.Context,
	triplets [][3]string,
	sourceType, sourceID string,
	confidence float64,
) (int, error) {
	if len(triplets) == 0 {
		return 0, nil
	}
	if confidence <= 0 {
		confidence = 1.0
	}
	if sourceType == "" {
		sourceType = "conversation"
	}

	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return 0, fmt.Errorf("begin tx: %w", err)
	}
	defer func() { _ = tx.Rollback(ctx) }()

	inserted := 0
	for _, t := range triplets {
		subj := strings.TrimSpace(t[0])
		pred := strings.TrimSpace(t[1])
		obj := strings.TrimSpace(t[2])
		if subj == "" || pred == "" || obj == "" {
			continue
		}
		_, err := tx.Exec(ctx, `
			INSERT INTO triplets (subject, predicate, object, source_type, source_id, confidence)
			VALUES ($1, $2, $3, $4, $5, $6)
		`, subj, pred, obj, sourceType, sourceID, confidence)
		if err != nil {
			return inserted, fmt.Errorf("insert triplet (%s,%s,%s): %w", subj, pred, obj, err)
		}
		inserted++
	}

	if err := tx.Commit(ctx); err != nil {
		return inserted, fmt.Errorf("commit triplets: %w", err)
	}
	return inserted, nil
}

// WalkGraph performs a BFS from seed entities, collecting triplets up to N
// hops. Uses `subject = ANY($1::text[]) OR object = ANY($1::text[])` for
// efficient multi-entity lookup per hop.
//
// Algorithm (spec §9.2):
//  1. frontier = entities, seenIDs = ∅, seenEntities = lowercase(entities)
//  2. For each hop up to hops:
//     a. Fetch triplets where subject OR object ∈ frontier.
//     b. Skip triplets already in seenIDs.
//     c. Record each new triplet; collect its previously-unseen endpoints
//        into next_frontier.
//     d. frontier = next_frontier. Stop if empty or if len(allTriplets) ≥ limit.
//  3. Return allTriplets truncated to limit (original chronological order).
//
// Returns triplets sorted by created_at ASC (oldest first), which mirrors the
// Python Nous behavior and keeps graph narratives readable top-to-bottom.
func (s *MemoryStore) WalkGraph(
	ctx context.Context,
	entities []string,
	hops, limit int,
) ([]types.Triplet, error) {
	if hops <= 0 {
		hops = 1
	}
	if limit <= 0 {
		limit = 50
	}
	if len(entities) == 0 {
		return nil, nil
	}

	// Normalize + dedupe the seed frontier. Seen entities are tracked
	// lowercase so case variants don't cause re-walks.
	frontier := dedupeTrim(entities)
	if len(frontier) == 0 {
		return nil, nil
	}

	seenEntities := make(map[string]struct{}, len(frontier))
	for _, e := range frontier {
		seenEntities[strings.ToLower(e)] = struct{}{}
	}
	seenIDs := make(map[int64]struct{}, limit)
	collected := make([]types.Triplet, 0, limit)

	for hop := 0; hop < hops; hop++ {
		if len(frontier) == 0 {
			break
		}

		rows, err := s.pool.Query(ctx, `
			SELECT id, subject, predicate, object, source_type, source_id, confidence, created_at
			FROM triplets
			WHERE subject = ANY($1::text[]) OR object = ANY($1::text[])
			ORDER BY created_at ASC
		`, frontier)
		if err != nil {
			return nil, fmt.Errorf("walk graph hop %d: %w", hop, err)
		}

		nextFrontier := make([]string, 0, 16)
		for rows.Next() {
			var t types.Triplet
			if err := rows.Scan(&t.ID, &t.Subject, &t.Predicate, &t.Object,
				&t.SourceType, &t.SourceID, &t.Confidence, &t.CreatedAt); err != nil {
				rows.Close()
				return nil, err
			}
			if _, dup := seenIDs[t.ID]; dup {
				continue
			}
			seenIDs[t.ID] = struct{}{}
			collected = append(collected, t)

			// Expand frontier with new endpoints.
			for _, e := range [2]string{t.Subject, t.Object} {
				lower := strings.ToLower(strings.TrimSpace(e))
				if lower == "" {
					continue
				}
				if _, ok := seenEntities[lower]; ok {
					continue
				}
				seenEntities[lower] = struct{}{}
				nextFrontier = append(nextFrontier, e)
			}

			if len(collected) >= limit {
				break
			}
		}
		rows.Close()
		if err := rows.Err(); err != nil {
			return nil, err
		}

		if len(collected) >= limit {
			break
		}
		frontier = nextFrontier
	}

	if len(collected) > limit {
		collected = collected[:limit]
	}
	return collected, nil
}

// GetTripletsBySource returns all triplets attached to a given (sourceType,
// sourceID). Useful when rebuilding memory-specific neighborhoods or showing
// the graph contributions of a single memory.
func (s *MemoryStore) GetTripletsBySource(
	ctx context.Context,
	sourceType, sourceID string,
) ([]types.Triplet, error) {
	rows, err := s.pool.Query(ctx, `
		SELECT id, subject, predicate, object, source_type, source_id, confidence, created_at
		FROM triplets
		WHERE source_type = $1 AND source_id = $2
		ORDER BY created_at ASC
	`, sourceType, sourceID)
	if err != nil {
		return nil, fmt.Errorf("get triplets by source: %w", err)
	}
	defer rows.Close()

	out := []types.Triplet{}
	for rows.Next() {
		var t types.Triplet
		if err := rows.Scan(&t.ID, &t.Subject, &t.Predicate, &t.Object,
			&t.SourceType, &t.SourceID, &t.Confidence, &t.CreatedAt); err != nil {
			return nil, err
		}
		out = append(out, t)
	}
	return out, rows.Err()
}

// GetEntityNeighborhood walks the graph starting from a single entity. Thin
// convenience wrapper around WalkGraph.
func (s *MemoryStore) GetEntityNeighborhood(
	ctx context.Context,
	entity string,
	hops int,
) ([]types.Triplet, error) {
	return s.WalkGraph(ctx, []string{entity}, hops, 100)
}

// dedupeTrim returns the input list with each entry trimmed, empties dropped,
// and duplicates (case-insensitive) collapsed while preserving first-seen order.
func dedupeTrim(in []string) []string {
	seen := make(map[string]struct{}, len(in))
	out := make([]string, 0, len(in))
	for _, e := range in {
		e = strings.TrimSpace(e)
		if e == "" {
			continue
		}
		k := strings.ToLower(e)
		if _, ok := seen[k]; ok {
			continue
		}
		seen[k] = struct{}{}
		out = append(out, e)
	}
	return out
}
