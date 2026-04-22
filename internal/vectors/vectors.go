// Package vectors provides pure-Go vector math for the Nous memory service.
//
// All operations are implemented without CGo and without external
// dependencies so the resulting binary has no C runtime bindings. The
// corpus sizes the memory service is designed for (single-digit thousands
// of memories per topic, a handful of topics) are well within the range
// where brute-force cosine similarity outperforms the fixed cost of
// calling out to a vector-search library.
//
// The binary vector format is IEEE-754 float32, little-endian, 4 bytes per
// dimension. This matches Python's struct.pack("<Nf", *vec) so embeddings
// written by the legacy Python Nous library remain readable by this
// service and vice-versa.
package vectors

import (
	"encoding/binary"
	"math"
	"sort"
)

// CosineSimilarity computes cosine similarity between two float32 vectors.
//
// Returns 0.0 if either vector is nil, empty, has a different length, or
// has zero magnitude. The result is in [-1.0, 1.0]; for L2-normalized
// embeddings the range narrows to [0.0, 1.0].
func CosineSimilarity(a, b []float32) float64 {
	if len(a) == 0 || len(b) == 0 || len(a) != len(b) {
		return 0.0
	}
	var dot, magA, magB float64
	for i := range a {
		av, bv := float64(a[i]), float64(b[i])
		dot += av * bv
		magA += av * av
		magB += bv * bv
	}
	if magA == 0.0 || magB == 0.0 {
		return 0.0
	}
	return dot / (math.Sqrt(magA) * math.Sqrt(magB))
}

// SerializeVector encodes a float32 slice as little-endian binary.
//
// Output length is len(v) * 4 bytes. Returns nil for an empty input so
// callers can persist a NULL column value instead of an empty blob.
func SerializeVector(v []float32) []byte {
	if len(v) == 0 {
		return nil
	}
	buf := make([]byte, len(v)*4)
	for i, f := range v {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(f))
	}
	return buf
}

// DeserializeVector decodes little-endian binary bytes to a float32 slice.
//
// Returns nil if data is nil or empty. Any trailing bytes that do not form
// a complete float32 (i.e. len(data) % 4 != 0) are dropped.
func DeserializeVector(data []byte) []float32 {
	if len(data) == 0 {
		return nil
	}
	n := len(data) / 4
	if n == 0 {
		return nil
	}
	v := make([]float32, n)
	for i := range v {
		bits := binary.LittleEndian.Uint32(data[i*4:])
		v[i] = math.Float32frombits(bits)
	}
	return v
}

// VectorCandidate is a single (id, vector) pair fed into TopKSimilar.
//
// Callers typically build a []VectorCandidate from a DB query result
// (e.g. rows of (id, embedding BYTEA)) where the vector is deserialized
// via DeserializeVector.
type VectorCandidate struct {
	ID     int64
	Vector []float32
}

// CandidateScore pairs a candidate ID with its similarity score.
type CandidateScore struct {
	ID    int64
	Score float64
}

// TopKSimilar finds the K most similar candidates to the query by cosine
// similarity.
//
// Candidates whose score is below threshold are filtered out before
// ranking. Results are returned sorted by score descending. If k <= 0
// no truncation is applied (a common pattern when the caller wants all
// above-threshold hits).
func TopKSimilar(query []float32, candidates []VectorCandidate, k int, threshold float64) []CandidateScore {
	if len(candidates) == 0 {
		return nil
	}
	scores := make([]CandidateScore, 0, len(candidates))
	for _, c := range candidates {
		s := CosineSimilarity(query, c.Vector)
		if s >= threshold {
			scores = append(scores, CandidateScore{ID: c.ID, Score: s})
		}
	}
	sort.Slice(scores, func(i, j int) bool { return scores[i].Score > scores[j].Score })
	if k > 0 && len(scores) > k {
		scores = scores[:k]
	}
	return scores
}

// MMR selects K diverse embeddings from candidates using Maximal Marginal
// Relevance.
//
// Algorithm:
//  1. Precompute each candidate's cosine similarity to the query.
//  2. On every iteration, pick the candidate that maximises
//     MMR_score = λ · sim(candidate, query) − (1 − λ) · max_i sim(candidate, selected_i)
//  3. Continue until K candidates have been selected or the pool is empty.
//
// λ controls the relevance-vs-diversity tradeoff: λ = 1.0 reduces to pure
// relevance (equivalent to TopKSimilar), λ = 0.0 reduces to pure
// diversity. For topic representative sets the specification recommends
// λ = 0.5.
//
// The first selection has no redundancy term (no already-selected items
// to compare against), so it is always the candidate most similar to the
// query — matching the "seed" step from the spec.
//
// Returns indices into the candidates slice in selection order. Empty
// candidate slices or k <= 0 yield a nil result.
func MMR(query []float32, candidates [][]float32, k int, lambda float64) []int {
	if len(candidates) == 0 || k <= 0 {
		return nil
	}
	if k > len(candidates) {
		k = len(candidates)
	}

	// Precompute similarity of each candidate to the query.
	queryScores := make([]float64, len(candidates))
	for i, c := range candidates {
		queryScores[i] = CosineSimilarity(query, c)
	}

	selected := make([]int, 0, k)
	// Boolean marker avoids O(n) scans of a "remaining" slice on every iteration.
	taken := make([]bool, len(candidates))

	for len(selected) < k {
		bestIdx := -1
		bestScore := math.Inf(-1)

		for ci := range candidates {
			if taken[ci] {
				continue
			}
			rel := lambda * queryScores[ci]

			// Redundancy term: max similarity to already-selected.
			var maxSim float64
			for _, si := range selected {
				sim := CosineSimilarity(candidates[ci], candidates[si])
				if sim > maxSim {
					maxSim = sim
				}
			}
			red := (1 - lambda) * maxSim

			score := rel - red
			if score > bestScore {
				bestScore = score
				bestIdx = ci
			}
		}

		if bestIdx < 0 {
			break
		}
		selected = append(selected, bestIdx)
		taken[bestIdx] = true
	}

	return selected
}
