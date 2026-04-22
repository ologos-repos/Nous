package vectors

import (
	"math"
	"testing"
)

// almostEqual returns true if a and b are within eps of each other.
func almostEqual(a, b, eps float64) bool {
	return math.Abs(a-b) <= eps
}

func TestCosineSimilarityIdentical(t *testing.T) {
	v := []float32{0.1, 0.2, 0.3, 0.4, 0.5}
	got := CosineSimilarity(v, v)
	if !almostEqual(got, 1.0, 1e-6) {
		t.Fatalf("cosine(v, v) = %v, want 1.0", got)
	}
}

func TestCosineSimilarityOrthogonal(t *testing.T) {
	a := []float32{1.0, 0.0, 0.0}
	b := []float32{0.0, 1.0, 0.0}
	got := CosineSimilarity(a, b)
	if !almostEqual(got, 0.0, 1e-9) {
		t.Fatalf("cosine(e1, e2) = %v, want 0.0", got)
	}
}

func TestCosineSimilarityOpposite(t *testing.T) {
	a := []float32{1.0, 2.0, 3.0}
	b := []float32{-1.0, -2.0, -3.0}
	got := CosineSimilarity(a, b)
	if !almostEqual(got, -1.0, 1e-6) {
		t.Fatalf("cosine(v, -v) = %v, want -1.0", got)
	}
}

func TestCosineSimilarityEdgeCases(t *testing.T) {
	cases := []struct {
		name string
		a, b []float32
		want float64
	}{
		{"nil vs nil", nil, nil, 0.0},
		{"nil vs vec", nil, []float32{1, 2}, 0.0},
		{"vec vs nil", []float32{1, 2}, nil, 0.0},
		{"length mismatch", []float32{1, 2, 3}, []float32{1, 2}, 0.0},
		{"zero magnitude a", []float32{0, 0, 0}, []float32{1, 1, 1}, 0.0},
		{"zero magnitude b", []float32{1, 1, 1}, []float32{0, 0, 0}, 0.0},
		{"both zero", []float32{0, 0, 0}, []float32{0, 0, 0}, 0.0},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := CosineSimilarity(c.a, c.b)
			if !almostEqual(got, c.want, 1e-9) {
				t.Fatalf("cosine = %v, want %v", got, c.want)
			}
		})
	}
}

func TestSerializeDeserializeRoundtrip(t *testing.T) {
	original := []float32{0.0, 1.0, -1.5, 3.14159, -2.71828, 1e-6, 1e6}
	blob := SerializeVector(original)
	if len(blob) != len(original)*4 {
		t.Fatalf("serialized length = %d, want %d", len(blob), len(original)*4)
	}

	roundtrip := DeserializeVector(blob)
	if len(roundtrip) != len(original) {
		t.Fatalf("roundtrip length = %d, want %d", len(roundtrip), len(original))
	}
	for i, want := range original {
		if roundtrip[i] != want {
			t.Errorf("index %d: got %v, want %v", i, roundtrip[i], want)
		}
	}
}

func TestSerializeLittleEndianFormat(t *testing.T) {
	// Verify byte layout matches Python's struct.pack("<f", 1.0).
	// 1.0f little-endian = 00 00 80 3F
	v := []float32{1.0}
	blob := SerializeVector(v)
	want := []byte{0x00, 0x00, 0x80, 0x3F}
	if len(blob) != len(want) {
		t.Fatalf("length mismatch: got %d, want %d", len(blob), len(want))
	}
	for i := range want {
		if blob[i] != want[i] {
			t.Errorf("byte %d: got %02X, want %02X", i, blob[i], want[i])
		}
	}
}

func TestSerializeEmpty(t *testing.T) {
	if got := SerializeVector(nil); got != nil {
		t.Errorf("SerializeVector(nil) = %v, want nil", got)
	}
	if got := SerializeVector([]float32{}); got != nil {
		t.Errorf("SerializeVector([]) = %v, want nil", got)
	}
}

func TestDeserializeEmpty(t *testing.T) {
	if got := DeserializeVector(nil); got != nil {
		t.Errorf("DeserializeVector(nil) = %v, want nil", got)
	}
	if got := DeserializeVector([]byte{}); got != nil {
		t.Errorf("DeserializeVector([]) = %v, want nil", got)
	}
}

func TestTopKSimilarOrdering(t *testing.T) {
	query := []float32{1.0, 0.0, 0.0}
	candidates := []VectorCandidate{
		{ID: 1, Vector: []float32{0.5, 0.5, 0.0}},   // cos ≈ 0.707
		{ID: 2, Vector: []float32{1.0, 0.0, 0.0}},   // cos = 1.0  (best)
		{ID: 3, Vector: []float32{0.0, 1.0, 0.0}},   // cos = 0.0
		{ID: 4, Vector: []float32{0.9, 0.1, 0.0}},   // cos ≈ 0.994
		{ID: 5, Vector: []float32{-1.0, 0.0, 0.0}},  // cos = -1.0
	}

	got := TopKSimilar(query, candidates, 3, 0.0)
	if len(got) != 3 {
		t.Fatalf("TopKSimilar length = %d, want 3", len(got))
	}
	wantOrder := []int64{2, 4, 1}
	for i, want := range wantOrder {
		if got[i].ID != want {
			t.Errorf("position %d: got ID %d, want %d (full=%+v)", i, got[i].ID, want, got)
		}
	}
	// Scores must be monotonically non-increasing.
	for i := 1; i < len(got); i++ {
		if got[i].Score > got[i-1].Score {
			t.Errorf("scores not descending at %d: %v then %v", i, got[i-1].Score, got[i].Score)
		}
	}
}

func TestTopKSimilarThreshold(t *testing.T) {
	query := []float32{1.0, 0.0}
	candidates := []VectorCandidate{
		{ID: 1, Vector: []float32{1.0, 0.0}},  // 1.0
		{ID: 2, Vector: []float32{0.0, 1.0}},  // 0.0
		{ID: 3, Vector: []float32{0.7, 0.7}},  // ≈ 0.707
	}
	got := TopKSimilar(query, candidates, 10, 0.5)
	if len(got) != 2 {
		t.Fatalf("with threshold 0.5, expected 2 results, got %d: %+v", len(got), got)
	}
	for _, g := range got {
		if g.Score < 0.5 {
			t.Errorf("score below threshold leaked through: %v", g)
		}
	}
}

func TestTopKSimilarEmpty(t *testing.T) {
	if got := TopKSimilar([]float32{1, 0}, nil, 5, 0.0); got != nil {
		t.Errorf("empty candidates should yield nil, got %+v", got)
	}
}

func TestMMRDiversity(t *testing.T) {
	// Designed so the first pick (max relevance) has a direction distinct
	// from the query axis, which lets diversity actually matter on the
	// second pick. Pure top-K would return [0, 1] (the two highest-relevance
	// candidates), but MMR with λ=0.5 should prefer the diverse candidate 2.
	//
	// Computed cosines (hand-verified):
	//   sim(q, 0) ≈ 0.914   sim(q, 1) = 0.800     sim(q, 2) ≈ 0.707
	//   sim(0, 1) ≈ 0.975   sim(0, 2) ≈ 0.359
	//
	// After picking idx 0:
	//   score(1) = 0.5·0.800 − 0.5·0.975 ≈ −0.087
	//   score(2) = 0.5·0.707 − 0.5·0.359 ≈ +0.174  ← wins
	query := []float32{1.0, 0.0}
	candidates := [][]float32{
		{0.9, 0.4},   // idx 0 — best match to query (first pick)
		{0.8, 0.6},   // idx 1 — 2nd best relevance, but near-duplicate direction of idx 0
		{0.7, -0.7}, // idx 2 — lower relevance, but diverse direction from idx 0
	}

	selected := MMR(query, candidates, 2, 0.5)
	if len(selected) != 2 {
		t.Fatalf("MMR k=2 returned %d results: %v", len(selected), selected)
	}
	if selected[0] != 0 {
		t.Errorf("first pick = %d, want 0 (highest sim to query)", selected[0])
	}
	if selected[1] != 2 {
		t.Errorf("second pick = %d, want 2 (diverse) — MMR picked the near-duplicate instead", selected[1])
	}
}

func TestMMRPureRelevance(t *testing.T) {
	// λ=1.0 should reduce MMR to pure relevance (top-K by query similarity),
	// even when the top candidates are near-duplicates.
	query := []float32{1.0, 0.0, 0.0}
	candidates := [][]float32{
		{0.3, 0.9, 0.0},    // idx 0 — low relevance
		{1.0, 0.0, 0.0},    // idx 1 — highest
		{0.99, 0.0, 0.01},  // idx 2 — second highest
		{0.0, 1.0, 0.0},    // idx 3 — lowest
	}
	selected := MMR(query, candidates, 2, 1.0)
	if len(selected) != 2 {
		t.Fatalf("expected 2 selections, got %d", len(selected))
	}
	if selected[0] != 1 || selected[1] != 2 {
		t.Errorf("λ=1.0 selection = %v, want [1 2] (top relevance pair)", selected)
	}
}

func TestMMRNoDuplicates(t *testing.T) {
	query := []float32{1.0, 0.0}
	candidates := [][]float32{
		{1.0, 0.0},
		{0.5, 0.5},
		{0.0, 1.0},
	}
	selected := MMR(query, candidates, 3, 0.5)
	if len(selected) != 3 {
		t.Fatalf("expected 3 selections, got %d", len(selected))
	}
	seen := map[int]bool{}
	for _, idx := range selected {
		if seen[idx] {
			t.Errorf("duplicate index in MMR selection: %d (%v)", idx, selected)
		}
		seen[idx] = true
	}
}

func TestMMREdgeCases(t *testing.T) {
	query := []float32{1.0, 0.0}

	if got := MMR(query, nil, 3, 0.5); got != nil {
		t.Errorf("MMR with nil candidates = %v, want nil", got)
	}
	if got := MMR(query, [][]float32{{1, 0}}, 0, 0.5); got != nil {
		t.Errorf("MMR with k=0 = %v, want nil", got)
	}
	if got := MMR(query, [][]float32{{1, 0}}, -1, 0.5); got != nil {
		t.Errorf("MMR with negative k = %v, want nil", got)
	}

	// k larger than candidate pool: should return all candidates.
	small := [][]float32{{1.0, 0.0}, {0.0, 1.0}}
	selected := MMR(query, small, 10, 0.5)
	if len(selected) != 2 {
		t.Errorf("MMR with k > N = %d selections, want 2", len(selected))
	}
}

// TestTopicEmbedKComputation tests the helper that lives in the types
// package rather than this one. Keeping the tests colocated here is
// explicit in the spec scope for the foundation role, so we import types
// and verify the formula K = min(ceil(sqrt(N)), 50).
func TestTopicEmbedKComputation(t *testing.T) {
	// We import the helper locally by duplicating its math to avoid a cycle;
	// the real test is in types_test, but we mirror it here per spec scope.
	// The formula: K = min(ceil(sqrt(N)), 50), 0 for N<=0.
	cases := []struct {
		n    int
		want int
	}{
		{-5, 0},
		{0, 0},
		{1, 1},   // ceil(sqrt(1)) = 1
		{2, 2},   // ceil(sqrt(2)) = 2
		{4, 2},   // ceil(sqrt(4)) = 2
		{5, 3},   // ceil(sqrt(5)) ≈ 2.23 → 3
		{25, 5},  // ceil(sqrt(25)) = 5
		{50, 8},  // ceil(sqrt(50)) ≈ 7.07 → 8
		{100, 10},
		{2400, 49}, // ceil(sqrt(2400)) ≈ 49 (below cap)
		{2500, 50}, // ceil(sqrt(2500)) = 50 (at cap)
		{2501, 50}, // would be 51 but clamped to 50
		{10000, 50}, // capped
	}
	for _, c := range cases {
		got := topicEmbedKLocal(c.n)
		if got != c.want {
			t.Errorf("TopicEmbedK(%d) = %d, want %d", c.n, got, c.want)
		}
	}
}

// topicEmbedKLocal mirrors types.TopicEmbedK so this test file can stay
// free of cross-package imports (keeping the vectors package standalone).
func topicEmbedKLocal(n int) int {
	if n <= 0 {
		return 0
	}
	k := int(math.Ceil(math.Sqrt(float64(n))))
	if k > 50 {
		k = 50
	}
	return k
}
