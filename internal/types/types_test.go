package types

import (
	"testing"
	"time"
)

func TestMemoryAgeDays(t *testing.T) {
	m := &Memory{CreatedAt: time.Now().Add(-48 * time.Hour)}
	age := m.AgeDays()
	if age < 1.95 || age > 2.05 {
		t.Errorf("AgeDays = %v, want ≈ 2.0", age)
	}
}

func TestDefaultRetentionPolicy(t *testing.T) {
	rp := DefaultRetentionPolicy()
	if rp.LowThreshold != 0.3 {
		t.Errorf("LowThreshold = %v, want 0.3", rp.LowThreshold)
	}
	if rp.MediumThreshold != 0.7 {
		t.Errorf("MediumThreshold = %v, want 0.7", rp.MediumThreshold)
	}
	if rp.LowRetentionDays != 30 {
		t.Errorf("LowRetentionDays = %v, want 30", rp.LowRetentionDays)
	}
	if rp.MediumRetentionDays != 90 {
		t.Errorf("MediumRetentionDays = %v, want 90", rp.MediumRetentionDays)
	}
	if rp.HighRetentionDays != 180 {
		t.Errorf("HighRetentionDays = %v, want 180", rp.HighRetentionDays)
	}
	wantCats := []string{"decision", "lesson"}
	if len(rp.PreserveCategories) != len(wantCats) {
		t.Fatalf("PreserveCategories = %v, want %v", rp.PreserveCategories, wantCats)
	}
	for i, c := range wantCats {
		if rp.PreserveCategories[i] != c {
			t.Errorf("PreserveCategories[%d] = %v, want %v", i, rp.PreserveCategories[i], c)
		}
	}
}

func TestShouldPrunePreserveCategory(t *testing.T) {
	rp := DefaultRetentionPolicy()
	m := Memory{
		Category:   "decision",
		Importance: 0.0,                                  // very low — would normally prune
		CreatedAt:  time.Now().Add(-365 * 24 * time.Hour), // 1 year old — way past cutoff
	}
	if rp.ShouldPrune(m) {
		t.Error("preserved-category memory was pruned; should always be retained")
	}
}

func TestShouldPruneByImportance(t *testing.T) {
	rp := DefaultRetentionPolicy()
	now := time.Now()

	cases := []struct {
		name       string
		importance float64
		age        time.Duration
		want       bool
	}{
		{"low importance, young", 0.1, 20 * 24 * time.Hour, false},
		{"low importance, old", 0.1, 40 * 24 * time.Hour, true},
		{"medium importance, young", 0.5, 60 * 24 * time.Hour, false},
		{"medium importance, old", 0.5, 100 * 24 * time.Hour, true},
		{"high importance, young", 0.9, 120 * 24 * time.Hour, false},
		{"high importance, old", 0.9, 200 * 24 * time.Hour, true},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			m := Memory{
				Category:   "fact",
				Importance: c.importance,
				CreatedAt:  now.Add(-c.age),
			}
			if got := rp.ShouldPrune(m); got != c.want {
				t.Errorf("ShouldPrune = %v, want %v (imp=%v, age=%v)", got, c.want, c.importance, c.age)
			}
		})
	}
}

func TestTopicEmbedK(t *testing.T) {
	cases := []struct {
		n, want int
	}{
		{-1, 0},
		{0, 0},
		{1, 1},
		{4, 2},
		{5, 3},
		{9, 3},
		{10, 4},
		{25, 5},
		{100, 10},
		{2500, 50},
		{10000, 50}, // clamped
	}
	for _, c := range cases {
		if got := TopicEmbedK(c.n); got != c.want {
			t.Errorf("TopicEmbedK(%d) = %d, want %d", c.n, got, c.want)
		}
	}
}

func TestMemoryTierConstants(t *testing.T) {
	// Round-trip through string() to make sure constants don't accidentally
	// get renamed and break callers (downstream store uses these verbatim).
	if string(TierDirector) != "director" {
		t.Errorf("TierDirector = %q, want %q", TierDirector, "director")
	}
	if string(TierShared) != "shared" {
		t.Errorf("TierShared = %q, want %q", TierShared, "shared")
	}
	if string(TierPrivate) != "private" {
		t.Errorf("TierPrivate = %q, want %q", TierPrivate, "private")
	}
}

func TestTopicSourceConstants(t *testing.T) {
	if string(TopicSourceCurated) != "curated" {
		t.Errorf("TopicSourceCurated = %q, want %q", TopicSourceCurated, "curated")
	}
	if string(TopicSourceEmergent) != "emergent" {
		t.Errorf("TopicSourceEmergent = %q, want %q", TopicSourceEmergent, "emergent")
	}
}
