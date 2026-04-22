package store

import (
	"strings"
	"testing"
)

// TestLoadMigrationFiles verifies the embedded migration set parses, is
// numerically ordered, and matches the expected spec DDL landmarks. This test
// runs without a database — it exercises only embed + parsing.
func TestLoadMigrationFiles(t *testing.T) {
	files, err := LoadMigrationFiles()
	if err != nil {
		t.Fatalf("LoadMigrationFiles: %v", err)
	}
	if len(files) < 3 {
		t.Fatalf("expected at least 3 migrations, got %d", len(files))
	}

	// Versions must be strictly ascending starting at 1.
	for i, f := range files {
		if f.Version != i+1 {
			t.Errorf("migration %d has version %d (expected %d)", i, f.Version, i+1)
		}
		if f.Name == "" {
			t.Errorf("migration %d has empty name", i)
		}
		if f.SQL == "" {
			t.Errorf("migration %d (%s) has empty SQL body", i, f.Name)
		}
	}

	// Sanity-check landmark statements required by the spec.
	tests := []struct {
		version int
		needle  string
	}{
		{1, "director_memory"},
		{1, "conversations"},
		{1, "worker_memory"},
		{1, "worker_history"},
		{1, "triplets"},
		{2, "topic_registry"},
		{3, "topic_embeddings"},
		{3, "schema_versions"},
	}
	for _, tt := range tests {
		var found bool
		for _, f := range files {
			if f.Version == tt.version && strings.Contains(f.SQL, tt.needle) {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("migration %d does not mention %q", tt.version, tt.needle)
		}
	}
}

// TestKeywordTermsFilter verifies the keyword splitter trims stop words,
// short tokens, and duplicates.
func TestKeywordTermsFilter(t *testing.T) {
	out := keywordTerms("The quick brown fox jumps over the lazy dog and the quick fox")
	for _, t2 := range out {
		if len(t2) < 3 {
			t.Errorf("term %q is too short", t2)
		}
		if _, ok := stopWords[t2]; ok {
			t.Errorf("term %q is a stop word", t2)
		}
	}
	// Duplicates should be collapsed ("quick" appears twice in the input).
	seen := map[string]int{}
	for _, t2 := range out {
		seen[t2]++
		if seen[t2] > 1 {
			t.Errorf("term %q appears more than once in output", t2)
		}
	}
	// Must include at least the meaningful content words.
	expected := []string{"quick", "brown", "fox", "jumps", "over", "lazy", "dog"}
	for _, w := range expected {
		found := false
		for _, t2 := range out {
			if t2 == w {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("expected term %q missing from %v", w, out)
		}
	}
}

// TestTermHitRatio verifies the keyword-score helper.
func TestTermHitRatio(t *testing.T) {
	content := "The Go programming language is great for writing HTTP services."
	terms := []string{"go", "http"}
	got := termHitRatio(content, terms)
	if got != 1.0 {
		t.Errorf("all terms present, expected 1.0, got %v", got)
	}

	terms = []string{"go", "rust"}
	got = termHitRatio(content, terms)
	if got != 0.5 {
		t.Errorf("half-present terms expected 0.5, got %v", got)
	}

	if termHitRatio("", nil) != 0.0 {
		t.Errorf("empty terms should score 0")
	}
}

// TestBuildILikeClause verifies the SQL fragment builder.
func TestBuildILikeClause(t *testing.T) {
	ilike, args := buildILikeClause("content", []string{"foo", "bar"}, 3)
	want := "content ILIKE $3 OR content ILIKE $4"
	if ilike != want {
		t.Errorf("ilike = %q, want %q", ilike, want)
	}
	if len(args) != 2 {
		t.Fatalf("expected 2 args, got %d", len(args))
	}
	if args[0] != "%foo%" || args[1] != "%bar%" {
		t.Errorf("unexpected args %v", args)
	}
}

// TestDedupeTrim verifies the BFS frontier helper.
func TestDedupeTrim(t *testing.T) {
	in := []string{"Alpha", " beta ", "alpha", "", "GAMMA", "gamma"}
	out := dedupeTrim(in)
	if len(out) != 3 {
		t.Fatalf("expected 3 unique entries, got %v", out)
	}
	// Preserves first-seen casing.
	if out[0] != "Alpha" || out[1] != "beta" || out[2] != "GAMMA" {
		t.Errorf("unexpected first-seen order: %v", out)
	}
}

// TestSanitizeWorkerFilename verifies filesystem-safe names.
func TestSanitizeWorkerFilename(t *testing.T) {
	cases := map[string]string{
		"Alpha":       "Alpha",
		"rhode-chi":   "rhode-chi",
		"../etc":      "___etc",
		"worker/../x": "worker____x",
		"":            "worker",
	}
	for in, want := range cases {
		if got := sanitizeWorkerFilename(in); got != want {
			t.Errorf("sanitizeWorkerFilename(%q) = %q, want %q", in, got, want)
		}
	}
}
