package store

import (
	"context"
	"path/filepath"
	"testing"
	"time"

	"github.com/ologos-repos/nous/internal/types"
)

// TestShellStoreCRUD exercises the full per-worker SQLite shell:
// schema creation, memory CRUD, knowledge CRUD, instruction CRUD, and Stat.
// Uses modernc.org/sqlite so no CGo / no external driver required.
func TestShellStoreCRUD(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	dir := t.TempDir()
	path := filepath.Join(dir, "alpha.db")

	sh, err := OpenShellAt("Alpha", path)
	if err != nil {
		t.Fatalf("OpenShellAt: %v", err)
	}
	defer sh.Close()

	if sh.WorkerName() != "Alpha" {
		t.Errorf("WorkerName mismatch: %q", sh.WorkerName())
	}
	if sh.Path() != path {
		t.Errorf("Path mismatch: %q vs %q", sh.Path(), path)
	}

	// -------------------------------------------------------------------------
	// Memories
	// -------------------------------------------------------------------------
	m, err := sh.RememberShell(ctx, "prefers go over python", string(types.CategoryPreference), 0.9, map[string]any{"source": "unit-test"})
	if err != nil {
		t.Fatalf("RememberShell: %v", err)
	}
	if m.ID == 0 {
		t.Error("expected non-zero memory id")
	}
	if m.Category != string(types.CategoryPreference) {
		t.Errorf("category = %q, want preference", m.Category)
	}
	if m.Importance != 0.9 {
		t.Errorf("importance = %v, want 0.9", m.Importance)
	}
	if m.Tier != types.TierPrivate {
		t.Errorf("tier = %q, want private", m.Tier)
	}
	if m.WorkerName != "Alpha" {
		t.Errorf("worker = %q, want Alpha", m.WorkerName)
	}
	if m.Metadata["source"] != "unit-test" {
		t.Errorf("metadata lost: %v", m.Metadata)
	}

	_, err = sh.RememberShell(ctx, "uses tabs not spaces", string(types.CategoryPreference), 0.5, nil)
	if err != nil {
		t.Fatalf("RememberShell 2: %v", err)
	}
	_, err = sh.RememberShell(ctx, "deployed service to prod last Tuesday", string(types.CategoryFact), 0.3, nil)
	if err != nil {
		t.Fatalf("RememberShell 3: %v", err)
	}

	all, err := sh.ListShellMemories(ctx, "", 10)
	if err != nil {
		t.Fatalf("ListShellMemories: %v", err)
	}
	if len(all) != 3 {
		t.Errorf("expected 3 memories, got %d", len(all))
	}

	prefs, err := sh.ListShellMemories(ctx, string(types.CategoryPreference), 10)
	if err != nil {
		t.Fatalf("ListShellMemories preference: %v", err)
	}
	if len(prefs) != 2 {
		t.Errorf("expected 2 preference memories, got %d", len(prefs))
	}

	hits, err := sh.RecallShell(ctx, "go python", "", 10)
	if err != nil {
		t.Fatalf("RecallShell: %v", err)
	}
	if len(hits) < 1 {
		t.Errorf("expected at least 1 recall hit, got %d", len(hits))
	}

	// High-importance memory should come first with the default ORDER BY.
	hits, err = sh.RecallShell(ctx, "tabs", "", 10)
	if err != nil {
		t.Fatalf("RecallShell tabs: %v", err)
	}
	if len(hits) != 1 || hits[0].Content != "uses tabs not spaces" {
		t.Errorf("expected single 'tabs' hit, got %+v", hits)
	}

	ok, err := sh.ForgetShell(ctx, m.ID)
	if err != nil {
		t.Fatalf("ForgetShell: %v", err)
	}
	if !ok {
		t.Errorf("expected forget to return true")
	}
	ok, err = sh.ForgetShell(ctx, m.ID)
	if err != nil {
		t.Fatalf("ForgetShell 2: %v", err)
	}
	if ok {
		t.Errorf("expected forget of missing id to return false")
	}

	// -------------------------------------------------------------------------
	// Knowledge
	// -------------------------------------------------------------------------
	k, err := sh.LearnKnowledge(ctx, "solution:sqlite", "Use PRAGMA journal_mode=WAL for concurrent readers.")
	if err != nil {
		t.Fatalf("LearnKnowledge: %v", err)
	}
	if k.ID == 0 {
		t.Error("expected non-zero knowledge id")
	}
	if k.Topic != "solution:sqlite" {
		t.Errorf("topic = %q, want solution:sqlite", k.Topic)
	}

	_, err = sh.LearnKnowledge(ctx, "solution:postgres", "Use pgxpool for connection pooling.")
	if err != nil {
		t.Fatalf("LearnKnowledge 2: %v", err)
	}

	// Topic-scoped lookup.
	rs, err := sh.LookupKnowledge(ctx, "solution:sqlite", "", 10)
	if err != nil {
		t.Fatalf("LookupKnowledge: %v", err)
	}
	if len(rs) != 1 {
		t.Errorf("expected 1 knowledge match, got %d", len(rs))
	}

	// Keyword-scoped lookup across topics.
	rs, err = sh.LookupKnowledge(ctx, "", "pool", 10)
	if err != nil {
		t.Fatalf("LookupKnowledge 2: %v", err)
	}
	if len(rs) != 1 {
		t.Errorf("expected 1 keyword match, got %d", len(rs))
	}

	// -------------------------------------------------------------------------
	// Instructions
	// -------------------------------------------------------------------------
	i1, err := sh.AddInstruction(ctx, "Always run go vet before committing", 10)
	if err != nil {
		t.Fatalf("AddInstruction: %v", err)
	}
	i2, err := sh.AddInstruction(ctx, "Prefer explicit error wrapping", 5)
	if err != nil {
		t.Fatalf("AddInstruction 2: %v", err)
	}
	_ = i2

	instructions, err := sh.ListInstructions(ctx, 10)
	if err != nil {
		t.Fatalf("ListInstructions: %v", err)
	}
	if len(instructions) != 2 {
		t.Fatalf("expected 2 instructions, got %d", len(instructions))
	}
	// Priority ordering: 10 first, then 5.
	if instructions[0].Priority != 10 {
		t.Errorf("expected priority 10 first, got %d", instructions[0].Priority)
	}

	ok, err = sh.RemoveInstruction(ctx, i1.ID)
	if err != nil {
		t.Fatalf("RemoveInstruction: %v", err)
	}
	if !ok {
		t.Errorf("expected remove to return true")
	}

	// -------------------------------------------------------------------------
	// Stat
	// -------------------------------------------------------------------------
	stats, err := sh.Stat(ctx)
	if err != nil {
		t.Fatalf("Stat: %v", err)
	}
	if stats.WorkerName != "Alpha" {
		t.Errorf("stat worker = %q, want Alpha", stats.WorkerName)
	}
	if stats.MemoriesCount != 2 {
		t.Errorf("memories count = %d, want 2", stats.MemoriesCount)
	}
	if stats.KnowledgeCount != 2 {
		t.Errorf("knowledge count = %d, want 2", stats.KnowledgeCount)
	}
	if stats.InstructionCount != 1 {
		t.Errorf("instruction count = %d, want 1", stats.InstructionCount)
	}
}

// TestShellSchemaIdempotent verifies opening an existing shell DB a second
// time does not fail — the schema SQL uses IF NOT EXISTS everywhere.
func TestShellSchemaIdempotent(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	dir := t.TempDir()
	path := filepath.Join(dir, "beta.db")

	sh1, err := OpenShellAt("Beta", path)
	if err != nil {
		t.Fatalf("OpenShellAt first: %v", err)
	}
	if _, err := sh1.RememberShell(ctx, "first", "", 0.5, nil); err != nil {
		t.Fatalf("remember first: %v", err)
	}
	if err := sh1.Close(); err != nil {
		t.Fatalf("close first: %v", err)
	}

	sh2, err := OpenShellAt("Beta", path)
	if err != nil {
		t.Fatalf("OpenShellAt second: %v", err)
	}
	defer sh2.Close()

	all, err := sh2.ListShellMemories(ctx, "", 10)
	if err != nil {
		t.Fatalf("list second: %v", err)
	}
	if len(all) != 1 {
		t.Errorf("expected 1 memory after reopen, got %d", len(all))
	}
	if all[0].Content != "first" {
		t.Errorf("content mismatch after reopen: %q", all[0].Content)
	}
}
