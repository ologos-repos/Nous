// Package store — PostgreSQL integration tests.
//
// These tests require a live Postgres database. Gate them with:
//
//	NOUS_POSTGRES_URL="postgres://rhode@localhost:5432/nous_test?sslmode=disable" \
//	  go test -v -count=1 ./internal/store/ -run TestIntegration
//
// If NOUS_POSTGRES_URL is not set, every test in this file is skipped.
// Each test uses unique string prefixes to avoid cross-test pollution and
// cleans up its own rows via DELETE at the end.
package store

import (
	"context"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/ologos-repos/nous/internal/types"
)

// testConnect opens a MemoryStore against NOUS_POSTGRES_URL, running
// migrations on first call. Registers t.Cleanup(store.Close). Tests that
// call this helper will be skipped if NOUS_POSTGRES_URL is not set.
func testConnect(t *testing.T) *MemoryStore {
	t.Helper()
	pgURL := os.Getenv("NOUS_POSTGRES_URL")
	if pgURL == "" {
		t.Skip("NOUS_POSTGRES_URL not set, skipping Postgres integration tests")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	s, err := Connect(ctx, Config{
		PostgresURL:   pgURL,
		RunMigrations: true,
		ShellDir:      t.TempDir(),
	})
	if err != nil {
		t.Fatalf("store.Connect: %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })
	return s
}

// tctx returns a background context with a 30-second timeout, registering
// the cancel function with t.Cleanup so vet's context-leak check is satisfied.
func tctx(t *testing.T) context.Context {
	t.Helper()
	c, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	t.Cleanup(cancel)
	return c
}

// cleanDirectorMemories deletes all director_memory rows whose content has the
// given prefix. Used for per-test cleanup.
func cleanDirectorMemories(t *testing.T, s *MemoryStore, prefix string) {
	t.Helper()
	_, err := s.pool.Exec(tctx(t), `DELETE FROM director_memory WHERE content LIKE $1`, prefix+"%")
	if err != nil {
		t.Logf("cleanup director_memory: %v", err)
	}
}

// cleanWorkerMemories removes worker_memory rows for a given worker name.
func cleanWorkerMemories(t *testing.T, s *MemoryStore, worker string) {
	t.Helper()
	_, err := s.pool.Exec(tctx(t), `DELETE FROM worker_memory WHERE worker_name = $1`, worker)
	if err != nil {
		t.Logf("cleanup worker_memory(%s): %v", worker, err)
	}
}

// cleanConversations removes all conversation rows whose content has the given prefix.
func cleanConversations(t *testing.T, s *MemoryStore, prefix string) {
	t.Helper()
	_, err := s.pool.Exec(tctx(t), `DELETE FROM conversations WHERE content LIKE $1`, prefix+"%")
	if err != nil {
		t.Logf("cleanup conversations: %v", err)
	}
}

// cleanWorkerHistory removes worker_history rows for a given worker name.
func cleanWorkerHistory(t *testing.T, s *MemoryStore, worker string) {
	t.Helper()
	_, err := s.pool.Exec(tctx(t), `DELETE FROM worker_history WHERE worker_name = $1`, worker)
	if err != nil {
		t.Logf("cleanup worker_history(%s): %v", worker, err)
	}
}

// cleanTriplets removes triplets by source_type+source_id prefix.
func cleanTriplets(t *testing.T, s *MemoryStore, sourceType, sourceIDPrefix string) {
	t.Helper()
	_, err := s.pool.Exec(tctx(t),
		`DELETE FROM triplets WHERE source_type = $1 AND source_id LIKE $2`,
		sourceType, sourceIDPrefix+"%",
	)
	if err != nil {
		t.Logf("cleanup triplets: %v", err)
	}
}

// cleanTopics removes topics by name prefix.
func cleanTopics(t *testing.T, s *MemoryStore, prefix string) {
	t.Helper()
	_, err := s.pool.Exec(tctx(t), `DELETE FROM topic_registry WHERE name LIKE $1`, prefix+"%")
	if err != nil {
		t.Logf("cleanup topic_registry: %v", err)
	}
}

// =============================================================================
// Tests
// =============================================================================

// TestIntegration_RememberAndRecall verifies the basic director-memory lifecycle:
// Remember → GetMemoryByID → GetAllMemories → Forget → verify gone.
func TestIntegration_RememberAndRecall(t *testing.T) {
	s := testConnect(t)
	c := tctx(t)

	const prefix = "IntTest_RememberAndRecall_"
	t.Cleanup(func() { cleanDirectorMemories(t, s, prefix) })

	content := prefix + "The sky is blue"
	meta := map[string]any{"source": "integration-test"}

	// Remember.
	mem, err := s.Remember(c, content, string(types.CategoryFact), nil, meta)
	if err != nil {
		t.Fatalf("Remember: %v", err)
	}
	if mem.ID == 0 {
		t.Fatal("expected non-zero ID after Remember")
	}
	if mem.Content != content {
		t.Errorf("content mismatch: got %q want %q", mem.Content, content)
	}
	if mem.Category != string(types.CategoryFact) {
		t.Errorf("category mismatch: got %q", mem.Category)
	}
	if mem.Tier != types.TierDirector {
		t.Errorf("tier mismatch: got %q", mem.Tier)
	}

	// GetMemoryByID.
	fetched, found, err := s.GetMemoryByID(c, mem.ID)
	if err != nil {
		t.Fatalf("GetMemoryByID: %v", err)
	}
	if !found {
		t.Fatal("GetMemoryByID: expected found=true")
	}
	if fetched.Content != content {
		t.Errorf("GetMemoryByID content: got %q", fetched.Content)
	}

	// GetAllMemories should include it.
	all, err := s.GetAllMemories(c, "", 500)
	if err != nil {
		t.Fatalf("GetAllMemories: %v", err)
	}
	found = false
	for _, m := range all {
		if m.ID == mem.ID {
			found = true
			break
		}
	}
	if !found {
		t.Error("GetAllMemories: inserted memory not present in results")
	}

	// Forget.
	deleted, err := s.Forget(c, mem.ID)
	if err != nil {
		t.Fatalf("Forget: %v", err)
	}
	if !deleted {
		t.Error("Forget: expected deleted=true")
	}

	// Should be gone now.
	_, found, err = s.GetMemoryByID(c, mem.ID)
	if err != nil {
		t.Fatalf("GetMemoryByID after Forget: %v", err)
	}
	if found {
		t.Error("GetMemoryByID: expected found=false after Forget")
	}

	// Forget a second time should return false without error.
	deleted, err = s.Forget(c, mem.ID)
	if err != nil {
		t.Fatalf("double-Forget: %v", err)
	}
	if deleted {
		t.Error("double-Forget: expected deleted=false for missing row")
	}
}

// TestIntegration_HybridRecall stores 5 memories with varied content and
// verifies that a keyword search returns the correct memories.
func TestIntegration_HybridRecall(t *testing.T) {
	s := testConnect(t)
	c := tctx(t)

	const prefix = "IntTest_HybridRecall_"
	t.Cleanup(func() { cleanDirectorMemories(t, s, prefix) })

	items := []struct {
		content  string
		category string
	}{
		{prefix + "postgresql database stores rows efficiently", string(types.CategoryFact)},
		{prefix + "golang channels enable concurrency", string(types.CategoryLesson)},
		{prefix + "vector embeddings represent semantic meaning", string(types.CategoryFact)},
		{prefix + "dendritic recall routes queries via topics", string(types.CategoryProject)},
		{prefix + "the capital of France is Paris", string(types.CategoryFact)},
	}

	ids := make([]int64, len(items))
	for i, it := range items {
		m, err := s.Remember(c, it.content, it.category, nil, nil)
		if err != nil {
			t.Fatalf("Remember[%d]: %v", i, err)
		}
		ids[i] = m.ID
	}

	// Search for "postgresql" — should hit the first row.
	results, err := s.HybridRecall(c, "postgresql database", "", 10, 0.0, 0, 0)
	if err != nil {
		t.Fatalf("HybridRecall: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("HybridRecall: expected at least one result for 'postgresql'")
	}
	foundDB := false
	for _, r := range results {
		if r.Memory.ID == ids[0] {
			foundDB = true
			break
		}
	}
	if !foundDB {
		t.Error("HybridRecall: postgresql memory not in results")
	}

	// Search for "golang concurrency" — should hit index 1.
	results2, err := s.HybridRecall(c, "golang concurrency", "", 10, 0.0, 0, 0)
	if err != nil {
		t.Fatalf("HybridRecall[golang]: %v", err)
	}
	foundGo := false
	for _, r := range results2 {
		if r.Memory.ID == ids[1] {
			foundGo = true
			break
		}
	}
	if !foundGo {
		t.Error("HybridRecall: golang memory not in results for 'golang concurrency'")
	}

	// Category filter: searching "postgresql" in category "lesson" should NOT
	// return index 0 (which is category "fact").
	results3, err := s.HybridRecall(c, "postgresql database", string(types.CategoryLesson), 10, 0.0, 0, 0)
	if err != nil {
		t.Fatalf("HybridRecall[category filter]: %v", err)
	}
	for _, r := range results3 {
		if r.Memory.ID == ids[0] {
			t.Error("HybridRecall category filter: fact memory leaked into lesson results")
		}
	}
}

// TestIntegration_TopicCRUD exercises create, read (by id + name), list, update,
// and delete of topic registry entries.
func TestIntegration_TopicCRUD(t *testing.T) {
	s := testConnect(t)
	c := tctx(t)

	const prefix = "inttest-topiccrud-"
	t.Cleanup(func() { cleanTopics(t, s, prefix) })

	// CreateTopic.
	topic, err := s.CreateTopic(c,
		prefix+"golang-patterns",
		"Go Patterns",
		"Patterns and idioms specific to the Go programming language",
		types.TopicSourceCurated,
	)
	if err != nil {
		t.Fatalf("CreateTopic: %v", err)
	}
	if topic.ID == 0 {
		t.Fatal("CreateTopic: expected non-zero ID")
	}
	if topic.Name != prefix+"golang-patterns" {
		t.Errorf("CreateTopic name: got %q", topic.Name)
	}
	if topic.Source != types.TopicSourceCurated {
		t.Errorf("CreateTopic source: got %q", topic.Source)
	}

	// GetTopic by ID.
	got, found, err := s.GetTopic(c, topic.ID)
	if err != nil {
		t.Fatalf("GetTopic: %v", err)
	}
	if !found {
		t.Fatal("GetTopic: expected found=true")
	}
	if got.DisplayName != "Go Patterns" {
		t.Errorf("GetTopic display_name: got %q", got.DisplayName)
	}

	// GetTopicByName.
	gotByName, found, err := s.GetTopicByName(c, prefix+"golang-patterns")
	if err != nil {
		t.Fatalf("GetTopicByName: %v", err)
	}
	if !found {
		t.Fatal("GetTopicByName: expected found=true")
	}
	if gotByName.ID != topic.ID {
		t.Errorf("GetTopicByName ID mismatch: got %d want %d", gotByName.ID, topic.ID)
	}

	// GetTopicByName for missing slug.
	_, found, err = s.GetTopicByName(c, "does-not-exist-xyz-99")
	if err != nil {
		t.Fatalf("GetTopicByName missing: %v", err)
	}
	if found {
		t.Error("GetTopicByName: expected found=false for missing topic")
	}

	// ListTopics — our topic should appear.
	topics, err := s.ListTopics(c, "")
	if err != nil {
		t.Fatalf("ListTopics: %v", err)
	}
	foundInList := false
	for _, tp := range topics {
		if tp.ID == topic.ID {
			foundInList = true
			break
		}
	}
	if !foundInList {
		t.Error("ListTopics: created topic not in list")
	}

	// UpdateTopic description.
	updated, ok, err := s.UpdateTopic(c, topic.ID, "Go Patterns (updated)", "Updated description for Go patterns")
	if err != nil {
		t.Fatalf("UpdateTopic: %v", err)
	}
	if !ok {
		t.Fatal("UpdateTopic: expected ok=true")
	}
	if updated.Description != "Updated description for Go patterns" {
		t.Errorf("UpdateTopic description: got %q", updated.Description)
	}
	if updated.DisplayName != "Go Patterns (updated)" {
		t.Errorf("UpdateTopic display_name: got %q", updated.DisplayName)
	}

	// UpdateTopic for missing ID should return ok=false.
	_, ok, err = s.UpdateTopic(c, 999999999, "x", "y")
	if err != nil {
		t.Fatalf("UpdateTopic missing: %v", err)
	}
	if ok {
		t.Error("UpdateTopic: expected ok=false for non-existent ID")
	}

	// DeleteTopic.
	if err := s.DeleteTopic(c, topic.ID); err != nil {
		t.Fatalf("DeleteTopic: %v", err)
	}
	_, found, err = s.GetTopic(c, topic.ID)
	if err != nil {
		t.Fatalf("GetTopic after Delete: %v", err)
	}
	if found {
		t.Error("GetTopic: expected found=false after DeleteTopic")
	}
}

// TestIntegration_TopicAssignment creates a topic and memory, assigns the memory
// to the topic, then verifies GetTopicMemories and HybridRecallScoped return it.
func TestIntegration_TopicAssignment(t *testing.T) {
	s := testConnect(t)
	c := tctx(t)

	const prefix = "IntTest_TopicAssignment_"
	const topicPrefix = "inttest-topicassign-"
	t.Cleanup(func() {
		cleanDirectorMemories(t, s, prefix)
		cleanTopics(t, s, topicPrefix)
	})

	// Create topic.
	topic, err := s.CreateTopic(c,
		topicPrefix+"rust-patterns",
		"Rust Patterns",
		"Safe systems programming patterns in Rust",
		types.TopicSourceCurated,
	)
	if err != nil {
		t.Fatalf("CreateTopic: %v", err)
	}

	// Create memories — one to assign, one as noise in a different category.
	mem, err := s.Remember(c, prefix+"ownership model prevents data races in Rust", string(types.CategoryLesson), nil, nil)
	if err != nil {
		t.Fatalf("Remember: %v", err)
	}
	noise, err := s.Remember(c, prefix+"unrelated noise memory about databases", string(types.CategoryFact), nil, nil)
	if err != nil {
		t.Fatalf("Remember noise: %v", err)
	}
	_ = noise

	// AssignMemoryToTopic.
	if err := s.AssignMemoryToTopic(c, mem.ID, topic.ID); err != nil {
		t.Fatalf("AssignMemoryToTopic: %v", err)
	}

	// GetTopicMemories should return the assigned memory.
	topicMems, err := s.GetTopicMemories(c, topic.ID, 100)
	if err != nil {
		t.Fatalf("GetTopicMemories: %v", err)
	}
	foundAssigned := false
	for _, m := range topicMems {
		if m.ID == mem.ID {
			foundAssigned = true
			break
		}
	}
	if !foundAssigned {
		t.Error("GetTopicMemories: assigned memory not found in topic")
	}

	// The noise memory should NOT appear.
	for _, m := range topicMems {
		if m.ID == noise.ID {
			t.Error("GetTopicMemories: unassigned noise memory appeared in topic results")
		}
	}

	// Verify the denormalized memory_count was updated.
	tp, found, err := s.GetTopic(c, topic.ID)
	if err != nil {
		t.Fatalf("GetTopic for count check: %v", err)
	}
	if !found {
		t.Fatal("GetTopic: topic missing after assignment")
	}
	if tp.MemoryCount < 1 {
		t.Errorf("topic memory_count: got %d, want >= 1", tp.MemoryCount)
	}

	// HybridRecallScoped: should surface the assigned memory.
	scoped, err := s.HybridRecallScoped(c, "ownership rust races", topic.ID, 10, 0.0)
	if err != nil {
		t.Fatalf("HybridRecallScoped: %v", err)
	}
	foundScoped := false
	for _, r := range scoped {
		if r.Memory.ID == mem.ID {
			foundScoped = true
			break
		}
	}
	if !foundScoped {
		t.Error("HybridRecallScoped: assigned memory not returned by scoped search")
	}

	// HybridRecallScoped should NOT return the noise memory (different topic).
	for _, r := range scoped {
		if r.Memory.ID == noise.ID {
			t.Error("HybridRecallScoped: unscoped noise memory appeared in scoped results")
		}
	}
}

// TestIntegration_TripletCRUD stores a small graph, retrieves triplets by source,
// walks the graph 1 hop, and checks GetEntityNeighborhood.
func TestIntegration_TripletCRUD(t *testing.T) {
	s := testConnect(t)
	c := tctx(t)

	const srcType = "inttest"
	const srcID = "inttest-tripletcrud-001"
	t.Cleanup(func() { cleanTriplets(t, s, srcType, "inttest-tripletcrud") })

	triplets := [][3]string{
		{"Alice", "knows", "Bob"},
		{"Bob", "works_at", "Acme"},
		{"Carol", "knows", "Alice"},
		{"Acme", "located_in", "Springfield"},
	}

	n, err := s.StoreTriplets(c, triplets, srcType, srcID, 0.95)
	if err != nil {
		t.Fatalf("StoreTriplets: %v", err)
	}
	if n != 4 {
		t.Errorf("StoreTriplets: inserted %d, want 4", n)
	}

	// StoreTriplets with empties should skip silently.
	nEmpty, err := s.StoreTriplets(c, [][3]string{{"", "pred", "obj"}, {"subj", "", "obj"}}, srcType, srcID, 1.0)
	if err != nil {
		t.Fatalf("StoreTriplets with empties: %v", err)
	}
	if nEmpty != 0 {
		t.Errorf("StoreTriplets with empties: expected 0 inserted, got %d", nEmpty)
	}

	// GetTripletsBySource.
	sourced, err := s.GetTripletsBySource(c, srcType, srcID)
	if err != nil {
		t.Fatalf("GetTripletsBySource: %v", err)
	}
	if len(sourced) != 4 {
		t.Errorf("GetTripletsBySource: got %d, want 4", len(sourced))
	}
	for _, tr := range sourced {
		if tr.SourceType != srcType || tr.SourceID != srcID {
			t.Errorf("GetTripletsBySource: wrong source on triplet id=%d", tr.ID)
		}
	}

	// WalkGraph 1 hop from Alice — should reach Bob and Carol.
	walked, err := s.WalkGraph(c, []string{"Alice"}, 1, 50)
	if err != nil {
		t.Fatalf("WalkGraph: %v", err)
	}
	if len(walked) == 0 {
		t.Fatal("WalkGraph: expected at least one triplet from Alice")
	}
	// We should see at least "Alice knows Bob" and "Carol knows Alice".
	subjectSet := map[string]bool{}
	objectSet := map[string]bool{}
	for _, tr := range walked {
		subjectSet[tr.Subject] = true
		objectSet[tr.Object] = true
	}
	if !subjectSet["Alice"] && !objectSet["Alice"] {
		t.Error("WalkGraph: Alice not present in any walked triplet")
	}

	// WalkGraph 2 hops from Alice should reach Acme (via Bob).
	walked2, err := s.WalkGraph(c, []string{"Alice"}, 2, 50)
	if err != nil {
		t.Fatalf("WalkGraph 2 hops: %v", err)
	}
	foundAcme := false
	for _, tr := range walked2 {
		if tr.Subject == "Acme" || tr.Object == "Acme" {
			foundAcme = true
			break
		}
	}
	if !foundAcme {
		t.Error("WalkGraph 2 hops: expected Acme reachable via Bob")
	}

	// GetEntityNeighborhood (convenience wrapper).
	nbr, err := s.GetEntityNeighborhood(c, "Bob", 1)
	if err != nil {
		t.Fatalf("GetEntityNeighborhood: %v", err)
	}
	if len(nbr) == 0 {
		t.Error("GetEntityNeighborhood: Bob has no neighbors, expected at least 1")
	}
}

// TestIntegration_WorkerMemory verifies worker-scoped remember/recall/forget and
// that worker A cannot see worker B's memories.
func TestIntegration_WorkerMemory(t *testing.T) {
	s := testConnect(t)
	c := tctx(t)

	const workerA = "inttest-workerA-6f3a"
	const workerB = "inttest-workerB-6f3a"
	t.Cleanup(func() {
		cleanWorkerMemories(t, s, workerA)
		cleanWorkerMemories(t, s, workerB)
	})

	// WorkerRemember for A.
	mA, err := s.WorkerRemember(c, workerA, "WorkerA secret: dendritic recall is the best", string(types.CategoryLesson), nil)
	if err != nil {
		t.Fatalf("WorkerRemember A: %v", err)
	}
	if mA.ID == 0 {
		t.Fatal("WorkerRemember A: expected non-zero ID")
	}
	if mA.WorkerName != workerA {
		t.Errorf("WorkerRemember A: worker_name got %q want %q", mA.WorkerName, workerA)
	}

	// WorkerRemember for B.
	mB, err := s.WorkerRemember(c, workerB, "WorkerB note: golang is awesome", string(types.CategoryFact), nil)
	if err != nil {
		t.Fatalf("WorkerRemember B: %v", err)
	}

	// WorkerRecall for A — should find A's memory.
	resA, err := s.WorkerRecall(c, workerA, "dendritic recall", 10)
	if err != nil {
		t.Fatalf("WorkerRecall A: %v", err)
	}
	foundA := false
	for _, r := range resA {
		if r.Memory.ID == mA.ID {
			foundA = true
		}
		// Scoping: A's recall must never return B's memory.
		if r.Memory.ID == mB.ID {
			t.Error("WorkerRecall A: returned worker B's memory (scoping broken)")
		}
	}
	if !foundA {
		t.Error("WorkerRecall A: worker A's own memory not found")
	}

	// WorkerRecall for B — should find B's memory, not A's.
	resB, err := s.WorkerRecall(c, workerB, "golang awesome", 10)
	if err != nil {
		t.Fatalf("WorkerRecall B: %v", err)
	}
	foundB := false
	for _, r := range resB {
		if r.Memory.ID == mB.ID {
			foundB = true
		}
		if r.Memory.ID == mA.ID {
			t.Error("WorkerRecall B: returned worker A's memory (scoping broken)")
		}
	}
	if !foundB {
		t.Error("WorkerRecall B: worker B's own memory not found")
	}

	// WorkerForget — A cannot forget B's memory.
	deleted, err := s.WorkerForget(c, workerA, mB.ID)
	if err != nil {
		t.Fatalf("WorkerForget cross-worker: %v", err)
	}
	if deleted {
		t.Error("WorkerForget: worker A should NOT be able to delete worker B's memory")
	}

	// WorkerForget — A can forget its own memory.
	deleted, err = s.WorkerForget(c, workerA, mA.ID)
	if err != nil {
		t.Fatalf("WorkerForget own: %v", err)
	}
	if !deleted {
		t.Error("WorkerForget: expected deleted=true for own memory")
	}

	// Verify A's memory is gone: recall returns nothing matching.
	resA2, err := s.WorkerRecall(c, workerA, "dendritic recall", 10)
	if err != nil {
		t.Fatalf("WorkerRecall A after forget: %v", err)
	}
	for _, r := range resA2 {
		if r.Memory.ID == mA.ID {
			t.Error("WorkerRecall A: deleted memory still returned after WorkerForget")
		}
	}

	// Empty worker name should be rejected.
	_, err = s.WorkerRemember(c, "", "content", "general", nil)
	if err == nil {
		t.Error("WorkerRemember with empty name: expected error")
	}
}

// TestIntegration_Conversations logs a user+assistant exchange and verifies
// GetRecentConversations returns the turns in chronological order.
func TestIntegration_Conversations(t *testing.T) {
	s := testConnect(t)
	c := tctx(t)

	const prefix = "IntTest_Conversations_"
	t.Cleanup(func() { cleanConversations(t, s, prefix) })

	turns := []struct{ role, content string }{
		{"user", prefix + "What is dendritic recall?"},
		{"assistant", prefix + "Dendritic recall is a three-tier topic-based RAG pipeline."},
		{"user", prefix + "How does Tier 1 routing work?"},
		{"assistant", prefix + "Tier 1 uses cosine similarity against topic embedding arrays."},
	}

	for _, turn := range turns {
		t, err := s.LogConversation(c, turn.role, turn.content)
		if err != nil {
			fmt.Printf("LogConversation: %v\n", err)
		}
		if t.ID == 0 {
			fmt.Println("LogConversation: expected non-zero ID")
		}
	}

	// GetRecentConversations — limit 4, should return all 4 turns in chrono order.
	recent, err := s.GetRecentConversations(c, 4, 0)
	if err != nil {
		t.Fatalf("GetRecentConversations: %v", err)
	}

	// Filter to just our test turns.
	var mine []types.ConversationTurn
	for _, turn := range recent {
		if strings.HasPrefix(turn.Content, prefix) {
			mine = append(mine, turn)
		}
	}
	if len(mine) < 4 {
		t.Errorf("GetRecentConversations: got %d of our turns, want 4", len(mine))
	}

	// Verify chronological order.
	for i := 1; i < len(mine); i++ {
		if mine[i].CreatedAt.Before(mine[i-1].CreatedAt) || mine[i].ID < mine[i-1].ID {
			t.Error("GetRecentConversations: turns not in chronological order")
			break
		}
	}

	// Verify roles.
	if len(mine) >= 2 {
		if mine[0].Role != "user" {
			t.Errorf("first turn role: got %q want user", mine[0].Role)
		}
		if mine[1].Role != "assistant" {
			t.Errorf("second turn role: got %q want assistant", mine[1].Role)
		}
	}

	// Limit should be respected — request only 2.
	limited, err := s.GetRecentConversations(c, 2, 0)
	if err != nil {
		t.Fatalf("GetRecentConversations limited: %v", err)
	}
	if len(limited) > 2 {
		t.Errorf("GetRecentConversations: limit=2 returned %d rows", len(limited))
	}
}

// TestIntegration_WorkerResume records task completions and reads back the resume.
func TestIntegration_WorkerResume(t *testing.T) {
	s := testConnect(t)
	c := tctx(t)

	const worker = "inttest-resume-worker-7a"
	t.Cleanup(func() { cleanWorkerHistory(t, s, worker) })

	now := time.Now()
	r1, err := s.RecordTaskCompletion(c,
		worker,
		"task-001",
		"Deploy the API service",
		"completed",
		"go,docker",
		"Deployed v2.3 of the Nous API service with zero downtime.",
		&now,
	)
	if err != nil {
		t.Fatalf("RecordTaskCompletion: %v", err)
	}
	if r1.TaskID != "task-001" {
		t.Errorf("RecordTaskCompletion: task_id got %q want task-001", r1.TaskID)
	}
	if r1.Outcome != "completed" {
		t.Errorf("RecordTaskCompletion: outcome got %q want completed", r1.Outcome)
	}
	if r1.FinishedAt == nil {
		t.Error("RecordTaskCompletion: expected non-nil FinishedAt")
	}

	// Second entry.
	_, err = s.RecordTaskCompletion(c,
		worker,
		"task-002",
		"Write integration tests",
		"completed",
		"go,testing",
		"Wrote 9 integration tests covering all store methods.",
		nil, // no started_at
	)
	if err != nil {
		t.Fatalf("RecordTaskCompletion 2: %v", err)
	}

	// GetWorkerResume — should return both entries.
	resume, err := s.GetWorkerResume(c, worker, 10)
	if err != nil {
		t.Fatalf("GetWorkerResume: %v", err)
	}
	if len(resume) < 2 {
		t.Errorf("GetWorkerResume: got %d entries, want >= 2", len(resume))
	}

	// Verify task IDs present.
	ids := map[string]bool{}
	for _, r := range resume {
		ids[r.TaskID] = true
	}
	if !ids["task-001"] {
		t.Error("GetWorkerResume: task-001 not found")
	}
	if !ids["task-002"] {
		t.Error("GetWorkerResume: task-002 not found")
	}

	// Empty worker name should error.
	_, err = s.GetWorkerResume(c, "", 5)
	if err == nil {
		t.Error("GetWorkerResume with empty name: expected error")
	}
}

// TestIntegration_MigrationIdempotency runs migrations twice and verifies no
// error is returned on the second run (schema_versions prevents re-application).
func TestIntegration_MigrationIdempotency(t *testing.T) {
	s := testConnect(t)

	// First run — should have 0 remaining (Connect already ran migrations).
	runner := NewMigrationRunner(s.Pool())
	n, err := runner.Run(tctx(t))
	if err != nil {
		t.Fatalf("migrations second run: %v", err)
	}
	if n != 0 {
		t.Errorf("migrations second run: expected 0 new migrations, got %d", n)
	}

	// Third run for good measure.
	n2, err := runner.Run(tctx(t))
	if err != nil {
		t.Fatalf("migrations third run: %v", err)
	}
	if n2 != 0 {
		t.Errorf("migrations third run: expected 0 new migrations, got %d", n2)
	}

	// Applied() should list all our migration versions.
	applied, err := runner.Applied(tctx(t))
	if err != nil {
		t.Fatalf("Applied: %v", err)
	}
	files, err := LoadMigrationFiles()
	if err != nil {
		t.Fatalf("LoadMigrationFiles: %v", err)
	}
	for _, f := range files {
		if !applied[f.Version] {
			t.Errorf("migration %d (%s) not in applied set", f.Version, f.Name)
		}
	}
}
