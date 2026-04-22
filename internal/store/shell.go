package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	_ "modernc.org/sqlite" // pure-Go SQLite driver

	"github.com/ologos-repos/nous/internal/types"
)

// =============================================================================
// ShellStore — per-worker SQLite database
// =============================================================================
//
// Each worker ("Alpha", "Beta", ...) gets its own SQLite file under the
// MemoryStore's ShellDir: e.g. ./shells/Alpha.db. The shell stores:
//
//   - memories:     importance-weighted facts with retention policies
//   - knowledge:    topic-indexed snippets of domain knowledge
//   - instructions: priority-ordered standing directives
//
// The schema is applied on first open with PRAGMA journal_mode=WAL and
// PRAGMA foreign_keys=ON. Shells are opened lazily via MemoryStore.Shell(name).

// shellSchemaSQL is the full DDL applied to a new worker DB. Idempotent —
// safe to re-run on open.
const shellSchemaSQL = `
CREATE TABLE IF NOT EXISTS memories (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    content     TEXT NOT NULL,
    category    TEXT NOT NULL DEFAULT 'general',
    importance  REAL NOT NULL DEFAULT 0.5,
    metadata    TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS knowledge (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    topic      TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS instructions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    content    TEXT NOT NULL,
    priority   INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance DESC);
CREATE INDEX IF NOT EXISTS idx_knowledge_topic ON knowledge(topic);
`

// ShellStore is a handle to a single worker's SQLite shell. Instances are
// cached inside MemoryStore and safe for concurrent use — SQLite serializes
// writes internally via the WAL journal.
type ShellStore struct {
	workerName string
	dbPath     string
	db         *sql.DB
}

// openShellAt opens (or creates) a SQLite DB at path and applies the shell
// schema. Caller is responsible for closing.
func openShellAt(workerName, path string) (*ShellStore, error) {
	// Use file: URI so we can set pragmas via the driver if needed.
	// modernc.org/sqlite handles "file:..." and plain paths.
	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, fmt.Errorf("open sqlite %q: %w", path, err)
	}

	// modernc.org/sqlite is safe for concurrent callers but benefits from
	// a single writer connection; keep the pool conservative.
	db.SetMaxOpenConns(1)
	db.SetMaxIdleConns(1)
	db.SetConnMaxLifetime(0)

	if _, err := db.Exec("PRAGMA journal_mode=WAL;"); err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("enable WAL: %w", err)
	}
	if _, err := db.Exec("PRAGMA foreign_keys=ON;"); err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("enable foreign_keys: %w", err)
	}
	if _, err := db.Exec(shellSchemaSQL); err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("apply shell schema: %w", err)
	}

	return &ShellStore{workerName: workerName, dbPath: path, db: db}, nil
}

// Close releases the underlying SQLite handle.
func (sh *ShellStore) Close() error {
	if sh == nil || sh.db == nil {
		return nil
	}
	return sh.db.Close()
}

// WorkerName returns the worker owning this shell.
func (sh *ShellStore) WorkerName() string { return sh.workerName }

// Path returns the filesystem path of the shell DB.
func (sh *ShellStore) Path() string { return sh.dbPath }

// DB exposes the underlying *sql.DB for advanced callers. Tests use this.
func (sh *ShellStore) DB() *sql.DB { return sh.db }

// =============================================================================
// MemoryStore shell-management API
// =============================================================================

// Shell returns (creating if needed) the ShellStore for the given worker.
// Shells are cached on the MemoryStore; a second call with the same name
// returns the same handle.
func (s *MemoryStore) Shell(workerName string) (*ShellStore, error) {
	if strings.TrimSpace(workerName) == "" {
		return nil, errors.New("worker name required")
	}
	// Fast path: already cached.
	s.mu.RLock()
	sh, ok := s.shells[workerName]
	s.mu.RUnlock()
	if ok {
		return sh, nil
	}

	// Slow path: open + cache. Re-check under write lock to avoid races.
	s.mu.Lock()
	defer s.mu.Unlock()
	if sh, ok := s.shells[workerName]; ok {
		return sh, nil
	}
	if err := os.MkdirAll(s.shellDir, 0o755); err != nil {
		return nil, fmt.Errorf("create shell dir: %w", err)
	}
	path := filepath.Join(s.shellDir, sanitizeWorkerFilename(workerName)+".db")
	opened, err := openShellAt(workerName, path)
	if err != nil {
		return nil, err
	}
	s.shells[workerName] = opened
	return opened, nil
}

// OpenShellAt is a constructor for a standalone shell (tests, CLI utilities).
// Callers are responsible for Close().
func OpenShellAt(workerName, path string) (*ShellStore, error) {
	return openShellAt(workerName, path)
}

// sanitizeWorkerFilename replaces filesystem-hostile characters in worker
// names so users can't create paths like "../etc" via worker name abuse.
// Dots are intentionally disallowed (they can form "." / ".." traversal
// segments); callers that want "Alpha.db" get the ".db" suffix appended
// separately by Shell().
func sanitizeWorkerFilename(name string) string {
	// Allow only [A-Za-z0-9_-]; replace everything else with '_'.
	var b strings.Builder
	for _, r := range name {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9',
			r == '_', r == '-':
			b.WriteRune(r)
		default:
			b.WriteRune('_')
		}
	}
	out := b.String()
	if out == "" {
		out = "worker"
	}
	return out
}

// =============================================================================
// Shell CRUD — memories
// =============================================================================

// RememberShell stores a memory in the worker's private shell.
func (sh *ShellStore) RememberShell(
	ctx context.Context,
	content, category string,
	importance float64,
	metadata map[string]any,
) (types.Memory, error) {
	if category == "" {
		category = string(types.CategoryGeneral)
	}
	if importance <= 0 {
		importance = 0.5
	}
	if metadata == nil {
		metadata = map[string]any{}
	}
	metaJSON, err := json.Marshal(metadata)
	if err != nil {
		return types.Memory{}, fmt.Errorf("marshal metadata: %w", err)
	}

	res, err := sh.db.ExecContext(ctx, `
		INSERT INTO memories (content, category, importance, metadata)
		VALUES (?, ?, ?, ?)
	`, content, category, importance, string(metaJSON))
	if err != nil {
		return types.Memory{}, fmt.Errorf("insert shell memory: %w", err)
	}
	id, err := res.LastInsertId()
	if err != nil {
		return types.Memory{}, err
	}
	return sh.getMemoryByID(ctx, id)
}

// RecallShell does keyword retrieval against the worker's shell memories.
// Results are ordered by (importance DESC, created_at DESC).
func (sh *ShellStore) RecallShell(
	ctx context.Context,
	query, category string,
	limit int,
) ([]types.Memory, error) {
	if limit <= 0 {
		limit = 20
	}

	clauses := []string{}
	args := []any{}
	terms := keywordTerms(query)
	if len(terms) > 0 {
		orParts := make([]string, 0, len(terms))
		for _, t := range terms {
			orParts = append(orParts, "content LIKE ?")
			args = append(args, "%"+t+"%")
		}
		clauses = append(clauses, "("+strings.Join(orParts, " OR ")+")")
	}
	if strings.TrimSpace(category) != "" {
		clauses = append(clauses, "category = ?")
		args = append(args, category)
	}
	sqlText := `
		SELECT id, content, category, importance, metadata, created_at, updated_at
		FROM memories
	`
	if len(clauses) > 0 {
		sqlText += "WHERE " + strings.Join(clauses, " AND ") + "\n"
	}
	sqlText += `ORDER BY importance DESC, created_at DESC LIMIT ?`
	args = append(args, limit)

	rows, err := sh.db.QueryContext(ctx, sqlText, args...)
	if err != nil {
		return nil, fmt.Errorf("query shell memories: %w", err)
	}
	defer rows.Close()

	out := []types.Memory{}
	for rows.Next() {
		m, err := scanShellMemory(rows, sh.workerName)
		if err != nil {
			return nil, err
		}
		out = append(out, m)
	}
	return out, rows.Err()
}

// ForgetShell deletes a memory from the shell. Returns false if no row matched.
func (sh *ShellStore) ForgetShell(ctx context.Context, id int64) (bool, error) {
	res, err := sh.db.ExecContext(ctx, `DELETE FROM memories WHERE id = ?`, id)
	if err != nil {
		return false, fmt.Errorf("delete shell memory: %w", err)
	}
	n, err := res.RowsAffected()
	if err != nil {
		return false, err
	}
	return n > 0, nil
}

// ListShellMemories returns all memories, optionally filtered by category.
func (sh *ShellStore) ListShellMemories(
	ctx context.Context,
	category string,
	limit int,
) ([]types.Memory, error) {
	if limit <= 0 {
		limit = 100
	}

	var (
		rows *sql.Rows
		err  error
	)
	if strings.TrimSpace(category) == "" {
		rows, err = sh.db.QueryContext(ctx, `
			SELECT id, content, category, importance, metadata, created_at, updated_at
			FROM memories ORDER BY created_at DESC LIMIT ?
		`, limit)
	} else {
		rows, err = sh.db.QueryContext(ctx, `
			SELECT id, content, category, importance, metadata, created_at, updated_at
			FROM memories WHERE category = ? ORDER BY created_at DESC LIMIT ?
		`, category, limit)
	}
	if err != nil {
		return nil, fmt.Errorf("list shell memories: %w", err)
	}
	defer rows.Close()

	out := []types.Memory{}
	for rows.Next() {
		m, err := scanShellMemory(rows, sh.workerName)
		if err != nil {
			return nil, err
		}
		out = append(out, m)
	}
	return out, rows.Err()
}

// getMemoryByID fetches a single shell memory by ID.
func (sh *ShellStore) getMemoryByID(ctx context.Context, id int64) (types.Memory, error) {
	row := sh.db.QueryRowContext(ctx, `
		SELECT id, content, category, importance, metadata, created_at, updated_at
		FROM memories WHERE id = ?
	`, id)
	return scanShellMemory(row, sh.workerName)
}

// =============================================================================
// Shell CRUD — knowledge
// =============================================================================

// KnowledgeEntry is a topic-indexed snippet from a worker's shell.
type KnowledgeEntry struct {
	ID        int64     `json:"id"`
	Topic     string    `json:"topic"`
	Content   string    `json:"content"`
	CreatedAt time.Time `json:"created_at"`
}

// LearnKnowledge inserts a knowledge entry.
func (sh *ShellStore) LearnKnowledge(ctx context.Context, topic, content string) (KnowledgeEntry, error) {
	if strings.TrimSpace(topic) == "" {
		return KnowledgeEntry{}, errors.New("topic required")
	}
	res, err := sh.db.ExecContext(ctx, `INSERT INTO knowledge (topic, content) VALUES (?, ?)`, topic, content)
	if err != nil {
		return KnowledgeEntry{}, fmt.Errorf("insert knowledge: %w", err)
	}
	id, err := res.LastInsertId()
	if err != nil {
		return KnowledgeEntry{}, err
	}
	return sh.lookupKnowledgeByID(ctx, id)
}

// LookupKnowledge returns knowledge entries matching a topic (exact) and
// optional content keywords.
func (sh *ShellStore) LookupKnowledge(
	ctx context.Context,
	topic, query string,
	limit int,
) ([]KnowledgeEntry, error) {
	if limit <= 0 {
		limit = 20
	}

	clauses := []string{}
	args := []any{}
	if strings.TrimSpace(topic) != "" {
		clauses = append(clauses, "topic = ?")
		args = append(args, topic)
	}
	terms := keywordTerms(query)
	if len(terms) > 0 {
		orParts := make([]string, 0, len(terms))
		for _, t := range terms {
			orParts = append(orParts, "content LIKE ?")
			args = append(args, "%"+t+"%")
		}
		clauses = append(clauses, "("+strings.Join(orParts, " OR ")+")")
	}
	sqlText := `SELECT id, topic, content, created_at FROM knowledge`
	if len(clauses) > 0 {
		sqlText += " WHERE " + strings.Join(clauses, " AND ")
	}
	sqlText += ` ORDER BY created_at DESC LIMIT ?`
	args = append(args, limit)

	rows, err := sh.db.QueryContext(ctx, sqlText, args...)
	if err != nil {
		return nil, fmt.Errorf("query knowledge: %w", err)
	}
	defer rows.Close()

	out := []KnowledgeEntry{}
	for rows.Next() {
		k, err := scanKnowledge(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, k)
	}
	return out, rows.Err()
}

// lookupKnowledgeByID is a helper for LearnKnowledge.
func (sh *ShellStore) lookupKnowledgeByID(ctx context.Context, id int64) (KnowledgeEntry, error) {
	row := sh.db.QueryRowContext(ctx, `
		SELECT id, topic, content, created_at FROM knowledge WHERE id = ?
	`, id)
	return scanKnowledge(row)
}

// =============================================================================
// Shell CRUD — instructions
// =============================================================================

// Instruction is a standing directive recorded in a worker's shell.
type Instruction struct {
	ID        int64     `json:"id"`
	Content   string    `json:"content"`
	Priority  int       `json:"priority"`
	CreatedAt time.Time `json:"created_at"`
}

// AddInstruction stores a new instruction at the given priority.
func (sh *ShellStore) AddInstruction(ctx context.Context, content string, priority int) (Instruction, error) {
	if strings.TrimSpace(content) == "" {
		return Instruction{}, errors.New("content required")
	}
	res, err := sh.db.ExecContext(ctx, `
		INSERT INTO instructions (content, priority) VALUES (?, ?)
	`, content, priority)
	if err != nil {
		return Instruction{}, fmt.Errorf("insert instruction: %w", err)
	}
	id, err := res.LastInsertId()
	if err != nil {
		return Instruction{}, err
	}

	row := sh.db.QueryRowContext(ctx, `
		SELECT id, content, priority, created_at FROM instructions WHERE id = ?
	`, id)
	return scanInstruction(row)
}

// ListInstructions returns instructions ordered by (priority DESC, created_at DESC).
func (sh *ShellStore) ListInstructions(ctx context.Context, limit int) ([]Instruction, error) {
	if limit <= 0 {
		limit = 100
	}
	rows, err := sh.db.QueryContext(ctx, `
		SELECT id, content, priority, created_at
		FROM instructions
		ORDER BY priority DESC, created_at DESC
		LIMIT ?
	`, limit)
	if err != nil {
		return nil, fmt.Errorf("list instructions: %w", err)
	}
	defer rows.Close()

	out := []Instruction{}
	for rows.Next() {
		i, err := scanInstruction(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, i)
	}
	return out, rows.Err()
}

// RemoveInstruction deletes an instruction by ID.
func (sh *ShellStore) RemoveInstruction(ctx context.Context, id int64) (bool, error) {
	res, err := sh.db.ExecContext(ctx, `DELETE FROM instructions WHERE id = ?`, id)
	if err != nil {
		return false, fmt.Errorf("delete instruction: %w", err)
	}
	n, err := res.RowsAffected()
	return n > 0, err
}

// =============================================================================
// Scan helpers
// =============================================================================

// rowScanner abstracts sql.Row and sql.Rows so scan helpers can share code.
type rowScanner interface {
	Scan(...any) error
}

func scanShellMemory(row rowScanner, workerName string) (types.Memory, error) {
	var (
		m             types.Memory
		metaStr       string
		createdStr    string
		updatedStr    string
		importanceVal float64
	)
	if err := row.Scan(&m.ID, &m.Content, &m.Category, &importanceVal, &metaStr, &createdStr, &updatedStr); err != nil {
		return types.Memory{}, err
	}
	m.Importance = importanceVal
	m.Tier = types.TierPrivate
	m.WorkerName = workerName
	if metaStr != "" {
		_ = json.Unmarshal([]byte(metaStr), &m.Metadata)
	}
	if m.Metadata == nil {
		m.Metadata = map[string]any{}
	}
	m.CreatedAt = parseSQLiteTime(createdStr)
	m.UpdatedAt = parseSQLiteTime(updatedStr)
	return m, nil
}

func scanKnowledge(row rowScanner) (KnowledgeEntry, error) {
	var (
		k          KnowledgeEntry
		createdStr string
	)
	if err := row.Scan(&k.ID, &k.Topic, &k.Content, &createdStr); err != nil {
		return KnowledgeEntry{}, err
	}
	k.CreatedAt = parseSQLiteTime(createdStr)
	return k, nil
}

func scanInstruction(row rowScanner) (Instruction, error) {
	var (
		i          Instruction
		createdStr string
	)
	if err := row.Scan(&i.ID, &i.Content, &i.Priority, &createdStr); err != nil {
		return Instruction{}, err
	}
	i.CreatedAt = parseSQLiteTime(createdStr)
	return i, nil
}

// parseSQLiteTime parses SQLite's default "YYYY-MM-DD HH:MM:SS" time format.
// Returns zero time on parse failure.
func parseSQLiteTime(s string) time.Time {
	if s == "" {
		return time.Time{}
	}
	layouts := []string{
		"2006-01-02 15:04:05",
		"2006-01-02T15:04:05Z",
		time.RFC3339Nano,
		time.RFC3339,
	}
	for _, l := range layouts {
		if t, err := time.Parse(l, s); err == nil {
			return t
		}
	}
	return time.Time{}
}

// Stat returns a lightweight summary of the shell (row counts etc.).
// Useful for WorkerShell responses from the HTTP API.
func (sh *ShellStore) Stat(ctx context.Context) (types.WorkerShell, error) {
	out := types.WorkerShell{
		WorkerName: sh.workerName,
		DBPath:     sh.dbPath,
	}
	if err := sh.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM memories`).Scan(&out.MemoriesCount); err != nil {
		return out, fmt.Errorf("count memories: %w", err)
	}
	if err := sh.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM knowledge`).Scan(&out.KnowledgeCount); err != nil {
		return out, fmt.Errorf("count knowledge: %w", err)
	}
	if err := sh.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM instructions`).Scan(&out.InstructionCount); err != nil {
		return out, fmt.Errorf("count instructions: %w", err)
	}
	return out, nil
}
