package store

import (
	"context"
	"fmt"
	"io/fs"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/ologos-repos/nous/migrations"
)

// migrationFilePattern matches files named like "001_initial_schema.sql".
// Capture group 1 is the numeric version prefix.
var migrationFilePattern = regexp.MustCompile(`^(\d+)_[\w-]+\.sql$`)

// MigrationRunner applies pending SQL migrations to a PostgreSQL database.
//
// Migrations are numbered files embedded into the binary under the
// github.com/ologos-repos/nous/migrations package: 001_*.sql, 002_*.sql, etc.
// The schema_versions table tracks which migrations have been applied.
// Each migration runs inside its own transaction — if any statement fails,
// the migration is rolled back and the error is returned without applying
// later migrations.
//
// Migrations should be written idempotently (use IF NOT EXISTS) so they are
// safe to run repeatedly during development.
type MigrationRunner struct {
	pool *pgxpool.Pool
}

// NewMigrationRunner creates a runner bound to a pgxpool.
func NewMigrationRunner(pool *pgxpool.Pool) *MigrationRunner {
	return &MigrationRunner{pool: pool}
}

// migrationFile represents a single migration parsed from the embedded FS.
type migrationFile struct {
	version int
	name    string // e.g. "001_initial_schema"
	path    string // path inside the embed FS
	sql     string
}

// Run applies all unapplied migrations in ascending version order.
// Returns the count of migrations applied.
//
// The very first migration is allowed to bootstrap the schema_versions table
// itself — the runner degrades gracefully when the table does not yet exist.
func (mr *MigrationRunner) Run(ctx context.Context) (int, error) {
	files, err := loadMigrationFiles()
	if err != nil {
		return 0, fmt.Errorf("load migrations: %w", err)
	}
	if len(files) == 0 {
		return 0, nil
	}

	applied, err := mr.Applied(ctx)
	if err != nil {
		return 0, fmt.Errorf("read applied migrations: %w", err)
	}

	count := 0
	for _, f := range files {
		if applied[f.version] {
			continue
		}
		if err := mr.apply(ctx, f); err != nil {
			return count, fmt.Errorf("apply migration %03d (%s): %w", f.version, f.name, err)
		}
		count++
	}
	return count, nil
}

// Applied returns the set of already-applied migration versions.
// If the schema_versions table does not exist yet, returns an empty set.
func (mr *MigrationRunner) Applied(ctx context.Context) (map[int]bool, error) {
	applied := make(map[int]bool)

	// Check for the schema_versions table via information_schema.
	// If missing, we have no applied migrations yet.
	var exists bool
	err := mr.pool.QueryRow(ctx, `
		SELECT EXISTS (
			SELECT 1 FROM information_schema.tables
			WHERE table_schema = current_schema()
			  AND table_name = 'schema_versions'
		)
	`).Scan(&exists)
	if err != nil {
		return nil, fmt.Errorf("check schema_versions existence: %w", err)
	}
	if !exists {
		return applied, nil
	}

	rows, err := mr.pool.Query(ctx, `SELECT version FROM schema_versions`)
	if err != nil {
		return nil, fmt.Errorf("query schema_versions: %w", err)
	}
	defer rows.Close()

	for rows.Next() {
		var v int
		if err := rows.Scan(&v); err != nil {
			return nil, fmt.Errorf("scan version: %w", err)
		}
		applied[v] = true
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate schema_versions: %w", err)
	}
	return applied, nil
}

// apply runs a single migration inside a transaction and records it in
// schema_versions. The INSERT into schema_versions is idempotent via
// ON CONFLICT DO NOTHING to tolerate concurrent runners on the same DB.
func (mr *MigrationRunner) apply(ctx context.Context, f migrationFile) error {
	tx, err := mr.pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("begin tx: %w", err)
	}
	defer func() { _ = tx.Rollback(ctx) }()

	if _, err := tx.Exec(ctx, f.sql); err != nil {
		return fmt.Errorf("exec migration body: %w", err)
	}

	// After the migration body runs, schema_versions must exist (migration 003
	// creates it in the reference spec). Gracefully handle its absence by
	// checking existence first. If it still doesn't exist, the migration set
	// is incomplete and the caller should be alerted on the next Run().
	var hasTable bool
	if err := tx.QueryRow(ctx, `
		SELECT EXISTS (
			SELECT 1 FROM information_schema.tables
			WHERE table_schema = current_schema()
			  AND table_name = 'schema_versions'
		)
	`).Scan(&hasTable); err != nil {
		return fmt.Errorf("check schema_versions: %w", err)
	}

	if hasTable {
		if _, err := tx.Exec(ctx, `
			INSERT INTO schema_versions (version, description)
			VALUES ($1, $2)
			ON CONFLICT (version) DO NOTHING
		`, f.version, f.name); err != nil {
			return fmt.Errorf("record schema_version: %w", err)
		}
	}

	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("commit tx: %w", err)
	}
	return nil
}

// loadMigrationFiles reads all migrations from the embedded FS, parses their
// numeric prefixes, and returns them sorted ascending by version.
func loadMigrationFiles() ([]migrationFile, error) {
	entries, err := fs.ReadDir(migrations.FS, ".")
	if err != nil {
		return nil, fmt.Errorf("read migration dir: %w", err)
	}

	out := make([]migrationFile, 0, len(entries))
	for _, ent := range entries {
		if ent.IsDir() {
			continue
		}
		name := ent.Name()
		if !strings.HasSuffix(name, ".sql") {
			continue
		}
		m := migrationFilePattern.FindStringSubmatch(name)
		if m == nil {
			return nil, fmt.Errorf("malformed migration filename %q (expected NNN_name.sql)", name)
		}
		version, err := strconv.Atoi(m[1])
		if err != nil {
			return nil, fmt.Errorf("parse version from %q: %w", name, err)
		}
		body, err := fs.ReadFile(migrations.FS, name)
		if err != nil {
			return nil, fmt.Errorf("read %q: %w", name, err)
		}
		out = append(out, migrationFile{
			version: version,
			name:    strings.TrimSuffix(name, ".sql"),
			path:    name,
			sql:     string(body),
		})
	}

	sort.Slice(out, func(i, j int) bool { return out[i].version < out[j].version })

	// Detect duplicate versions early so ambiguous ordering never silently applies.
	for i := 1; i < len(out); i++ {
		if out[i].version == out[i-1].version {
			return nil, fmt.Errorf("duplicate migration version %d: %q and %q",
				out[i].version, out[i-1].name, out[i].name)
		}
	}
	return out, nil
}

// LoadMigrationFiles is the exported variant of loadMigrationFiles, useful for
// tests that need to inspect the embedded migration set without a live DB.
func LoadMigrationFiles() ([]MigrationInfo, error) {
	raw, err := loadMigrationFiles()
	if err != nil {
		return nil, err
	}
	info := make([]MigrationInfo, len(raw))
	for i, f := range raw {
		info[i] = MigrationInfo{Version: f.version, Name: f.name, Path: f.path, SQL: f.sql}
	}
	return info, nil
}

// MigrationInfo describes a parsed migration file. Exposed for testing.
type MigrationInfo struct {
	Version int
	Name    string
	Path    string
	SQL     string
}
