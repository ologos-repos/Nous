// Package migrations contains Nous's SQL schema migrations embedded into the binary.
//
// Each migration is a numbered SQL file: 001_initial_schema.sql,
// 002_topic_registry.sql, etc. The embedded FS is consumed by the
// internal/store migration runner, which parses version numbers from the
// filenames and applies them in ascending order inside a transaction.
//
// Go forbids ".." in go:embed patterns, so we expose the FS from this
// package rather than embedding the top-level migrations/ directory from
// inside internal/store.
package migrations

import "embed"

// FS is the embedded migration filesystem. All *.sql files next to this
// source file are included. Consumers should treat entries as read-only.
//
//go:embed *.sql
var FS embed.FS
