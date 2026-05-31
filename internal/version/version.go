// Package version holds the build-time version surface for the nous binary.
//
// Set at build time via ldflags:
//
//	go build -ldflags "-X github.com/ologos-repos/nous/internal/version.Version=v1.2.3.4 \
//	                   -X github.com/ologos-repos/nous/internal/version.BuildTime=2025-01-01T00:00:00Z" \
//	         ./cmd/nous
//
// In development builds (no ldflags), Version defaults to "dev".
package version

// Version is the release tag (e.g. "v0.1.0.0"). Injected via ldflags at
// release build time; defaults to "dev" for local/CI dev builds.
var Version = "dev"

// BuildTime is the UTC build timestamp in RFC 3339 format. Injected via
// ldflags at release build time; defaults to "unknown".
var BuildTime = "unknown"
