// Package main is a thin shim that lets `go build .` (without specifying
// cmd/nous/) produce a working binary.
//
// Most users should run the canonical entry point at ./cmd/nous, but tooling
// and convenience scripts that assume `go build .` produces a usable binary
// in the repo root are common enough that we mirror the entry here.
package main

import (
	"os"
	"os/exec"
)

func main() {
	// Re-exec the canonical CLI located at cmd/nous. We use exec.LookPath
	// to find the installed binary if one is on PATH, otherwise we fall
	// back to running the source via `go run`.
	if path, err := exec.LookPath("nous"); err == nil && path != selfPath() {
		cmd := exec.Command(path, os.Args[1:]...)
		cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
		if err := cmd.Run(); err != nil {
			if exitErr, ok := err.(*exec.ExitError); ok {
				os.Exit(exitErr.ExitCode())
			}
			os.Exit(1)
		}
		return
	}

	// Last-resort: run the canonical CLI source. Not ideal in production —
	// callers should `go build ./cmd/nous` and invoke the resulting binary
	// directly. This shim exists so a fresh clone can do `go run .` and
	// still get a working server.
	args := append([]string{"run", "./cmd/nous"}, os.Args[1:]...)
	cmd := exec.Command("go", args...)
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
	if err := cmd.Run(); err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			os.Exit(exitErr.ExitCode())
		}
		os.Exit(1)
	}
}

// selfPath returns the path of the currently-running binary, or "" if it
// cannot be determined. Used to avoid recursive re-exec when the shim is
// itself installed as `nous`.
func selfPath() string {
	if p, err := os.Executable(); err == nil {
		return p
	}
	return ""
}
