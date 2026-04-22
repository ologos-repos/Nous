// Package api implements the HTTP transport for Nous Go.
//
// Server, handlers, and middleware live here. The package is intentionally
// thin — all business logic lives in store/ and recall/. The handlers do
// JSON marshalling, parameter parsing, and orchestration only.
package api

import (
	"log"
	"net/http"
	"runtime/debug"
	"time"
)

// jsonContentType always emits application/json on responses. Handlers can
// override later by calling w.Header().Set(...).
func jsonContentType(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		next.ServeHTTP(w, r)
	})
}

// loggingMiddleware logs each request with method, path, status, and latency.
// Wraps the response writer to capture the status code without breaking
// streaming flushers.
func loggingMiddleware(logger *log.Logger) func(http.Handler) http.Handler {
	if logger == nil {
		logger = log.Default()
	}
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			start := time.Now()
			rw := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
			next.ServeHTTP(rw, r)
			logger.Printf("%s %s -> %d (%s)", r.Method, r.URL.RequestURI(), rw.status, time.Since(start))
		})
	}
}

// recoverMiddleware catches panics from handlers, logs them with stack trace,
// and emits a 500 with the standard error envelope so a misbehaving handler
// never takes down the whole server.
func recoverMiddleware(logger *log.Logger) func(http.Handler) http.Handler {
	if logger == nil {
		logger = log.Default()
	}
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			defer func() {
				if rec := recover(); rec != nil {
					logger.Printf("PANIC %s %s: %v\n%s", r.Method, r.URL.RequestURI(), rec, debug.Stack())
					writeError(w, http.StatusInternalServerError, "internal server error", "panic")
				}
			}()
			next.ServeHTTP(w, r)
		})
	}
}

// statusRecorder snapshots the response status code so the logging middleware
// can include it in the access log line.
type statusRecorder struct {
	http.ResponseWriter
	status      int
	wroteHeader bool
}

func (sr *statusRecorder) WriteHeader(code int) {
	if !sr.wroteHeader {
		sr.status = code
		sr.wroteHeader = true
		sr.ResponseWriter.WriteHeader(code)
	}
}

func (sr *statusRecorder) Write(b []byte) (int, error) {
	if !sr.wroteHeader {
		sr.WriteHeader(http.StatusOK)
	}
	return sr.ResponseWriter.Write(b)
}

// chain composes multiple middlewares in left-to-right order. The leftmost
// wrapper is the outermost; the rightmost runs closest to the handler.
func chain(h http.Handler, mws ...func(http.Handler) http.Handler) http.Handler {
	for i := len(mws) - 1; i >= 0; i-- {
		h = mws[i](h)
	}
	return h
}
