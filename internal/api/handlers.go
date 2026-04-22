package api

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/ologos-repos/nous/internal/recall"
	"github.com/ologos-repos/nous/internal/types"
)

// ===== Route registration =====

// registerRoutes wires every spec §8 endpoint to its handler. Uses the Go 1.22
// ServeMux pattern routing (`METHOD /path/{id}`).
func (s *Server) registerRoutes(mux *http.ServeMux) {
	mux.HandleFunc("GET /health", s.handleHealth)

	// Director memory
	mux.HandleFunc("POST /remember", s.handleRemember)
	mux.HandleFunc("POST /recall", s.handleRecall)
	mux.HandleFunc("DELETE /memory/{id}", s.handleForget)

	// Topics
	mux.HandleFunc("GET /topics", s.handleListTopics)
	mux.HandleFunc("POST /topics", s.handleCreateTopic)
	mux.HandleFunc("GET /topics/{id}", s.handleGetTopic)
	mux.HandleFunc("POST /topics/{id}/assign", s.handleAssignToTopic)

	// Triplets
	mux.HandleFunc("GET /triplets", s.handleGetTriplets)
	mux.HandleFunc("POST /triplets", s.handleStoreTriplets)
	mux.HandleFunc("POST /walk", s.handleWalkGraph)

	// Worker memory
	mux.HandleFunc("POST /worker/{name}/remember", s.handleWorkerRemember)
	mux.HandleFunc("POST /worker/{name}/recall", s.handleWorkerRecall)
	mux.HandleFunc("DELETE /worker/{name}/memory/{id}", s.handleWorkerForget)
	mux.HandleFunc("GET /worker/{name}/memories", s.handleWorkerMemories)
	mux.HandleFunc("POST /worker/{name}/history", s.handleWorkerHistoryPost)
	mux.HandleFunc("GET /worker/{name}/history", s.handleWorkerHistoryGet)

	// Conversation
	mux.HandleFunc("POST /conversation", s.handleConversationPost)
	mux.HandleFunc("GET /conversation", s.handleConversationGet)
}

// ===== Helpers =====

// errorResponse is the canonical error envelope per spec §8.3.
type errorResponse struct {
	Error string `json:"error"`
	Code  string `json:"code,omitempty"`
}

func writeJSON(w http.ResponseWriter, status int, body any) {
	w.WriteHeader(status)
	if body == nil {
		return
	}
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(body); err != nil {
		// Already wrote header — best we can do is log.
		// Logger comes through middleware via the wrapped writer.
	}
}

func writeError(w http.ResponseWriter, status int, msg, code string) {
	writeJSON(w, status, errorResponse{Error: msg, Code: code})
}

func decodeJSON(r *http.Request, dst any) error {
	defer r.Body.Close()
	body, err := io.ReadAll(io.LimitReader(r.Body, 10<<20)) // 10 MiB cap
	if err != nil {
		return fmt.Errorf("read body: %w", err)
	}
	if len(body) == 0 {
		return nil // empty body — caller decides if that's valid
	}
	return json.Unmarshal(body, dst)
}

func parseInt64Path(r *http.Request, name string) (int64, error) {
	raw := r.PathValue(name)
	if raw == "" {
		return 0, fmt.Errorf("missing path value %q", name)
	}
	v, err := strconv.ParseInt(raw, 10, 64)
	if err != nil {
		return 0, fmt.Errorf("invalid %s: %w", name, err)
	}
	return v, nil
}

func parseInt64Query(r *http.Request, name string, def int64) int64 {
	raw := r.URL.Query().Get(name)
	if raw == "" {
		return def
	}
	v, err := strconv.ParseInt(raw, 10, 64)
	if err != nil {
		return def
	}
	return v
}

func parseFloatQuery(r *http.Request, name string, def float64) float64 {
	raw := r.URL.Query().Get(name)
	if raw == "" {
		return def
	}
	v, err := strconv.ParseFloat(raw, 64)
	if err != nil {
		return def
	}
	return v
}

// ===== Health =====

type healthResponse struct {
	Status            string    `json:"status"`
	Embedder          string    `json:"embedder"`
	EmbedderAvailable bool      `json:"embedder_available"`
	PGPoolConns       int32     `json:"pg_pool_conns"`
	Timestamp         time.Time `json:"timestamp"`
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	resp := healthResponse{
		Status:    "ok",
		Embedder:  s.embedder.ModelName(),
		Timestamp: time.Now().UTC(),
	}

	pingCtx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()
	if err := s.store.Ping(pingCtx); err != nil {
		resp.Status = "degraded"
		writeJSON(w, http.StatusServiceUnavailable, resp)
		return
	}

	embedCtx, cancel2 := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel2()
	resp.EmbedderAvailable = s.embedder.IsAvailable(embedCtx)
	if pool := s.store.Pool(); pool != nil {
		resp.PGPoolConns = pool.Stat().TotalConns()
	}

	writeJSON(w, http.StatusOK, resp)
}

// ===== Director memory: /remember, /recall, /memory/{id} =====

type rememberRequest struct {
	Content    string         `json:"content"`
	Category   string         `json:"category"`
	TopicID    *int64         `json:"topic_id,omitempty"`
	Importance *float64       `json:"importance,omitempty"`
	Metadata   map[string]any `json:"metadata,omitempty"`
}

func (s *Server) handleRemember(w http.ResponseWriter, r *http.Request) {
	var req rememberRequest
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error(), "bad_json")
		return
	}
	if strings.TrimSpace(req.Content) == "" {
		writeError(w, http.StatusBadRequest, "content is required", "missing_content")
		return
	}
	if req.Category == "" {
		req.Category = string(types.CategoryGeneral)
	}

	mem, err := s.store.Remember(r.Context(), req.Content, req.Category, req.TopicID, req.Metadata)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error(), "store_failure")
		return
	}
	if req.Importance != nil {
		mem.Importance = *req.Importance
	}
	writeJSON(w, http.StatusCreated, mem)
}

type recallRequest struct {
	Query               string  `json:"query"`
	Mode                string  `json:"mode"`
	TopicK              int     `json:"topic_k"`
	ActivationThreshold float64 `json:"activation_threshold"`
	MemoryK             int     `json:"memory_k"`
	Threshold           float64 `json:"threshold"`
	Hops                int     `json:"hops"`
	GraphDiscount       float64 `json:"graph_discount"`
	Limit               int     `json:"limit"`
	Category            string  `json:"category"`
}

type hybridResponse struct {
	Query   string               `json:"query"`
	Mode    string               `json:"mode"`
	Results []types.SearchResult `json:"results"`
}

type dendriticResponse struct {
	Query           string                  `json:"query"`
	Mode            string                  `json:"mode"`
	ActivatedTopics []types.TopicActivation `json:"activated_topics"`
	Results         []types.ScoredMemory    `json:"results"`
	GraphTriplets   []types.Triplet         `json:"graph_triplets"`
	CrossTopicHits  []types.ScoredMemory    `json:"cross_topic_hits"`
}

func (s *Server) handleRecall(w http.ResponseWriter, r *http.Request) {
	var req recallRequest
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error(), "bad_json")
		return
	}
	if strings.TrimSpace(req.Query) == "" {
		writeError(w, http.StatusBadRequest, "query is required", "missing_query")
		return
	}

	mode := strings.ToLower(strings.TrimSpace(req.Mode))
	if mode == "" {
		mode = "dendritic"
	}

	switch mode {
	case "dendritic":
		dr := types.DendriticRecallRequest{
			Query:               req.Query,
			TopicK:              s.applyTopicK(req.TopicK),
			ActivationThreshold: s.applyActivationThreshold(req.ActivationThreshold),
			MemoryK:             s.applyMemoryK(req.MemoryK),
			Threshold:           s.applyThreshold(req.Threshold),
			Hops:                s.applyHops(req.Hops),
			GraphDiscount:       s.applyGraphDiscount(req.GraphDiscount),
			Limit:               s.applyLimit(req.Limit),
		}
		result, err := recall.DendriticRecall(r.Context(), dr, s.store, s.embedder)
		if err != nil {
			writeError(w, http.StatusInternalServerError, err.Error(), "recall_failure")
			return
		}
		writeJSON(w, http.StatusOK, dendriticResponse{
			Query:           result.Query,
			Mode:            "dendritic",
			ActivatedTopics: result.ActivatedTopics,
			Results:         result.Results,
			GraphTriplets:   result.GraphTriplets,
			CrossTopicHits:  result.CrossTopicHits,
		})

	case "hybrid":
		limit := s.applyLimit(req.Limit)
		threshold := s.applyThreshold(req.Threshold)
		results, err := s.store.HybridRecall(r.Context(), req.Query, req.Category, limit, threshold, 24.0, 0.1)
		if err != nil {
			writeError(w, http.StatusInternalServerError, err.Error(), "recall_failure")
			return
		}
		if results == nil {
			results = []types.SearchResult{}
		}
		writeJSON(w, http.StatusOK, hybridResponse{
			Query:   req.Query,
			Mode:    "hybrid",
			Results: results,
		})

	default:
		writeError(w, http.StatusBadRequest, fmt.Sprintf("unknown mode %q (expected 'dendritic' or 'hybrid')", mode), "bad_mode")
	}
}

func (s *Server) handleForget(w http.ResponseWriter, r *http.Request) {
	id, err := parseInt64Path(r, "id")
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error(), "bad_id")
		return
	}
	deleted, err := s.store.Forget(r.Context(), id)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error(), "store_failure")
		return
	}
	if !deleted {
		writeJSON(w, http.StatusNotFound, map[string]any{
			"deleted": false,
			"id":      id,
			"error":   "not found",
		})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"deleted": true, "id": id})
}

// ===== Topics =====

type topicListResponse struct {
	Topics []types.Topic `json:"topics"`
}

func (s *Server) handleListTopics(w http.ResponseWriter, r *http.Request) {
	source := types.TopicSource(strings.TrimSpace(r.URL.Query().Get("source")))
	topics, err := s.store.ListTopics(r.Context(), source)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error(), "store_failure")
		return
	}
	if topics == nil {
		topics = []types.Topic{}
	}
	writeJSON(w, http.StatusOK, topicListResponse{Topics: topics})
}

type createTopicRequest struct {
	Name        string `json:"name"`
	DisplayName string `json:"display_name"`
	Description string `json:"description"`
	Source      string `json:"source"`
}

func (s *Server) handleCreateTopic(w http.ResponseWriter, r *http.Request) {
	var req createTopicRequest
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error(), "bad_json")
		return
	}
	name := strings.TrimSpace(req.Name)
	if name == "" {
		writeError(w, http.StatusBadRequest, "name is required", "missing_name")
		return
	}
	if req.DisplayName == "" {
		req.DisplayName = name
	}
	source := types.TopicSource(req.Source)
	if source == "" {
		source = types.TopicSourceCurated
	}
	if source != types.TopicSourceCurated && source != types.TopicSourceEmergent {
		writeError(w, http.StatusBadRequest, fmt.Sprintf("invalid source %q", req.Source), "bad_source")
		return
	}

	topic, err := s.store.CreateTopic(r.Context(), name, req.DisplayName, req.Description, source)
	if err != nil {
		// Heuristic for duplicate-name errors so the client gets a 409.
		msg := strings.ToLower(err.Error())
		if strings.Contains(msg, "duplicate") || strings.Contains(msg, "already exists") || strings.Contains(msg, "unique") {
			writeError(w, http.StatusConflict, fmt.Sprintf("topic with name %q already exists", name), "duplicate_topic")
			return
		}
		writeError(w, http.StatusInternalServerError, err.Error(), "store_failure")
		return
	}
	writeJSON(w, http.StatusCreated, topic)
}

func (s *Server) handleGetTopic(w http.ResponseWriter, r *http.Request) {
	id, err := parseInt64Path(r, "id")
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error(), "bad_id")
		return
	}
	topic, found, err := s.store.GetTopic(r.Context(), id)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error(), "store_failure")
		return
	}
	if !found {
		writeError(w, http.StatusNotFound, fmt.Sprintf("topic %d not found", id), "topic_not_found")
		return
	}
	writeJSON(w, http.StatusOK, topic)
}

type assignTopicRequest struct {
	MemoryID int64 `json:"memory_id"`
}

func (s *Server) handleAssignToTopic(w http.ResponseWriter, r *http.Request) {
	id, err := parseInt64Path(r, "id")
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error(), "bad_id")
		return
	}
	var req assignTopicRequest
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error(), "bad_json")
		return
	}
	if req.MemoryID <= 0 {
		writeError(w, http.StatusBadRequest, "memory_id is required", "missing_memory_id")
		return
	}
	if err := s.store.AssignMemoryToTopic(r.Context(), req.MemoryID, id); err != nil {
		writeError(w, http.StatusInternalServerError, err.Error(), "assign_failure")
		return
	}

	// Recompute the topic embedding array off the request goroutine — large
	// topics can take several seconds and we don't want to block the client.
	go func(topicID int64) {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		defer cancel()
		if err := s.store.UpdateTopicEmbeddings(ctx, topicID); err != nil {
			s.logger.Printf("nous: UpdateTopicEmbeddings(topic=%d) failed: %v", topicID, err)
		}
	}(id)

	writeJSON(w, http.StatusOK, map[string]any{
		"topic_id":     id,
		"memory_id":    req.MemoryID,
		"embed_update": "queued",
	})
}

// ===== Triplets / graph walk =====

type tripletsResponse struct {
	Triplets []types.Triplet `json:"triplets"`
}

func (s *Server) handleGetTriplets(w http.ResponseWriter, r *http.Request) {
	sourceType := strings.TrimSpace(r.URL.Query().Get("source_type"))
	sourceID := strings.TrimSpace(r.URL.Query().Get("source_id"))
	if sourceType == "" || sourceID == "" {
		writeError(w, http.StatusBadRequest, "source_type and source_id query params are required", "missing_params")
		return
	}
	triplets, err := s.store.GetTripletsBySource(r.Context(), sourceType, sourceID)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error(), "store_failure")
		return
	}
	if triplets == nil {
		triplets = []types.Triplet{}
	}
	writeJSON(w, http.StatusOK, tripletsResponse{Triplets: triplets})
}

type storeTripletsRequest struct {
	Triplets   [][3]string `json:"triplets"`
	SourceType string      `json:"source_type"`
	SourceID   string      `json:"source_id"`
	Confidence float64     `json:"confidence"`
}

func (s *Server) handleStoreTriplets(w http.ResponseWriter, r *http.Request) {
	var req storeTripletsRequest
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error(), "bad_json")
		return
	}
	if len(req.Triplets) == 0 {
		writeError(w, http.StatusBadRequest, "triplets array must not be empty", "no_triplets")
		return
	}
	if req.Confidence == 0 {
		req.Confidence = 1.0
	}
	if req.SourceType == "" {
		req.SourceType = "memory"
	}
	count, err := s.store.StoreTriplets(r.Context(), req.Triplets, req.SourceType, req.SourceID, req.Confidence)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error(), "store_failure")
		return
	}
	writeJSON(w, http.StatusCreated, map[string]any{"stored": count})
}

type walkRequest struct {
	Entities []string `json:"entities"`
	Hops     int      `json:"hops"`
	Limit    int      `json:"limit"`
}

type walkResponse struct {
	Triplets           []types.Triplet `json:"triplets"`
	EntitiesDiscovered []string        `json:"entities_discovered"`
}

func (s *Server) handleWalkGraph(w http.ResponseWriter, r *http.Request) {
	var req walkRequest
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error(), "bad_json")
		return
	}
	if len(req.Entities) == 0 {
		writeError(w, http.StatusBadRequest, "entities array must not be empty", "no_entities")
		return
	}
	hops := req.Hops
	if hops <= 0 {
		hops = s.defaults.Hops
	}
	limit := req.Limit
	if limit <= 0 {
		limit = 50
	}
	triplets, err := s.store.WalkGraph(r.Context(), req.Entities, hops, limit)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error(), "graph_failure")
		return
	}
	if triplets == nil {
		triplets = []types.Triplet{}
	}

	// Compute the union of entities seen in the returned triplets — useful as
	// a quick "what did we touch?" view for clients.
	seen := make(map[string]struct{}, len(req.Entities)*2)
	out := make([]string, 0, len(req.Entities)*2)
	for _, e := range req.Entities {
		key := strings.ToLower(strings.TrimSpace(e))
		if key == "" {
			continue
		}
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		out = append(out, e)
	}
	for _, t := range triplets {
		for _, e := range []string{t.Subject, t.Object} {
			key := strings.ToLower(strings.TrimSpace(e))
			if key == "" {
				continue
			}
			if _, ok := seen[key]; ok {
				continue
			}
			seen[key] = struct{}{}
			out = append(out, e)
		}
	}

	writeJSON(w, http.StatusOK, walkResponse{
		Triplets:           triplets,
		EntitiesDiscovered: out,
	})
}

// ===== Worker memory =====

type workerRememberRequest struct {
	Content  string         `json:"content"`
	Category string         `json:"category"`
	Metadata map[string]any `json:"metadata"`
}

func (s *Server) handleWorkerRemember(w http.ResponseWriter, r *http.Request) {
	name := strings.TrimSpace(r.PathValue("name"))
	if name == "" {
		writeError(w, http.StatusBadRequest, "worker name is required", "missing_worker")
		return
	}
	var req workerRememberRequest
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error(), "bad_json")
		return
	}
	if strings.TrimSpace(req.Content) == "" {
		writeError(w, http.StatusBadRequest, "content is required", "missing_content")
		return
	}
	if req.Category == "" {
		req.Category = string(types.CategoryGeneral)
	}
	mem, err := s.store.WorkerRemember(r.Context(), name, req.Content, req.Category, req.Metadata)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error(), "store_failure")
		return
	}
	writeJSON(w, http.StatusCreated, mem)
}

type workerRecallRequest struct {
	Query string `json:"query"`
	Limit int    `json:"limit"`
}

func (s *Server) handleWorkerRecall(w http.ResponseWriter, r *http.Request) {
	name := strings.TrimSpace(r.PathValue("name"))
	if name == "" {
		writeError(w, http.StatusBadRequest, "worker name is required", "missing_worker")
		return
	}
	var req workerRecallRequest
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error(), "bad_json")
		return
	}
	if strings.TrimSpace(req.Query) == "" {
		writeError(w, http.StatusBadRequest, "query is required", "missing_query")
		return
	}
	limit := req.Limit
	if limit <= 0 {
		limit = 10
	}
	results, err := s.store.WorkerRecall(r.Context(), name, req.Query, limit)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error(), "store_failure")
		return
	}
	if results == nil {
		results = []types.SearchResult{}
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"worker_name": name,
		"query":       req.Query,
		"results":     results,
	})
}

func (s *Server) handleWorkerForget(w http.ResponseWriter, r *http.Request) {
	name := strings.TrimSpace(r.PathValue("name"))
	if name == "" {
		writeError(w, http.StatusBadRequest, "worker name is required", "missing_worker")
		return
	}
	id, err := parseInt64Path(r, "id")
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error(), "bad_id")
		return
	}
	deleted, err := s.store.WorkerForget(r.Context(), name, id)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error(), "store_failure")
		return
	}
	if !deleted {
		writeJSON(w, http.StatusNotFound, map[string]any{
			"deleted": false,
			"id":      id,
			"error":   "not found",
		})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"deleted": true, "id": id})
}

func (s *Server) handleWorkerMemories(w http.ResponseWriter, r *http.Request) {
	name := strings.TrimSpace(r.PathValue("name"))
	if name == "" {
		writeError(w, http.StatusBadRequest, "worker name is required", "missing_worker")
		return
	}
	limit := int(parseInt64Query(r, "limit", 50))

	// The storage layer doesn't expose a "list worker shared memories"
	// method directly — WorkerRecall("") falls back to most-recent ordering,
	// which is exactly what we want here.
	results, err := s.store.WorkerRecall(r.Context(), name, "", limit)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error(), "store_failure")
		return
	}
	mems := make([]types.Memory, 0, len(results))
	for _, r := range results {
		mems = append(mems, r.Memory)
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"worker_name": name,
		"memories":    mems,
	})
}

type workerHistoryPostRequest struct {
	TaskID      string     `json:"task_id"`
	Description string     `json:"description"`
	Outcome     string     `json:"outcome"`
	SkillsUsed  string     `json:"skills_used"`
	Summary     string     `json:"summary"`
	StartedAt   *time.Time `json:"started_at"`
}

func (s *Server) handleWorkerHistoryPost(w http.ResponseWriter, r *http.Request) {
	name := strings.TrimSpace(r.PathValue("name"))
	if name == "" {
		writeError(w, http.StatusBadRequest, "worker name is required", "missing_worker")
		return
	}
	var req workerHistoryPostRequest
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error(), "bad_json")
		return
	}
	resume, err := s.store.RecordTaskCompletion(r.Context(), name, req.TaskID, req.Description, req.Outcome, req.SkillsUsed, req.Summary, req.StartedAt)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error(), "store_failure")
		return
	}
	writeJSON(w, http.StatusCreated, resume)
}

func (s *Server) handleWorkerHistoryGet(w http.ResponseWriter, r *http.Request) {
	name := strings.TrimSpace(r.PathValue("name"))
	if name == "" {
		writeError(w, http.StatusBadRequest, "worker name is required", "missing_worker")
		return
	}
	limit := int(parseInt64Query(r, "limit", 20))
	resumes, err := s.store.GetWorkerResume(r.Context(), name, limit)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error(), "store_failure")
		return
	}
	if resumes == nil {
		resumes = []types.WorkerResume{}
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"worker_name": name,
		"history":     resumes,
	})
}

// ===== Conversation =====

type conversationPostRequest struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

func (s *Server) handleConversationPost(w http.ResponseWriter, r *http.Request) {
	var req conversationPostRequest
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error(), "bad_json")
		return
	}
	role := strings.TrimSpace(req.Role)
	if role == "" {
		writeError(w, http.StatusBadRequest, "role is required", "missing_role")
		return
	}
	if strings.TrimSpace(req.Content) == "" {
		writeError(w, http.StatusBadRequest, "content is required", "missing_content")
		return
	}
	turn, err := s.store.LogConversation(r.Context(), role, req.Content)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error(), "store_failure")
		return
	}
	writeJSON(w, http.StatusCreated, turn)
}

func (s *Server) handleConversationGet(w http.ResponseWriter, r *http.Request) {
	limit := int(parseInt64Query(r, "limit", 20))
	hours := parseFloatQuery(r, "hours", 0)
	turns, err := s.store.GetRecentConversations(r.Context(), limit, hours)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error(), "store_failure")
		return
	}
	if turns == nil {
		turns = []types.ConversationTurn{}
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"turns": turns,
		"limit": limit,
		"hours": hours,
	})
}

// ===== Default-application helpers =====

func (s *Server) applyTopicK(v int) int {
	if v <= 0 {
		return s.defaults.TopicK
	}
	return v
}
func (s *Server) applyActivationThreshold(v float64) float64 {
	if v <= 0 {
		return s.defaults.ActivationThreshold
	}
	return v
}
func (s *Server) applyMemoryK(v int) int {
	if v <= 0 {
		return s.defaults.MemoryK
	}
	return v
}
func (s *Server) applyThreshold(v float64) float64 {
	if v <= 0 {
		return s.defaults.Threshold
	}
	return v
}
func (s *Server) applyHops(v int) int {
	if v <= 0 {
		return s.defaults.Hops
	}
	return v
}
func (s *Server) applyGraphDiscount(v float64) float64 {
	if v <= 0 {
		return s.defaults.GraphDiscount
	}
	return v
}
func (s *Server) applyLimit(v int) int {
	if v <= 0 {
		return s.defaults.Limit
	}
	return v
}
