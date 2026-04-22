-- 003_topic_embeddings.sql
-- Per-topic embedding array. Each row is one representative embedding.
-- Topics have N embeddings (NOT one centroid) because embedding models have
-- chunk-length limits — a topic with 200 memories can't collapse to one vector.
--
-- Representative set uses MMR (maximal marginal relevance):
--   K = min(ceil(sqrt(N)), 50) where N = topic memory count
--
-- Same binary format as director_memory.embedding: float32 little-endian,
-- 4 bytes per dimension.

CREATE TABLE IF NOT EXISTS topic_embeddings (
    id          SERIAL PRIMARY KEY,
    topic_id    INTEGER NOT NULL REFERENCES topic_registry(id) ON DELETE CASCADE,
    memory_id   INTEGER NOT NULL REFERENCES director_memory(id) ON DELETE CASCADE,
    embedding   BYTEA NOT NULL,              -- float32 little-endian binary
    embed_model TEXT NOT NULL,               -- model that produced this embedding
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (topic_id, memory_id)             -- one representative row per (topic, memory) pair
);

CREATE INDEX IF NOT EXISTS idx_topic_embeddings_topic ON topic_embeddings(topic_id);
CREATE INDEX IF NOT EXISTS idx_topic_embeddings_memory ON topic_embeddings(memory_id);

-- Migration tracking: schema_versions records which migrations have been applied.
-- The migration runner reads this table to decide which files to execute.
CREATE TABLE IF NOT EXISTS schema_versions (
    version     INTEGER PRIMARY KEY,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    description TEXT
);
