-- 002_topic_registry.sql
-- First-class topic registry. Topics are agent-curated semantic buckets used
-- for dendritic recall Tier 1 routing.

CREATE TABLE IF NOT EXISTS topic_registry (
    id           SERIAL PRIMARY KEY,
    name         TEXT NOT NULL UNIQUE,                  -- slug, e.g. "golang-patterns"
    display_name TEXT NOT NULL,                         -- human readable, e.g. "Go Patterns"
    description  TEXT NOT NULL DEFAULT '',              -- semantic description used for embedding routing
    source       TEXT NOT NULL DEFAULT 'curated',       -- 'curated' or 'emergent'
    memory_count INTEGER NOT NULL DEFAULT 0,            -- denormalized, updated on insert/delete
    embed_count  INTEGER NOT NULL DEFAULT 0,            -- number of representative embeddings
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Wire director_memory.topic_id to the topic registry.
-- ON DELETE SET NULL: deleting a topic detaches memories, keeps them alive.
ALTER TABLE director_memory
    DROP CONSTRAINT IF EXISTS director_memory_topic_id_fkey;

ALTER TABLE director_memory
    ADD CONSTRAINT director_memory_topic_id_fkey
    FOREIGN KEY (topic_id) REFERENCES topic_registry(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_topic_registry_name ON topic_registry(name);
CREATE INDEX IF NOT EXISTS idx_topic_registry_source ON topic_registry(source);
