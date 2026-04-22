-- 001_initial_schema.sql
-- Initial schema for Nous Go: director memory, conversations, worker shared memory,
-- worker history (resumes), and knowledge graph triplets.
-- Matches the Python Nous schema for backward compatibility against the same DB.

CREATE TABLE IF NOT EXISTS director_memory (
    id              SERIAL PRIMARY KEY,
    content         TEXT NOT NULL,
    category        TEXT NOT NULL DEFAULT 'general',
    topic_id        INTEGER DEFAULT NULL,           -- FK to topic_registry (added in 002)
    embedding       BYTEA DEFAULT NULL,             -- float32 little-endian binary
    embedding_model TEXT DEFAULT NULL,              -- e.g. "nomic-embed-text"
    importance      REAL DEFAULT 0.5,               -- 0.0–1.0
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS conversations (
    id         SERIAL PRIMARY KEY,
    role       TEXT NOT NULL,                       -- "user", "assistant", "system"
    content    TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS worker_memory (
    id          SERIAL PRIMARY KEY,
    worker_name TEXT NOT NULL,
    content     TEXT NOT NULL,
    category    TEXT NOT NULL DEFAULT 'general',
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS worker_history (
    id               SERIAL PRIMARY KEY,
    worker_name      TEXT NOT NULL,
    task_id          TEXT,
    task_description TEXT,
    outcome          TEXT,                          -- "completed", "failed"
    skills_used      TEXT,
    summary          TEXT,
    started_at       TIMESTAMPTZ,
    finished_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS triplets (
    id          SERIAL PRIMARY KEY,
    subject     TEXT NOT NULL,
    predicate   TEXT NOT NULL,
    object      TEXT NOT NULL,
    source_type TEXT NOT NULL DEFAULT 'conversation',
    source_id   TEXT NOT NULL DEFAULT '',
    confidence  REAL DEFAULT 1.0,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_director_memory_category ON director_memory(category);
CREATE INDEX IF NOT EXISTS idx_director_memory_topic ON director_memory(topic_id) WHERE topic_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_worker_memory_worker ON worker_memory(worker_name);
CREATE INDEX IF NOT EXISTS idx_worker_history_worker ON worker_history(worker_name);
CREATE INDEX IF NOT EXISTS idx_conversations_created ON conversations(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_triplets_subject ON triplets(subject);
CREATE INDEX IF NOT EXISTS idx_triplets_object ON triplets(object);
CREATE INDEX IF NOT EXISTS idx_triplets_source ON triplets(source_type, source_id);
