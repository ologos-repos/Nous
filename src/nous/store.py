"""
Nous MemoryStore — the hybrid PostgreSQL + SQLite storage layer.

This is the central interface for all memory operations. It manages:
- Tier 1: Director memory (PostgreSQL, curated, embedded)
- Tier 2: Worker shared memory (PostgreSQL, name-scoped)
- Tier 3: Worker private shells (per-worker SQLite databases)
- Conversation log (PostgreSQL)
- Worker resumes / history (PostgreSQL)

Usage:
    store = await MemoryStore.connect(
        postgres_url="postgresql://user:pass@localhost/nous",
        shell_dir="./shells",
    )

    # Director memory (Tier 1)
    await store.remember("important fact", category="decision")
    results = await store.recall("what was that decision?")

    # Worker shared memory (Tier 2)
    await store.worker_remember("alpha", "found a bug in auth", category="lesson")
    results = await store.worker_recall("alpha", "auth bug")

    # Worker private shell (Tier 3)
    shell = store.get_shell("alpha")
    await shell.remember("private note", importance=0.8)

    # Conversation log
    await store.log_conversation("user", "hello")
    turns = await store.get_recent_conversations(limit=20)
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from nous.embeddings import EmbeddingProvider, NullEmbedder
from nous.types import (
    ConversationTurn,
    GraphContext,
    Memory,
    MemoryTier,
    RetentionPolicy,
    SearchResult,
    Triplet,
    WorkerResume,
    WorkerShell,
)
from nous.vectors import cosine_similarity, deserialize_vector, serialize_vector

logger = logging.getLogger(__name__)

# ─── Search Query Utilities ────────────────────────────────────────────────────

_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "for", "of",
    "and", "or", "but", "not", "with", "from", "by", "as", "was", "were",
    "been", "be", "are", "do", "does", "did", "have", "has", "had", "will",
    "would", "could", "should", "can", "may", "might", "shall", "i", "you",
    "he", "she", "we", "they", "me", "him", "her", "us", "them", "my",
    "your", "his", "its", "our", "their", "this", "that", "these", "those",
    "what", "which", "who", "whom", "how", "when", "where", "why", "if",
    "then", "than", "so", "no", "yes", "up", "out", "about", "just", "also",
    "very", "really", "before", "after", "into", "over", "such", "some",
    "any", "all", "each", "every", "both", "few", "more", "most", "other",
    "only", "own", "same", "too", "here", "there", "again", "once", "being",
    "doing",
})


def extract_search_queries(text: str) -> list[str]:
    """
    Generate 1-3 search queries from a user message.

    Always includes the original text. Attempts to extract key phrases,
    technical terms, and significant keywords to broaden recall coverage.

    Args:
        text: The input query text

    Returns:
        List of 1-3 deduplicated search queries
    """
    queries: list[str] = [text]

    key_phrases: list[str] = []

    # Multi-word capitalized sequences (e.g. "Centauri Carbon", "Memory Store")
    for match in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', text):
        key_phrases.append(match.group(1))

    # Quoted strings
    for match in re.finditer(r'"([^"]+)"|\'([^\']+)\'', text):
        phrase = match.group(1) or match.group(2)
        if phrase:
            key_phrases.append(phrase)

    # Technical terms (words with dots/hyphens/underscores)
    for match in re.finditer(r'\b[\w][\w.-]*[-._][\w.-]*[\w]\b', text):
        key_phrases.append(match.group(0))

    # Single capitalized words not in stop words
    for match in re.finditer(r'\b([A-Z][a-z]+)\b', text):
        word = match.group(1)
        if word.lower() not in _STOP_WORDS:
            key_phrases.append(word)

    if key_phrases:
        # Use the first/best key phrase as a focused query
        candidate = key_phrases[0]
        if candidate.lower() != text.lower():
            queries.append(candidate)

    # Significant keywords (words >4 chars, not stop words)
    words = re.findall(r'\b\w+\b', text.lower())
    significant = [w for w in words if len(w) > 4 and w not in _STOP_WORDS]
    if significant:
        keyword_query = " ".join(significant)
        if keyword_query.lower() != text.lower() and keyword_query not in queries:
            queries.append(keyword_query)

    # Deduplicate and cap at 3
    seen: set[str] = set()
    result: list[str] = []
    for q in queries:
        normalized = q.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
        if len(result) >= 3:
            break

    return result


class MemoryStore:
    """
    Hybrid PostgreSQL + SQLite memory store.

    PostgreSQL holds shared state (director memory, conversations, worker shared memory,
    resumes). Per-worker SQLite databases hold private state (shell memories, knowledge,
    instructions, training data).
    """

    def __init__(
        self,
        pg_pool: Any,  # asyncpg.Pool
        shell_dir: str | Path,
        embedder: EmbeddingProvider | None = None,
    ):
        self._pool = pg_pool
        self._shell_dir = Path(shell_dir)
        self._shell_dir.mkdir(parents=True, exist_ok=True)
        self._embedder = embedder or NullEmbedder()
        self._shells: dict[str, ShellStore] = {}

    @classmethod
    async def connect(
        cls,
        postgres_url: str,
        shell_dir: str | Path = "./shells",
        embedder: EmbeddingProvider | None = None,
        min_pool_size: int = 2,
        max_pool_size: int = 10,
        run_migrations: bool = True,
    ) -> MemoryStore:
        """
        Connect to PostgreSQL and initialize the memory store.

        Args:
            postgres_url: PostgreSQL connection string
            shell_dir: Directory for per-worker SQLite databases
            embedder: Embedding provider (default: NullEmbedder for keyword-only search)
            min_pool_size: Minimum connections in the pool
            max_pool_size: Maximum connections in the pool
            run_migrations: Whether to create tables on connect (default: True)

        Returns:
            Connected MemoryStore instance
        """
        import asyncpg

        pool = await asyncpg.create_pool(
            postgres_url,
            min_size=min_pool_size,
            max_size=max_pool_size,
        )

        store = cls(pg_pool=pool, shell_dir=shell_dir, embedder=embedder)

        if run_migrations:
            await store._run_migrations()

        return store

    async def close(self):
        """Close all connections."""
        for shell in self._shells.values():
            await shell.close()
        self._shells.clear()
        await self._pool.close()

    # ─── Schema Migrations ─────────────────────────────────────────────

    async def _run_migrations(self):
        """Create tables if they don't exist."""
        async with self._pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS director_memory (
                    id SERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    category TEXT DEFAULT 'general',
                    embedding BYTEA DEFAULT NULL,
                    embedding_model TEXT DEFAULT NULL,
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                );

                CREATE TABLE IF NOT EXISTS conversations (
                    id SERIAL PRIMARY KEY,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );

                CREATE TABLE IF NOT EXISTS worker_memory (
                    id SERIAL PRIMARY KEY,
                    worker_name TEXT NOT NULL,
                    content TEXT NOT NULL,
                    category TEXT DEFAULT 'general',
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );

                CREATE TABLE IF NOT EXISTS worker_history (
                    id SERIAL PRIMARY KEY,
                    worker_name TEXT NOT NULL,
                    task_id TEXT,
                    task_description TEXT,
                    outcome TEXT,
                    skills_used TEXT,
                    summary TEXT,
                    started_at TIMESTAMPTZ,
                    finished_at TIMESTAMPTZ DEFAULT NOW()
                );

                CREATE TABLE IF NOT EXISTS document_chunks (
                    id SERIAL PRIMARY KEY,
                    doc_path TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );

                CREATE TABLE IF NOT EXISTS document_embeddings (
                    id SERIAL PRIMARY KEY,
                    chunk_id INTEGER REFERENCES document_chunks(id) ON DELETE CASCADE,
                    embedding BYTEA NOT NULL,
                    vector_version TEXT DEFAULT '1',
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );

                CREATE INDEX IF NOT EXISTS idx_director_memory_category
                    ON director_memory(category);
                CREATE INDEX IF NOT EXISTS idx_worker_memory_worker
                    ON worker_memory(worker_name);
                CREATE INDEX IF NOT EXISTS idx_worker_history_worker
                    ON worker_history(worker_name);
                CREATE INDEX IF NOT EXISTS idx_conversations_created
                    ON conversations(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_document_chunks_path
                    ON document_chunks(doc_path);
            """)

            # Knowledge graph triplets — universal, not scoped to conversations.
            # source_type: 'conversation', 'memory', 'project', 'task', 'note', 'worker_resume', etc.
            # source_id: the ID within that source system (as text to be flexible).
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS triplets (
                    id SERIAL PRIMARY KEY,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    source_type TEXT NOT NULL DEFAULT 'conversation',
                    source_id TEXT NOT NULL DEFAULT '',
                    confidence REAL DEFAULT 1.0,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_triplets_subject ON triplets(subject);
                CREATE INDEX IF NOT EXISTS idx_triplets_object ON triplets(object);
                CREATE INDEX IF NOT EXISTS idx_triplets_source ON triplets(source_type, source_id);
            """)

    # ─── Tier 1: Director Memory ───────────────────────────────────────

    async def remember(
        self,
        content: str,
        category: str = "general",
        metadata: dict[str, Any] | None = None,
        embed: bool = True,
    ) -> Memory:
        """
        Store a director memory (Tier 1).

        Args:
            content: The memory content
            category: Memory category (e.g., "decision", "lesson", "fact")
            metadata: Optional JSON metadata
            embed: Whether to generate an embedding (default: True)

        Returns:
            The stored Memory object
        """
        embedding_data = None
        embedding_model = None

        if embed and not isinstance(self._embedder, NullEmbedder):
            try:
                vector = await self._embedder.embed(content)
                embedding_data = serialize_vector(vector)
                embedding_model = self._embedder.model_name
            except Exception as e:
                logger.warning(f"Embedding failed, storing without embedding: {e}")

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO director_memory (content, category, embedding, embedding_model, metadata)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING id, created_at
                """,
                content,
                category,
                embedding_data,
                embedding_model,
                metadata or {},
            )

        return Memory(
            id=row["id"],
            content=content,
            category=category,
            tier=MemoryTier.DIRECTOR,
            metadata=metadata or {},
            created_at=row["created_at"],
        )

    async def recall(
        self,
        query: str,
        category: str | None = None,
        limit: int = 10,
        threshold: float = 0.3,
    ) -> list[SearchResult]:
        """
        Search director memory (Tier 1). Uses multi-query hybrid search.

        Generates 1-3 search queries from the input, runs hybrid recall (semantic
        + OR-keyword + recency boost) for each, deduplicates by memory ID keeping
        the highest score, and returns the top results.

        Args:
            query: Search query
            category: Optional category filter
            limit: Maximum results
            threshold: Minimum relevance score for semantic results

        Returns:
            List of SearchResult objects sorted by relevance
        """
        queries = extract_search_queries(query)
        merged: dict[int, SearchResult] = {}

        for q in queries:
            try:
                results = await self.hybrid_recall(
                    q, category=category, limit=limit, threshold=threshold
                )
                for sr in results:
                    existing = merged.get(sr.memory.id)
                    if existing is None or sr.score > existing.score:
                        merged[sr.memory.id] = sr
            except Exception as e:
                logger.warning(f"hybrid_recall failed for query '{q}': {e}")

        sorted_results = sorted(merged.values(), key=lambda r: r.score, reverse=True)
        return sorted_results[:limit]

    async def _semantic_recall(
        self,
        query: str,
        category: str | None,
        limit: int,
        threshold: float,
    ) -> list[SearchResult]:
        """Semantic search using embedding similarity."""
        query_vector = await self._embedder.embed(query)
        if not query_vector:
            return []

        # Fetch all memories with embeddings
        async with self._pool.acquire() as conn:
            if category:
                rows = await conn.fetch(
                    """SELECT id, content, category, embedding, metadata, created_at, updated_at
                       FROM director_memory WHERE category = $1 AND embedding IS NOT NULL""",
                    category,
                )
            else:
                rows = await conn.fetch(
                    """SELECT id, content, category, embedding, metadata, created_at, updated_at
                       FROM director_memory WHERE embedding IS NOT NULL"""
                )

        results = []
        for row in rows:
            stored_vector = deserialize_vector(row["embedding"])
            score = cosine_similarity(query_vector, stored_vector)
            if score >= threshold:
                memory = Memory(
                    id=row["id"],
                    content=row["content"],
                    category=row["category"],
                    tier=MemoryTier.DIRECTOR,
                    metadata=dict(row["metadata"]) if row["metadata"] else {},
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
                results.append(SearchResult(memory=memory, score=score, match_type="semantic"))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    async def _keyword_recall(
        self,
        query: str,
        category: str | None,
        limit: int,
    ) -> list[SearchResult]:
        """Keyword fallback search — AND-style LIKE matching, sorted by recency."""
        words = query.lower().split()
        if not words:
            return []

        conditions = ["LOWER(content) LIKE $" + str(i + 1) for i in range(len(words))]
        params = [f"%{w}%" for w in words]

        if category:
            conditions.append(f"category = ${len(words) + 1}")
            params.append(category)

        where = " AND ".join(conditions)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""SELECT id, content, category, metadata, created_at, updated_at
                    FROM director_memory
                    WHERE {where}
                    ORDER BY created_at DESC
                    LIMIT {limit}""",
                *params,
            )

        results = []
        for i, row in enumerate(rows):
            memory = Memory(
                id=row["id"],
                content=row["content"],
                category=row["category"],
                tier=MemoryTier.DIRECTOR,
                metadata=dict(row["metadata"]) if row["metadata"] else {},
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            # Score by position (first = highest relevance)
            score = 1.0 - (i * 0.05)
            results.append(SearchResult(memory=memory, score=max(score, 0.1), match_type="keyword"))

        return results

    async def _keyword_recall_or(
        self,
        query: str,
        category: str | None,
        limit: int,
    ) -> list[SearchResult]:
        """
        OR-style keyword search — any word match scores a result.

        Unlike _keyword_recall() which requires ALL words to match (AND), this
        method returns results that match ANY significant word. Results are scored
        by the fraction of query words found in the content.

        Falls back to _keyword_recall() if no significant words are found.
        """
        words = re.findall(r'\b\w+\b', query.lower())
        significant = [w for w in words if len(w) > 2 and w not in _STOP_WORDS]

        if not significant:
            return await self._keyword_recall(query, category, limit)

        total_words = len(significant)

        # Build OR conditions with asyncpg $N placeholders
        conditions = [f"LOWER(content) LIKE ${i + 1}" for i in range(len(significant))]
        params: list[Any] = [f"%{w}%" for w in significant]

        if category:
            conditions.append(f"category = ${len(significant) + 1}")
            params.append(category)

        where = "(" + " OR ".join(conditions[:len(significant)]) + ")"
        if category:
            where += f" AND category = ${len(significant) + 1}"

        fetch_limit = limit * 3

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""SELECT id, content, category, metadata, created_at, updated_at
                    FROM director_memory
                    WHERE {where}
                    ORDER BY created_at DESC
                    LIMIT {fetch_limit}""",
                *params,
            )

        results = []
        for row in rows:
            content_lower = row["content"].lower()
            matched = sum(1 for w in significant if w in content_lower)
            score = (matched / total_words) * 0.7
            if score <= 0:
                continue
            memory = Memory(
                id=row["id"],
                content=row["content"],
                category=row["category"],
                tier=MemoryTier.DIRECTOR,
                metadata=dict(row["metadata"]) if row["metadata"] else {},
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            results.append(SearchResult(memory=memory, score=score, match_type="keyword"))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    async def hybrid_recall(
        self,
        query: str,
        category: str | None = None,
        limit: int = 15,
        threshold: float = 0.3,
        recency_boost_hours: float = 24.0,
        recency_boost_value: float = 0.1,
    ) -> list[SearchResult]:
        """
        Hybrid semantic + keyword recall with recency boost.

        Combines semantic similarity (when embeddings are available) with
        OR-style keyword search, deduplicates by memory ID keeping the highest
        score, then applies a recency boost for recent memories.

        Args:
            query: Search query
            category: Optional category filter
            limit: Maximum results to return
            threshold: Minimum score for semantic results
            recency_boost_hours: Hours within which memories get a boost
            recency_boost_value: Score bonus for recent memories (capped at 1.0)

        Returns:
            List of SearchResult objects sorted by score descending
        """
        merged: dict[int, SearchResult] = {}

        # Semantic search (if embedder is available)
        if not isinstance(self._embedder, NullEmbedder):
            try:
                semantic_results = await self._semantic_recall(
                    query, category, limit * 2, threshold
                )
                for sr in semantic_results:
                    merged[sr.memory.id] = sr
            except Exception as e:
                logger.warning(f"Semantic search failed in hybrid_recall: {e}")

        # OR-style keyword search
        try:
            keyword_results = await self._keyword_recall_or(query, category, limit)
            for sr in keyword_results:
                existing = merged.get(sr.memory.id)
                if existing is None or sr.score > existing.score:
                    # Mark as hybrid if it also appeared in semantic results
                    match_type = "hybrid" if sr.memory.id in merged else "keyword"
                    merged[sr.memory.id] = SearchResult(
                        memory=sr.memory,
                        score=sr.score,
                        match_type=match_type,
                    )
                elif existing is not None:
                    # It was already semantic — upgrade to hybrid
                    merged[sr.memory.id] = SearchResult(
                        memory=existing.memory,
                        score=existing.score,
                        match_type="hybrid",
                    )
        except Exception as e:
            logger.warning(f"Keyword search failed in hybrid_recall: {e}")

        # Apply recency boost
        now = datetime.now(timezone.utc)
        boost_cutoff = timedelta(hours=recency_boost_hours)
        boosted: list[SearchResult] = []
        for sr in merged.values():
            score = sr.score
            if sr.memory.created_at:
                created = sr.memory.created_at
                # Ensure timezone-aware for comparison
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                if (now - created) < boost_cutoff:
                    score = min(1.0, score + recency_boost_value)
            boosted.append(SearchResult(
                memory=sr.memory,
                score=score,
                match_type=sr.match_type,
            ))

        boosted.sort(key=lambda r: r.score, reverse=True)
        return boosted[:limit]

    async def forget(self, memory_id: int) -> bool:
        """Delete a specific director memory by ID."""
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM director_memory WHERE id = $1", memory_id
            )
            return result == "DELETE 1"

    async def prune_memories(self, max_age_days: int = 90) -> int:
        """Delete director memories older than max_age_days. Returns count deleted."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM director_memory WHERE created_at < $1", cutoff
            )
            count = int(result.split()[-1])
            return count

    async def get_all_memories(
        self, category: str | None = None, limit: int = 100
    ) -> list[Memory]:
        """Get all director memories, optionally filtered by category."""
        async with self._pool.acquire() as conn:
            if category:
                rows = await conn.fetch(
                    """SELECT id, content, category, metadata, created_at, updated_at
                       FROM director_memory WHERE category = $1
                       ORDER BY created_at DESC LIMIT $2""",
                    category,
                    limit,
                )
            else:
                rows = await conn.fetch(
                    """SELECT id, content, category, metadata, created_at, updated_at
                       FROM director_memory ORDER BY created_at DESC LIMIT $1""",
                    limit,
                )

        return [
            Memory(
                id=row["id"],
                content=row["content"],
                category=row["category"],
                tier=MemoryTier.DIRECTOR,
                metadata=dict(row["metadata"]) if row["metadata"] else {},
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in rows
        ]

    # ─── Tier 2: Worker Shared Memory ──────────────────────────────────

    async def worker_remember(
        self,
        worker_name: str,
        content: str,
        category: str = "general",
        metadata: dict[str, Any] | None = None,
    ) -> Memory:
        """Store a worker shared memory (Tier 2). Name-scoped."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """INSERT INTO worker_memory (worker_name, content, category, metadata)
                   VALUES ($1, $2, $3, $4)
                   RETURNING id, created_at""",
                worker_name,
                content,
                category,
                metadata or {},
            )

        return Memory(
            id=row["id"],
            content=content,
            category=category,
            tier=MemoryTier.SHARED,
            worker_name=worker_name,
            metadata=metadata or {},
            created_at=row["created_at"],
        )

    async def worker_recall(
        self,
        worker_name: str,
        query: str,
        limit: int = 20,
    ) -> list[SearchResult]:
        """Search a worker's shared memories (Tier 2). Keyword search, name-scoped."""
        words = query.lower().split()
        if not words:
            return []

        conditions = ["worker_name = $1"]
        params: list[Any] = [worker_name]
        for i, word in enumerate(words):
            conditions.append(f"LOWER(content) LIKE ${i + 2}")
            params.append(f"%{word}%")

        where = " AND ".join(conditions)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""SELECT id, content, category, metadata, created_at
                    FROM worker_memory WHERE {where}
                    ORDER BY created_at DESC LIMIT {limit}""",
                *params,
            )

        results = []
        for i, row in enumerate(rows):
            memory = Memory(
                id=row["id"],
                content=row["content"],
                category=row["category"],
                tier=MemoryTier.SHARED,
                worker_name=worker_name,
                metadata=dict(row["metadata"]) if row["metadata"] else {},
                created_at=row["created_at"],
            )
            score = 1.0 - (i * 0.05)
            results.append(SearchResult(memory=memory, score=max(score, 0.1), match_type="keyword"))

        return results

    async def worker_forget(self, worker_name: str, memory_id: int) -> bool:
        """Delete a specific worker shared memory. Enforces name-scoping."""
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM worker_memory WHERE id = $1 AND worker_name = $2",
                memory_id,
                worker_name,
            )
            return result == "DELETE 1"

    async def get_worker_memories(
        self, worker_name: str, limit: int = 50, category: str | None = None
    ) -> list[Memory]:
        """Get all shared memories for a worker."""
        async with self._pool.acquire() as conn:
            if category:
                rows = await conn.fetch(
                    """SELECT id, content, category, metadata, created_at
                       FROM worker_memory WHERE worker_name = $1 AND category = $2
                       ORDER BY created_at DESC LIMIT $3""",
                    worker_name,
                    category,
                    limit,
                )
            else:
                rows = await conn.fetch(
                    """SELECT id, content, category, metadata, created_at
                       FROM worker_memory WHERE worker_name = $1
                       ORDER BY created_at DESC LIMIT $2""",
                    worker_name,
                    limit,
                )

        return [
            Memory(
                id=row["id"],
                content=row["content"],
                category=row["category"],
                tier=MemoryTier.SHARED,
                worker_name=worker_name,
                metadata=dict(row["metadata"]) if row["metadata"] else {},
                created_at=row["created_at"],
            )
            for row in rows
        ]

    # ─── Conversation Log ──────────────────────────────────────────────

    async def log_conversation(
        self, role: str, content: str, max_length: int = 10000
    ) -> ConversationTurn:
        """Log a conversation turn. Content truncated to max_length."""
        truncated = content[:max_length] if len(content) > max_length else content
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """INSERT INTO conversations (role, content) VALUES ($1, $2)
                   RETURNING id, created_at""",
                role,
                truncated,
            )
        return ConversationTurn(
            id=row["id"],
            role=role,
            content=truncated,
            created_at=row["created_at"],
        )

    async def get_recent_conversations(
        self,
        limit: int = 20,
        hours_window: float | None = None,
    ) -> list[ConversationTurn]:
        """Get recent conversation turns, optionally within a time window."""
        async with self._pool.acquire() as conn:
            if hours_window:
                cutoff = datetime.now(timezone.utc) - timedelta(hours=hours_window)
                rows = await conn.fetch(
                    """SELECT id, role, content, created_at FROM conversations
                       WHERE created_at > $1 ORDER BY created_at DESC LIMIT $2""",
                    cutoff,
                    limit,
                )
            else:
                rows = await conn.fetch(
                    """SELECT id, role, content, created_at FROM conversations
                       ORDER BY created_at DESC LIMIT $1""",
                    limit,
                )

        turns = [
            ConversationTurn(
                id=row["id"],
                role=row["role"],
                content=row["content"],
                created_at=row["created_at"],
            )
            for row in rows
        ]
        turns.reverse()  # Chronological order
        return turns

    # ─── Worker History / Resumes ──────────────────────────────────────

    async def record_task_completion(
        self,
        worker_name: str,
        task_id: str,
        description: str,
        outcome: str,
        skills_used: str | None = None,
        summary: str | None = None,
        started_at: datetime | None = None,
    ) -> WorkerResume:
        """Record a task completion in the worker's resume."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """INSERT INTO worker_history
                   (worker_name, task_id, task_description, outcome, skills_used, summary, started_at)
                   VALUES ($1, $2, $3, $4, $5, $6, $7)
                   RETURNING finished_at""",
                worker_name,
                task_id,
                description,
                outcome,
                skills_used,
                summary,
                started_at,
            )

        return WorkerResume(
            task_id=task_id,
            description=description,
            outcome=outcome,
            skills_used=skills_used,
            summary=summary,
            started_at=started_at,
            finished_at=row["finished_at"],
        )

    async def get_worker_resume(
        self, worker_name: str, limit: int = 10
    ) -> list[WorkerResume]:
        """Get a worker's recent task history (resume)."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT task_id, task_description, outcome, skills_used, summary,
                          started_at, finished_at
                   FROM worker_history WHERE worker_name = $1
                   ORDER BY finished_at DESC LIMIT $2""",
                worker_name,
                limit,
            )

        return [
            WorkerResume(
                task_id=row["task_id"],
                description=row["task_description"],
                outcome=row["outcome"],
                skills_used=row["skills_used"],
                summary=row["summary"],
                started_at=row["started_at"],
                finished_at=row["finished_at"],
            )
            for row in rows
        ]

    # ─── RAG / Document Search ─────────────────────────────────────────

    async def index_document(
        self,
        doc_path: str,
        chunks: list[str],
        vector_version: str = "1",
    ) -> int:
        """
        Index a document by storing its chunks and embeddings.

        Args:
            doc_path: Path or identifier for the document
            chunks: List of text chunks
            vector_version: Version tag for the embeddings

        Returns:
            Number of chunks indexed
        """
        if not chunks:
            return 0

        # Generate embeddings for all chunks
        embeddings = []
        if not isinstance(self._embedder, NullEmbedder):
            try:
                embeddings = await self._embedder.embed_batch(chunks)
            except Exception as e:
                logger.warning(f"Batch embedding failed for {doc_path}: {e}")

        async with self._pool.acquire() as conn:
            # Clear old chunks for this document
            await conn.execute(
                "DELETE FROM document_chunks WHERE doc_path = $1", doc_path
            )

            for i, chunk in enumerate(chunks):
                # Insert chunk
                row = await conn.fetchrow(
                    """INSERT INTO document_chunks (doc_path, chunk_index, content)
                       VALUES ($1, $2, $3) RETURNING id""",
                    doc_path,
                    i,
                    chunk,
                )
                chunk_id = row["id"]

                # Insert embedding if available
                if i < len(embeddings) and embeddings[i]:
                    await conn.execute(
                        """INSERT INTO document_embeddings (chunk_id, embedding, vector_version)
                           VALUES ($1, $2, $3)""",
                        chunk_id,
                        serialize_vector(embeddings[i]),
                        vector_version,
                    )

        return len(chunks)

    async def search_documents(
        self,
        query: str,
        limit: int = 5,
        threshold: float = 0.4,
    ) -> list[tuple[str, str, float]]:
        """
        Search indexed documents using semantic similarity.

        Returns:
            List of (doc_path, chunk_content, score) tuples
        """
        if isinstance(self._embedder, NullEmbedder):
            return []

        try:
            query_vector = await self._embedder.embed(query)
        except Exception as e:
            logger.warning(f"Document search embedding failed: {e}")
            return []

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT c.doc_path, c.content, e.embedding
                   FROM document_chunks c
                   JOIN document_embeddings e ON e.chunk_id = c.id"""
            )

        results = []
        for row in rows:
            stored_vector = deserialize_vector(row["embedding"])
            score = cosine_similarity(query_vector, stored_vector)
            if score >= threshold:
                results.append((row["doc_path"], row["content"], score))

        results.sort(key=lambda x: x[2], reverse=True)
        return results[:limit]

    # ─── Knowledge Graph ───────────────────────────────────────────────
    #
    # Universal knowledge graph — not scoped to conversations. Any entity in
    # Nous (memories, projects, tasks, worker resumes, notes, …) can be a
    # triplet source. source_type identifies the origin system; source_id is
    # the ID within that system (stored as text for flexibility).

    async def store_triplets(
        self,
        triplets: list[tuple[str, str, str]],
        source_type: str,
        source_id: str,
        confidence: float = 1.0,
    ) -> int:
        """
        Batch INSERT (subject, predicate, object) triplets into the knowledge graph.

        Args:
            triplets: List of (subject, predicate, object) tuples
            source_type: Origin system — 'conversation', 'memory', 'project',
                         'task', 'note', 'worker_resume', etc.
            source_id: ID within the source system (will be cast to str)
            confidence: Confidence score for all triplets in this batch (0.0–1.0)

        Returns:
            Number of triplets inserted
        """
        if not triplets:
            return 0

        source_id_str = str(source_id)
        count = 0
        try:
            async with self._pool.acquire() as conn:
                for subject, predicate, obj in triplets:
                    subject = subject.strip()
                    predicate = predicate.strip()
                    obj = obj.strip()
                    if not subject or not predicate or not obj:
                        continue
                    await conn.execute(
                        """
                        INSERT INTO triplets (subject, predicate, object, source_type, source_id, confidence)
                        VALUES ($1, $2, $3, $4, $5, $6)
                        """,
                        subject,
                        predicate,
                        obj,
                        source_type,
                        source_id_str,
                        float(confidence),
                    )
                    count += 1
        except Exception as e:
            logger.error(
                f"store_triplets failed for source_type={source_type} source_id={source_id}: {e}"
            )
        return count

    async def decompose_turn(
        self,
        turn_id: int | None,
        content: str,
        decomposer=None,
    ) -> list[dict]:
        """
        Extract triplets from a conversation turn and store them in the knowledge graph.

        Convenience wrapper around decompose_text() that sets source_type='conversation'
        and source_id=str(turn_id).

        If decomposer is provided (callable), it is called with content and should
        return a list of (subject, predicate, object) tuples. This is designed to
        accept an LLM call from the caller — Nous stays LLM-agnostic.

        If decomposer is None, a simple heuristic extractor is used as fallback.

        Args:
            turn_id: Source conversation turn ID
            content: Text content to decompose
            decomposer: Optional callable(content) -> list of (subject, predicate, object)

        Returns:
            List of triplet dicts (subject, predicate, object, source_type, source_id, confidence)
        """
        return await self.decompose_text(
            content=content,
            source_type="conversation",
            source_id=str(turn_id) if turn_id is not None else "",
            decomposer=decomposer,
        )

    async def decompose_text(
        self,
        content: str,
        source_type: str,
        source_id: str,
        decomposer=None,
        confidence: float = 1.0,
    ) -> list[dict]:
        """
        Extract (subject, predicate, object) triplets from any text and store them.

        Universal version of decompose_turn — works for any source type (memory,
        project, task, note, worker_resume, conversation, …).

        Args:
            content: Text content to decompose
            source_type: Origin system identifier
            source_id: ID within that system
            decomposer: Optional callable(content) -> list of (subject, predicate, object)
            confidence: Confidence score for stored triplets

        Returns:
            List of triplet dicts stored
        """
        raw_triplets: list[tuple[str, str, str]] = []

        if decomposer is not None:
            try:
                result = decomposer(content)
                # Support both sync and async callables
                if hasattr(result, "__await__"):
                    result = await result
                raw_triplets = [(s, p, o) for s, p, o in result if s and p and o]
            except Exception as e:
                logger.warning(f"decomposer callable failed, falling back to heuristic: {e}")
                raw_triplets = self._heuristic_extract(content)
        else:
            raw_triplets = self._heuristic_extract(content)

        await self.store_triplets(
            raw_triplets,
            source_type=source_type,
            source_id=source_id,
            confidence=confidence,
        )

        return [
            {
                "subject": s,
                "predicate": p,
                "object": o,
                "source_type": source_type,
                "source_id": source_id,
                "confidence": confidence,
            }
            for s, p, o in raw_triplets
        ]

    # Blacklist of words that are never valid entity subjects/objects.
    # Pronouns, determiners, and generic function words that appear as
    # sentence-initial fragments when the regex is too greedy.
    _ENTITY_BLACKLIST: frozenset[str] = frozenset({
        "This", "It", "That", "Here", "There", "Everything", "Nothing",
        "Something", "Always", "Never", "Each", "Every", "These", "Those",
        "The", "A", "An", "We", "They", "You", "He", "She", "My", "Your",
        "His", "Her", "Its", "Our", "Their", "What", "Which", "Who", "Whom",
        "How", "When", "Where", "Why", "Just", "Also", "Since", "Because",
        "Although", "However", "Therefore", "Thus", "Then", "Now", "Still",
        "And", "But", "Or", "So", "Yet",
    })

    # Leading words to strip from captured subjects/objects before validation.
    # Order matters — longest first so "and the" is stripped before "the".
    _STRIP_PREFIXES: tuple[str, ...] = (
        "and the ", "but the ", "or the ", "if the ", "since the ",
        "when the ", "that the ", "as the ",
        "and a ", "but a ", "or a ",
        "and an ", "but an ",
        "and ", "but ", "or ", "since ", "when ", "if ", "that ",
        "the ", "a ", "an ",
        "this ", "that ", "these ", "those ", "it ", "its ",
    )

    def _heuristic_extract(self, content: str) -> list[tuple[str, str, str]]:
        """
        Simple heuristic triplet extractor — fallback when no LLM decomposer is provided.

        Matches patterns like "<Subject> is/has/uses/was/are/can/will/supports/contains <Object>"
        within each sentence. Returns deduplicated (subject, predicate, object) tuples.

        Improvements over the original:
        - Strips leading articles/conjunctions from subjects and objects.
        - Rejects subjects/objects that are pure pronouns or function words.
        - Requires subject to start with a capital letter after stripping.
        - Tighter regex capture (≤20 chars) to avoid sentence-fragment matches.
        - Truncates objects longer than 30 chars (likely sentence fragments).
        """
        triplets: list[tuple[str, str, str]] = []

        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', content.strip())

        # Each pattern: (regex, subject_group, predicate_group, object_group)
        # Subject capture: lazy (≤20 chars) to avoid consuming the predicate.
        # Object capture: greedy (≤30 chars) so multi-word objects like "a language"
        # are captured in full rather than stopping at the first word.
        patterns = [
            # "X is/was/are/were/will be Y"
            (r'\b([A-Za-z][\w\s\-]{1,20}?)\s+(is|was|are|were|will be)\s+([A-Za-z][\w\s\-]{1,30})\b', 1, 2, 3),
            # "X has/have/had Y"
            (r'\b([A-Za-z][\w\s\-]{1,20}?)\s+(has|have|had)\s+([A-Za-z][\w\s\-]{1,30})\b', 1, 2, 3),
            # "X uses/stores/supports/contains/provides/requires/includes Y"
            (r'\b([A-Za-z][\w\s\-]{1,20}?)\s+(uses?|used|stores?|supports?|contains?|provides?|requires?|includes?)\s+([A-Za-z][\w\s\-]{1,30})\b', 1, 2, 3),
            # "X can/will/should/must/may Y"
            (r'\b([A-Za-z][\w\s\-]{1,20}?)\s+(can|will|should|must|may)\s+([A-Za-z][\w\s\-]{1,30})\b', 1, 2, 3),
        ]

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 5:
                continue
            for pattern, subj_idx, pred_idx, obj_idx in patterns:
                for match in re.finditer(pattern, sentence, re.IGNORECASE):
                    subject = match.group(subj_idx).strip().rstrip('.,;:')
                    predicate = match.group(pred_idx).strip()
                    obj = match.group(obj_idx).strip().rstrip('.,;:')

                    # Strip leading articles/conjunctions
                    subject = self._strip_entity_prefixes(subject)
                    obj = self._strip_entity_prefixes(obj)

                    # Truncate objects that are too long (sentence fragments)
                    if len(obj) > 30:
                        obj = obj[:30].rsplit(' ', 1)[0]
                    obj = obj.rstrip('.,;:').strip()

                    # Reject too-short or too-long subjects/objects
                    if len(subject) < 2 or len(obj) < 2:
                        continue
                    if len(subject) > 40 or len(obj) > 40:
                        continue

                    # Require subject to start with a capital letter
                    # (proper nouns, technical names — not sentence fragments)
                    if not subject[0].isupper():
                        continue

                    # Reject blacklisted subjects and objects
                    if subject in self._ENTITY_BLACKLIST:
                        continue
                    if obj in self._ENTITY_BLACKLIST:
                        continue

                    triplets.append((subject, predicate, obj))
                    break  # One match per sentence per pattern

        # Deduplicate (case-insensitive)
        seen: set[tuple[str, str, str]] = set()
        result: list[tuple[str, str, str]] = []
        for t in triplets:
            key = (t[0].lower(), t[1].lower(), t[2].lower())
            if key not in seen:
                seen.add(key)
                result.append(t)

        return result

    def _strip_entity_prefixes(self, text: str) -> str:
        """Strip leading articles, conjunctions, and function words from entity text."""
        lowered = text.lower()
        for prefix in self._STRIP_PREFIXES:
            if lowered.startswith(prefix):
                text = text[len(prefix):]
                lowered = text.lower()
                # Only strip one prefix layer
                break
        return text.strip()

    async def walk_graph(
        self,
        entities: list[str],
        hops: int = 2,
        limit: int = 50,
    ) -> list[dict]:
        """
        Walk the knowledge graph from a set of seed entities.

        Performs up to `hops` SQL hops: each hop fetches all triplets where
        subject or object matches any entity in the current frontier, then
        expands to newly discovered entities for the next hop.

        Args:
            entities: Seed entity strings (subjects or objects to start from)
            hops: Number of traversal hops (1–2 recommended)
            limit: Maximum total triplets to return

        Returns:
            Deduplicated list of triplet dicts, each with keys:
            id, subject, predicate, object, source_type, source_id, confidence, created_at
        """
        if not entities:
            return []

        seen_ids: set[int] = set()
        all_triplets: list[dict] = []
        frontier = list(entities)
        seen_entities: set[str] = set(e.lower() for e in entities)

        try:
            async with self._pool.acquire() as conn:
                for _hop in range(max(1, hops)):
                    if not frontier:
                        break

                    rows = await conn.fetch(
                        f"""
                        SELECT id, subject, predicate, object,
                               source_type, source_id, confidence, created_at
                        FROM triplets
                        WHERE subject = ANY($1::text[]) OR object = ANY($1::text[])
                        LIMIT {limit}
                        """,
                        frontier,
                    )

                    new_entities: list[str] = []
                    for row in rows:
                        if row["id"] in seen_ids:
                            continue
                        seen_ids.add(row["id"])
                        t = {
                            "id": row["id"],
                            "subject": row["subject"],
                            "predicate": row["predicate"],
                            "object": row["object"],
                            "source_type": row["source_type"],
                            "source_id": row["source_id"],
                            "confidence": row["confidence"],
                            "created_at": row["created_at"],
                        }
                        all_triplets.append(t)

                        # Expand frontier with newly discovered entities
                        for entity in (row["subject"], row["object"]):
                            if entity.lower() not in seen_entities:
                                seen_entities.add(entity.lower())
                                new_entities.append(entity)

                    frontier = new_entities
                    if len(all_triplets) >= limit:
                        break

        except Exception as e:
            logger.error(f"walk_graph failed for entities={entities[:3]}: {e}")

        return all_triplets[:limit]

    async def get_triplets_for_source(
        self,
        source_type: str,
        source_id: str | int,
    ) -> list[dict]:
        """
        Get all triplets from a specific source (type + id combination).

        Args:
            source_type: Origin system ('conversation', 'memory', 'task', etc.)
            source_id: ID within that system

        Returns:
            List of triplet dicts
        """
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id, subject, predicate, object,
                           source_type, source_id, confidence, created_at
                    FROM triplets
                    WHERE source_type = $1 AND source_id = $2
                    ORDER BY created_at
                    """,
                    source_type,
                    str(source_id),
                )
            return [
                {
                    "id": row["id"],
                    "subject": row["subject"],
                    "predicate": row["predicate"],
                    "object": row["object"],
                    "source_type": row["source_type"],
                    "source_id": row["source_id"],
                    "confidence": row["confidence"],
                    "created_at": row["created_at"],
                }
                for row in rows
            ]
        except Exception as e:
            logger.error(f"get_triplets_for_source failed ({source_type}/{source_id}): {e}")
            return []

    async def get_triplets_for_turns(self, turn_ids: list[int]) -> list[dict]:
        """
        Get all triplets whose source is a conversation turn.

        Convenience wrapper around get_triplets_for_source for the common
        case of fetching triplets by conversation turn IDs.

        Args:
            turn_ids: List of conversation turn IDs

        Returns:
            List of triplet dicts
        """
        if not turn_ids:
            return []

        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id, subject, predicate, object,
                           source_type, source_id, confidence, created_at
                    FROM triplets
                    WHERE source_type = 'conversation'
                      AND source_id = ANY($1::text[])
                    ORDER BY created_at
                    """,
                    [str(tid) for tid in turn_ids],
                )
            return [
                {
                    "id": row["id"],
                    "subject": row["subject"],
                    "predicate": row["predicate"],
                    "object": row["object"],
                    "source_type": row["source_type"],
                    "source_id": row["source_id"],
                    "confidence": row["confidence"],
                    "created_at": row["created_at"],
                }
                for row in rows
            ]
        except Exception as e:
            logger.error(f"get_triplets_for_turns failed: {e}")
            return []

    async def get_entity_neighborhood(
        self,
        entity: str,
        hops: int = 1,
    ) -> list[dict]:
        """
        Convenience wrapper: walk the graph from a single entity.

        Args:
            entity: Starting entity string
            hops: Number of hops (default 1)

        Returns:
            List of triplet dicts within N hops of the entity
        """
        return await self.walk_graph([entity], hops=hops)

    async def graph_enhanced_recall(
        self,
        query: str,
        category: str | None = None,
        limit: int = 10,
        threshold: float = 0.3,
        hops: int = 2,
    ) -> GraphContext:
        """
        Graph-enhanced memory recall: combines RAG hits with knowledge graph traversal.

        Steps:
        1. hybrid_recall() → top-N RAG hits from director memory
        2. Collect source conversation turn IDs from those hits' metadata
        3. Look up triplets linked to those source turns
        4. Extract unique entities from those triplets
        5. walk_graph() to expand the entity neighborhood
        6. Collect additional conversation turns discovered via graph walk
        7. Return GraphContext with RAG results + graph triplets + discovered turns

        Graph-discovered results represent indirect matches — callers should
        weight them lower than direct RAG hits (suggested: 0.6 * threshold).

        Args:
            query: Search query
            category: Optional memory category filter
            limit: Max RAG results
            threshold: Minimum relevance score for RAG
            hops: Graph traversal depth

        Returns:
            GraphContext with rag_results, graph_triplets, discovered_turns, entities
        """
        # Step 1: RAG hits
        rag_results: list[SearchResult] = []
        try:
            rag_results = await self.hybrid_recall(
                query, category=category, limit=limit, threshold=threshold
            )
        except Exception as e:
            logger.warning(f"graph_enhanced_recall: hybrid_recall failed: {e}")

        # Step 2: Collect conversation source turn IDs from RAG hit metadata
        source_turn_ids: set[int] = set()
        for sr in rag_results:
            tid = sr.memory.metadata.get("source_turn_id")
            if tid is not None:
                try:
                    source_turn_ids.add(int(tid))
                except (ValueError, TypeError):
                    pass

        # Step 3: Get triplets linked to those source turns
        initial_triplets: list[dict] = []
        if source_turn_ids:
            initial_triplets = await self.get_triplets_for_turns(list(source_turn_ids))

        # Step 4: Extract unique entities from initial triplets
        entities: set[str] = set()
        for t in initial_triplets:
            entities.add(t["subject"])
            entities.add(t["object"])

        # Step 5: Walk the graph; fallback to query entities if no seed triplets
        graph_triplets: list[dict] = []
        if entities:
            graph_triplets = await self.walk_graph(list(entities), hops=hops, limit=50)
        elif not initial_triplets:
            # Fallback: try to seed from capitalized terms in the query
            query_entities = [
                m.group(0) for m in re.finditer(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
            ]
            if query_entities:
                graph_triplets = await self.walk_graph(query_entities, hops=hops, limit=50)

        # Step 6: Collect conversation turns discovered via graph walk
        walked_conv_ids: set[int] = set()
        for t in graph_triplets:
            if t.get("source_type") == "conversation" and t.get("source_id"):
                try:
                    walked_conv_ids.add(int(t["source_id"]))
                except (ValueError, TypeError):
                    pass
        new_conv_ids = walked_conv_ids - source_turn_ids

        discovered_turns: list[ConversationTurn] = []
        if new_conv_ids:
            try:
                async with self._pool.acquire() as conn:
                    rows = await conn.fetch(
                        """
                        SELECT id, role, content, created_at
                        FROM conversations
                        WHERE id = ANY($1::int[])
                        ORDER BY created_at
                        """,
                        list(new_conv_ids),
                    )
                discovered_turns = [
                    ConversationTurn(
                        id=row["id"],
                        role=row["role"],
                        content=row["content"],
                        created_at=row["created_at"],
                    )
                    for row in rows
                ]
            except Exception as e:
                logger.warning(f"graph_enhanced_recall: failed to fetch discovered turns: {e}")

        # Step 7: Collect all entities encountered in the walk
        all_entities: set[str] = set()
        for t in graph_triplets:
            all_entities.add(t["subject"])
            all_entities.add(t["object"])

        # Convert raw dicts to Triplet objects
        triplet_objects = [
            Triplet(
                id=t["id"],
                subject=t["subject"],
                predicate=t["predicate"],
                object=t["object"],
                source_type=t.get("source_type", "conversation"),
                source_id=t.get("source_id", ""),
                confidence=t.get("confidence", 1.0),
                created_at=str(t["created_at"]) if t.get("created_at") else None,
            )
            for t in graph_triplets
        ]

        return GraphContext(
            rag_results=rag_results,
            graph_triplets=triplet_objects,
            discovered_turns=discovered_turns,
            entities=all_entities,
        )

    # ─── Tier 3: Worker Shells ─────────────────────────────────────────

    def get_shell(self, worker_name: str) -> ShellStore:
        """
        Get or create a worker's private shell (Tier 3).

        The shell is a per-worker SQLite database that provides:
        - Private memories with importance weighting
        - Structured knowledge entries
        - Standing instructions
        - Training data
        - Task history
        """
        if worker_name not in self._shells:
            db_path = self._shell_dir / f"{worker_name}.db"
            self._shells[worker_name] = ShellStore(worker_name, str(db_path))
        return self._shells[worker_name]

    async def list_shells(self) -> list[WorkerShell]:
        """List all worker shells in the shell directory."""
        shells = []
        for db_file in self._shell_dir.glob("*.db"):
            name = db_file.stem
            shell = self.get_shell(name)
            await shell._ensure_init()
            stats = await shell.stats()
            shells.append(
                WorkerShell(
                    worker_name=name,
                    db_path=str(db_file),
                    memories_count=stats.get("memories", 0),
                    knowledge_count=stats.get("knowledge", 0),
                    instructions_count=stats.get("instructions", 0),
                    tasks_completed=stats.get("tasks", 0),
                )
            )
        return shells


class ShellStore:
    """
    Per-worker SQLite shell (Tier 3).

    Each worker gets an independent SQLite database. This IS the worker's identity —
    portable, self-contained, and evolving across task assignments.
    """

    def __init__(self, worker_name: str, db_path: str):
        self.worker_name = worker_name
        self.db_path = db_path
        self._db = None
        self._initialized = False

    async def _ensure_init(self):
        """Lazy initialization — connect and create tables on first use."""
        if self._initialized:
            return

        import aiosqlite

        self._db = await aiosqlite.connect(self.db_path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA busy_timeout=5000")

        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                category TEXT DEFAULT 'general',
                importance REAL DEFAULT 0.5,
                embedding BLOB DEFAULT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                content TEXT NOT NULL,
                source TEXT,
                confidence REAL DEFAULT 1.0,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS training_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT,
                status TEXT DEFAULT 'active',
                created_at TEXT DEFAULT (datetime('now')),
                completed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS training_pairs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER REFERENCES training_sessions(id),
                input TEXT NOT NULL,
                output TEXT NOT NULL,
                quality_score REAL DEFAULT 1.0
            );

            CREATE TABLE IF NOT EXISTS instructions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                priority INTEGER DEFAULT 0,
                active INTEGER DEFAULT 1,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS task_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT,
                description TEXT,
                outcome TEXT,
                summary TEXT,
                started_at TEXT,
                completed_at TEXT DEFAULT (datetime('now'))
            );
        """)
        await self._db.commit()
        self._initialized = True

    async def close(self):
        """Close the SQLite connection."""
        if self._db:
            await self._db.close()
            self._db = None
            self._initialized = False

    # ─── Private Memories ──────────────────────────────────────────────

    async def remember(
        self,
        content: str,
        category: str = "general",
        importance: float = 0.5,
    ) -> Memory:
        """Store a private memory with importance weighting."""
        await self._ensure_init()
        cursor = await self._db.execute(
            "INSERT INTO memory (content, category, importance) VALUES (?, ?, ?)",
            (content, category, max(0.0, min(1.0, importance))),
        )
        await self._db.commit()

        return Memory(
            id=cursor.lastrowid,
            content=content,
            category=category,
            tier=MemoryTier.PRIVATE,
            importance=importance,
            worker_name=self.worker_name,
        )

    async def recall(self, query: str, limit: int = 20) -> list[SearchResult]:
        """Keyword search over private memories."""
        await self._ensure_init()
        words = query.lower().split()
        if not words:
            return []

        conditions = " AND ".join(["LOWER(content) LIKE ?" for _ in words])
        params = [f"%{w}%" for w in words]

        cursor = await self._db.execute(
            f"""SELECT id, content, category, importance, created_at
                FROM memory WHERE {conditions}
                ORDER BY created_at DESC LIMIT ?""",
            (*params, limit),
        )
        rows = await cursor.fetchall()

        results = []
        for i, row in enumerate(rows):
            memory = Memory(
                id=row[0],
                content=row[1],
                category=row[2],
                tier=MemoryTier.PRIVATE,
                importance=row[3],
                worker_name=self.worker_name,
            )
            score = 1.0 - (i * 0.05)
            results.append(SearchResult(memory=memory, score=max(score, 0.1), match_type="keyword"))

        return results

    async def forget(self, memory_id: int) -> bool:
        """Delete a private memory by ID."""
        await self._ensure_init()
        cursor = await self._db.execute("DELETE FROM memory WHERE id = ?", (memory_id,))
        await self._db.commit()
        return cursor.rowcount > 0

    async def get_memories(
        self, limit: int = 50, category: str | None = None
    ) -> list[Memory]:
        """Get all private memories, optionally filtered by category."""
        await self._ensure_init()
        if category:
            cursor = await self._db.execute(
                """SELECT id, content, category, importance, created_at
                   FROM memory WHERE category = ?
                   ORDER BY importance DESC, created_at DESC LIMIT ?""",
                (category, limit),
            )
        else:
            cursor = await self._db.execute(
                """SELECT id, content, category, importance, created_at
                   FROM memory ORDER BY importance DESC, created_at DESC LIMIT ?""",
                (limit,),
            )
        rows = await cursor.fetchall()

        return [
            Memory(
                id=row[0],
                content=row[1],
                category=row[2],
                tier=MemoryTier.PRIVATE,
                importance=row[3],
                worker_name=self.worker_name,
            )
            for row in rows
        ]

    # ─── Knowledge ─────────────────────────────────────────────────────

    async def learn(
        self,
        topic: str,
        content: str,
        source: str | None = None,
        confidence: float = 1.0,
    ) -> int:
        """Store a structured knowledge entry."""
        await self._ensure_init()
        cursor = await self._db.execute(
            "INSERT INTO knowledge (topic, content, source, confidence) VALUES (?, ?, ?, ?)",
            (topic, content, source, max(0.0, min(1.0, confidence))),
        )
        await self._db.commit()
        return cursor.lastrowid

    async def lookup(self, topic: str, limit: int = 10) -> list[dict]:
        """Search knowledge by topic."""
        await self._ensure_init()
        cursor = await self._db.execute(
            """SELECT id, topic, content, source, confidence, created_at
               FROM knowledge WHERE LOWER(topic) LIKE ?
               ORDER BY confidence DESC LIMIT ?""",
            (f"%{topic.lower()}%", limit),
        )
        rows = await cursor.fetchall()

        return [
            {
                "id": row[0],
                "topic": row[1],
                "content": row[2],
                "source": row[3],
                "confidence": row[4],
                "created_at": row[5],
            }
            for row in rows
        ]

    # ─── Instructions ──────────────────────────────────────────────────

    async def add_instruction(self, content: str, priority: int = 0) -> int:
        """Add a standing instruction (persists across tasks)."""
        await self._ensure_init()
        cursor = await self._db.execute(
            "INSERT INTO instructions (content, priority) VALUES (?, ?)",
            (content, priority),
        )
        await self._db.commit()
        return cursor.lastrowid

    async def get_instructions(self, active_only: bool = True) -> list[dict]:
        """Get standing instructions."""
        await self._ensure_init()
        if active_only:
            cursor = await self._db.execute(
                "SELECT id, content, priority FROM instructions WHERE active = 1 ORDER BY priority DESC"
            )
        else:
            cursor = await self._db.execute(
                "SELECT id, content, priority, active FROM instructions ORDER BY priority DESC"
            )
        rows = await cursor.fetchall()

        return [
            {"id": row[0], "content": row[1], "priority": row[2]}
            for row in rows
        ]

    async def deactivate_instruction(self, instruction_id: int) -> bool:
        """Deactivate a standing instruction."""
        await self._ensure_init()
        cursor = await self._db.execute(
            "UPDATE instructions SET active = 0 WHERE id = ?", (instruction_id,)
        )
        await self._db.commit()
        return cursor.rowcount > 0

    # ─── Training ──────────────────────────────────────────────────────

    async def start_training(self, topic: str) -> int:
        """Start a self-training session."""
        await self._ensure_init()
        cursor = await self._db.execute(
            "INSERT INTO training_sessions (topic) VALUES (?)", (topic,)
        )
        await self._db.commit()
        return cursor.lastrowid

    async def add_training_pair(
        self,
        session_id: int,
        input_text: str,
        output_text: str,
        quality_score: float = 1.0,
    ) -> int:
        """Add an input/output training pair to a session."""
        await self._ensure_init()
        cursor = await self._db.execute(
            "INSERT INTO training_pairs (session_id, input, output, quality_score) VALUES (?, ?, ?, ?)",
            (session_id, input_text, output_text, quality_score),
        )
        await self._db.commit()
        return cursor.lastrowid

    async def complete_training(self, session_id: int) -> bool:
        """Complete a training session."""
        await self._ensure_init()
        cursor = await self._db.execute(
            "UPDATE training_sessions SET status = 'completed', completed_at = datetime('now') WHERE id = ?",
            (session_id,),
        )
        await self._db.commit()
        return cursor.rowcount > 0

    # ─── Task History ──────────────────────────────────────────────────

    async def record_task(
        self,
        task_id: str,
        description: str,
        outcome: str,
        summary: str | None = None,
        started_at: str | None = None,
    ) -> int:
        """Record a task in the shell's private history."""
        await self._ensure_init()
        cursor = await self._db.execute(
            """INSERT INTO task_history (task_id, description, outcome, summary, started_at)
               VALUES (?, ?, ?, ?, ?)""",
            (task_id, description, outcome, summary, started_at),
        )
        await self._db.commit()
        return cursor.lastrowid

    async def get_task_history(self, limit: int = 20) -> list[dict]:
        """Get the shell's task execution history."""
        await self._ensure_init()
        cursor = await self._db.execute(
            """SELECT task_id, description, outcome, summary, started_at, completed_at
               FROM task_history ORDER BY completed_at DESC LIMIT ?""",
            (limit,),
        )
        rows = await cursor.fetchall()

        return [
            {
                "task_id": row[0],
                "description": row[1],
                "outcome": row[2],
                "summary": row[3],
                "started_at": row[4],
                "completed_at": row[5],
            }
            for row in rows
        ]

    # ─── Retention / Pruning ───────────────────────────────────────────

    async def prune(self, policy: RetentionPolicy | None = None) -> int:
        """Prune memories based on retention policy. Returns count deleted."""
        await self._ensure_init()
        policy = policy or RetentionPolicy()

        cursor = await self._db.execute(
            "SELECT id, content, category, importance, created_at FROM memory"
        )
        rows = await cursor.fetchall()

        to_delete = []
        for row in rows:
            memory = Memory(
                id=row[0],
                content=row[1],
                category=row[2],
                tier=MemoryTier.PRIVATE,
                importance=row[3],
                worker_name=self.worker_name,
                created_at=datetime.fromisoformat(row[4]) if row[4] else None,
            )
            if policy.should_prune(memory):
                to_delete.append(memory.id)

        if to_delete:
            placeholders = ",".join("?" * len(to_delete))
            await self._db.execute(
                f"DELETE FROM memory WHERE id IN ({placeholders})", to_delete
            )
            await self._db.commit()

        return len(to_delete)

    # ─── Stats ─────────────────────────────────────────────────────────

    async def stats(self) -> dict:
        """Get shell statistics."""
        await self._ensure_init()
        result = {}

        for table, key in [
            ("memory", "memories"),
            ("knowledge", "knowledge"),
            ("instructions", "instructions"),
            ("task_history", "tasks"),
            ("training_sessions", "training_sessions"),
            ("training_pairs", "training_pairs"),
        ]:
            cursor = await self._db.execute(f"SELECT COUNT(*) FROM {table}")
            row = await cursor.fetchone()
            result[key] = row[0] if row else 0

        return result
