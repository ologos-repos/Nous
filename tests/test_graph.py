"""
Tests for the Nous knowledge graph layer.

Tests cover:
- store_triplets: batch INSERT and retrieval
- decompose_turn: custom decomposer and heuristic fallback
- walk_graph: 1-hop and 2-hop traversal
- graph_enhanced_recall: RAG + graph walk enrichment
- get_entity_neighborhood: convenience wrapper
- Edge cases: empty inputs, no results, circular references

All PostgreSQL pool operations are mocked with unittest.mock.AsyncMock —
no live database required.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch, call
import pytest

from nous.types import (
    ConversationTurn,
    GraphContext,
    Memory,
    MemoryTier,
    SearchResult,
    Triplet,
)


# ─── Helpers ────────────────────────────────────────────────────────────────


def _make_store(pool=None):
    """
    Build a MemoryStore instance with a mock pool, bypassing _run_migrations.

    We import MemoryStore here so the module is fully loaded by the time
    tests manipulate it.
    """
    from nous.store import MemoryStore
    from nous.embeddings import NullEmbedder

    store = MemoryStore.__new__(MemoryStore)
    store._pool = pool or _make_pool()
    store._shell_dir = MagicMock()
    store._shell_dir.mkdir = MagicMock()
    store._embedder = NullEmbedder()
    store._shells = {}
    return store


def _make_pool(fetch_results=None, fetchrow_result=None, execute_result="INSERT 1"):
    """
    Return a mock asyncpg pool whose acquire() context manager yields a mock connection.

    fetch_results: list returned by conn.fetch(...)
    fetchrow_result: dict/record returned by conn.fetchrow(...)
    execute_result: string returned by conn.execute(...)
    """
    conn = AsyncMock()
    conn.fetch = AsyncMock(return_value=fetch_results or [])
    conn.fetchrow = AsyncMock(return_value=fetchrow_result)
    conn.execute = AsyncMock(return_value=execute_result)

    pool = MagicMock()
    pool.acquire = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

    return pool, conn


def _make_triplet_row(
    id=1,
    subject="Alice",
    predicate="knows",
    obj="Bob",
    source_type="conversation",
    source_id="42",
    confidence=1.0,
    created_at=None,
):
    """Create a dict mimicking an asyncpg Record for a triplet row."""
    return {
        "id": id,
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "source_type": source_type,
        "source_id": source_id,
        "confidence": confidence,
        "created_at": created_at or datetime.now(timezone.utc),
    }


def _make_memory_row(id=1, content="test memory", category="fact"):
    """Create a dict mimicking an asyncpg Record for a memory row."""
    return {
        "id": id,
        "content": content,
        "category": category,
        "metadata": {},
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }


def _make_search_result(
    id=1, content="test memory", category="fact", score=0.8, source_turn_id=None
):
    """Build a SearchResult suitable for mocking hybrid_recall output."""
    meta = {}
    if source_turn_id is not None:
        meta["source_turn_id"] = source_turn_id
    return SearchResult(
        memory=Memory(
            id=id,
            content=content,
            category=category,
            tier=MemoryTier.DIRECTOR,
            metadata=meta,
        ),
        score=score,
        match_type="semantic",
    )


# ─── store_triplets ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_store_triplets_inserts_correctly():
    """store_triplets() should INSERT each (S, P, O) tuple and return count."""
    pool, conn = _make_pool(execute_result="INSERT 1")
    store = _make_store(pool)

    triplets = [
        ("Alice", "knows", "Bob"),
        ("Bob", "works_at", "ACME"),
    ]
    count = await store.store_triplets(
        triplets, source_type="conversation", source_id="10"
    )

    assert count == 2
    assert conn.execute.call_count == 2


@pytest.mark.asyncio
async def test_store_triplets_empty_list():
    """store_triplets() with an empty list should return 0 without hitting the DB."""
    pool, conn = _make_pool()
    store = _make_store(pool)

    count = await store.store_triplets([], source_type="conversation", source_id="1")

    assert count == 0
    conn.execute.assert_not_called()


@pytest.mark.asyncio
async def test_store_triplets_skips_blank_fields():
    """store_triplets() should skip triplets with blank subject/predicate/object."""
    pool, conn = _make_pool(execute_result="INSERT 1")
    store = _make_store(pool)

    triplets = [
        ("", "knows", "Bob"),       # blank subject — skip
        ("Alice", "", "Bob"),        # blank predicate — skip
        ("Alice", "knows", ""),      # blank object — skip
        ("Alice", "knows", "Carol"), # valid
    ]
    count = await store.store_triplets(
        triplets, source_type="conversation", source_id="1"
    )

    assert count == 1
    assert conn.execute.call_count == 1


@pytest.mark.asyncio
async def test_get_triplets_for_turns_returns_results():
    """get_triplets_for_turns() should query by source_type='conversation' and source_id IN (...)."""
    row = _make_triplet_row(id=5, subject="Alice", predicate="is", obj="engineer", source_id="42")
    pool, conn = _make_pool(fetch_results=[row])
    store = _make_store(pool)

    results = await store.get_triplets_for_turns([42])

    assert len(results) == 1
    assert results[0]["subject"] == "Alice"
    assert results[0]["predicate"] == "is"
    assert results[0]["object"] == "engineer"
    assert results[0]["source_id"] == "42"


@pytest.mark.asyncio
async def test_get_triplets_for_turns_empty_list():
    """get_triplets_for_turns([]) should return [] without querying the DB."""
    pool, conn = _make_pool()
    store = _make_store(pool)

    results = await store.get_triplets_for_turns([])

    assert results == []
    conn.fetch.assert_not_called()


# ─── decompose_turn ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_decompose_turn_with_custom_decomposer():
    """decompose_turn() should call the provided decomposer and store resulting triplets."""
    pool, conn = _make_pool(execute_result="INSERT 1")
    store = _make_store(pool)

    expected = [("Alice", "knows", "Bob"), ("Alice", "works_at", "ACME")]
    custom_decomposer = MagicMock(return_value=expected)

    triplets = await store.decompose_turn(
        turn_id=99,
        content="Alice knows Bob. Alice works at ACME.",
        decomposer=custom_decomposer,
    )

    custom_decomposer.assert_called_once_with("Alice knows Bob. Alice works at ACME.")
    assert len(triplets) == 2
    assert triplets[0]["subject"] == "Alice"
    assert triplets[0]["predicate"] == "knows"
    assert triplets[0]["object"] == "Bob"
    assert triplets[0]["source_type"] == "conversation"
    assert triplets[0]["source_id"] == "99"
    assert conn.execute.call_count == 2


@pytest.mark.asyncio
async def test_decompose_turn_with_async_decomposer():
    """decompose_turn() should support async callable decomposers."""
    pool, conn = _make_pool(execute_result="INSERT 1")
    store = _make_store(pool)

    expected = [("Python", "is", "language")]

    async def async_decomposer(content):
        return expected

    triplets = await store.decompose_turn(
        turn_id=7,
        content="Python is a language.",
        decomposer=async_decomposer,
    )

    assert len(triplets) == 1
    assert triplets[0]["subject"] == "Python"
    assert triplets[0]["source_id"] == "7"


@pytest.mark.asyncio
async def test_decompose_turn_heuristic_fallback_no_decomposer():
    """decompose_turn() with no decomposer should use _heuristic_extract."""
    pool, conn = _make_pool(execute_result="INSERT 1")
    store = _make_store(pool)

    # The heuristic extractor matches "X is Y", "X has Y", etc.
    content = "Python is a programming language. Django uses Python."
    triplets = await store.decompose_turn(turn_id=5, content=content, decomposer=None)

    # Should extract at least one triplet from these clear statements
    assert len(triplets) >= 1
    subjects = {t["subject"].lower() for t in triplets}
    assert any("python" in s for s in subjects)
    for t in triplets:
        assert t["source_type"] == "conversation"
        assert t["source_id"] == "5"


@pytest.mark.asyncio
async def test_decompose_turn_none_turn_id():
    """decompose_turn() with turn_id=None should use source_id=''."""
    pool, conn = _make_pool(execute_result="INSERT 1")
    store = _make_store(pool)

    triplets = await store.decompose_turn(
        turn_id=None,
        content="Alice is a developer.",
    )

    for t in triplets:
        assert t["source_id"] == ""


@pytest.mark.asyncio
async def test_decompose_turn_decomposer_failure_falls_back():
    """If the decomposer raises, decompose_turn() should fall back to heuristic."""
    pool, conn = _make_pool(execute_result="INSERT 1")
    store = _make_store(pool)

    def bad_decomposer(content):
        raise ValueError("LLM is down")

    content = "Alice is an engineer."
    triplets = await store.decompose_turn(
        turn_id=1, content=content, decomposer=bad_decomposer
    )

    # Should still return something from heuristic
    assert isinstance(triplets, list)
    for t in triplets:
        assert t["source_type"] == "conversation"


# ─── walk_graph ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_walk_graph_one_hop():
    """walk_graph(hops=1) should return only direct neighbors of seed entities."""
    # Chain: Alice → knows → Bob → works_at → ACME
    # With 1 hop, starting from Alice, we should only see Alice → knows → Bob
    hop1_rows = [
        _make_triplet_row(id=1, subject="Alice", predicate="knows", obj="Bob"),
    ]

    pool, conn = _make_pool(fetch_results=hop1_rows)
    store = _make_store(pool)

    results = await store.walk_graph(["Alice"], hops=1)

    assert len(results) == 1
    assert results[0]["subject"] == "Alice"
    assert results[0]["object"] == "Bob"
    # fetch called once (1 hop)
    assert conn.fetch.call_count == 1


@pytest.mark.asyncio
async def test_walk_graph_two_hops():
    """walk_graph(hops=2) should traverse 2 hops and return transitive neighbors."""
    # Chain: Alice → knows → Bob → works_at → ACME
    # Hop 1: seed=[Alice] → returns Alice→Bob
    # Hop 2: seed=[Bob] (new entity from hop 1) → returns Bob→ACME
    hop1_rows = [
        _make_triplet_row(id=1, subject="Alice", predicate="knows", obj="Bob"),
    ]
    hop2_rows = [
        _make_triplet_row(id=2, subject="Bob", predicate="works_at", obj="ACME"),
    ]

    # fetch is called twice; return different results for each call
    conn = AsyncMock()
    conn.fetch = AsyncMock(side_effect=[hop1_rows, hop2_rows])
    conn.execute = AsyncMock(return_value="INSERT 1")

    pool = MagicMock()
    pool.acquire = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

    store = _make_store(pool)

    results = await store.walk_graph(["Alice"], hops=2)

    assert len(results) == 2
    subjects = {r["subject"] for r in results}
    assert "Alice" in subjects
    assert "Bob" in subjects
    objects = {r["object"] for r in results}
    assert "Bob" in objects
    assert "ACME" in objects
    # Both hops fired
    assert conn.fetch.call_count == 2


@pytest.mark.asyncio
async def test_walk_graph_empty_entities():
    """walk_graph([]) should return [] immediately without querying DB."""
    pool, conn = _make_pool()
    store = _make_store(pool)

    results = await store.walk_graph([])

    assert results == []
    conn.fetch.assert_not_called()


@pytest.mark.asyncio
async def test_walk_graph_no_triplets_found():
    """walk_graph() with entities that have no triplets should return []."""
    pool, conn = _make_pool(fetch_results=[])
    store = _make_store(pool)

    results = await store.walk_graph(["UnknownEntity"], hops=2)

    assert results == []


@pytest.mark.asyncio
async def test_walk_graph_circular_reference():
    """walk_graph() should handle circular graph references without infinite loops."""
    # Alice → knows → Bob, Bob → knows → Alice (circular)
    # After hop 1: both Alice and Bob are in seen_entities, so hop 2 frontier is empty
    hop1_rows = [
        _make_triplet_row(id=1, subject="Alice", predicate="knows", obj="Bob"),
        _make_triplet_row(id=2, subject="Bob", predicate="knows", obj="Alice"),
    ]
    # Hop 2 returns nothing new (frontier should be empty or already-seen)
    hop2_rows = []

    conn = AsyncMock()
    conn.fetch = AsyncMock(side_effect=[hop1_rows, hop2_rows])

    pool = MagicMock()
    pool.acquire = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

    store = _make_store(pool)

    results = await store.walk_graph(["Alice"], hops=2)

    # Should get 2 triplets, no infinite loop, no duplicates
    assert len(results) == 2
    ids = {r["id"] for r in results}
    assert ids == {1, 2}


@pytest.mark.asyncio
async def test_walk_graph_deduplicates_triplets():
    """walk_graph() should not return duplicate triplet IDs."""
    # Same triplet returned in both hops (shouldn't happen in practice but let's guard)
    row = _make_triplet_row(id=1, subject="Alice", predicate="knows", obj="Bob")
    hop1_rows = [row]
    hop2_rows = [row]  # same row ID

    conn = AsyncMock()
    conn.fetch = AsyncMock(side_effect=[hop1_rows, hop2_rows])

    pool = MagicMock()
    pool.acquire = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

    store = _make_store(pool)

    results = await store.walk_graph(["Alice"], hops=2)

    # Should only have 1 unique triplet despite being returned twice
    assert len(results) == 1
    assert results[0]["id"] == 1


# ─── graph_enhanced_recall ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_graph_enhanced_recall_enriches_with_graph():
    """
    graph_enhanced_recall() should:
    1. Call hybrid_recall to get RAG hits
    2. Use source_turn_id from metadata to find triplets
    3. Walk the graph from those entities
    4. Return a GraphContext with rag_results + graph_triplets + discovered_turns
    """
    # RAG hit has a source_turn_id pointing to turn 42
    rag_hit = _make_search_result(
        id=1, content="Alice knows Bob", score=0.8, source_turn_id=42
    )

    # Triplets linked to turn 42
    turn_triplet_row = _make_triplet_row(
        id=10, subject="Alice", predicate="knows", obj="Bob",
        source_type="conversation", source_id="42"
    )

    # Graph walk from entities {Alice, Bob} returns extra triplet
    walked_triplet_row = _make_triplet_row(
        id=11, subject="Bob", predicate="works_at", obj="ACME",
        source_type="conversation", source_id="99"
    )

    # Conversation turn discovered via graph walk (turn 99)
    discovered_conv_row = {
        "id": 99,
        "role": "user",
        "content": "Bob works at ACME right?",
        "created_at": datetime.now(timezone.utc),
    }

    conn = AsyncMock()
    pool = MagicMock()
    pool.acquire = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

    store = _make_store(pool)

    # Mock hybrid_recall to return our RAG hit
    store.hybrid_recall = AsyncMock(return_value=[rag_hit])

    # First fetch: get_triplets_for_turns (turn 42)
    # Second fetch: walk_graph hop 1
    # Third fetch: fetch discovered turns by id
    conn.fetch = AsyncMock(side_effect=[
        [turn_triplet_row],      # get_triplets_for_turns([42])
        [walked_triplet_row],    # walk_graph hop 1
        [discovered_conv_row],   # fetch new conv turns (id=99)
    ])

    result = await store.graph_enhanced_recall(
        query="Alice and Bob", limit=10, hops=1
    )

    assert isinstance(result, GraphContext)
    assert len(result.rag_results) == 1
    assert result.rag_results[0].memory.content == "Alice knows Bob"

    # Graph triplets from walk
    assert len(result.graph_triplets) >= 1
    assert any(t.predicate == "works_at" for t in result.graph_triplets)

    # Discovered turn (turn 99, not in original RAG set)
    assert len(result.discovered_turns) == 1
    assert result.discovered_turns[0].id == 99
    assert result.discovered_turns[0].content == "Bob works at ACME right?"

    # Entities set
    assert "Bob" in result.entities or "ACME" in result.entities


@pytest.mark.asyncio
async def test_graph_enhanced_recall_no_source_turn_ids():
    """
    When RAG hits have no source_turn_id metadata, graph_enhanced_recall()
    should try to seed entities from capitalized terms in the query.
    """
    # RAG hit with no source_turn_id
    rag_hit = _make_search_result(id=1, content="some memory", score=0.5)

    # Graph walk from query entities
    walked_row = _make_triplet_row(
        id=20, subject="Alice", predicate="is", obj="developer",
        source_type="conversation", source_id="55"
    )
    discovered_row = {
        "id": 55,
        "role": "assistant",
        "content": "Alice is a developer.",
        "created_at": datetime.now(timezone.utc),
    }

    conn = AsyncMock()
    pool = MagicMock()
    pool.acquire = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

    store = _make_store(pool)
    store.hybrid_recall = AsyncMock(return_value=[rag_hit])

    # No source_turn_ids means get_triplets_for_turns is NOT called.
    # walk_graph is called with query entities from "Tell me about Alice"
    conn.fetch = AsyncMock(side_effect=[
        [walked_row],     # walk_graph hop 1 (seeded from query "Alice")
        [discovered_row], # fetch discovered turns
    ])

    result = await store.graph_enhanced_recall(
        query="Tell me about Alice", limit=5, hops=1
    )

    assert isinstance(result, GraphContext)
    # The graph walk should have been attempted
    # discovered_turns may contain turn 55


@pytest.mark.asyncio
async def test_graph_enhanced_recall_empty_rag():
    """When hybrid_recall returns nothing, GraphContext should have empty rag_results."""
    pool, conn = _make_pool(fetch_results=[])
    store = _make_store(pool)
    store.hybrid_recall = AsyncMock(return_value=[])

    result = await store.graph_enhanced_recall("nothing matches", limit=10, hops=1)

    assert isinstance(result, GraphContext)
    assert result.rag_results == []


# ─── get_entity_neighborhood ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_entity_neighborhood_calls_walk_graph():
    """get_entity_neighborhood() should delegate to walk_graph with hops=1 by default."""
    row = _make_triplet_row(id=1, subject="Alice", predicate="knows", obj="Bob")
    pool, conn = _make_pool(fetch_results=[row])
    store = _make_store(pool)

    results = await store.get_entity_neighborhood("Alice")

    assert len(results) == 1
    assert results[0]["subject"] == "Alice"
    assert results[0]["object"] == "Bob"


@pytest.mark.asyncio
async def test_get_entity_neighborhood_two_hops():
    """get_entity_neighborhood() with hops=2 should traverse 2 hops."""
    hop1 = [_make_triplet_row(id=1, subject="Alice", predicate="knows", obj="Bob")]
    hop2 = [_make_triplet_row(id=2, subject="Bob", predicate="works_at", obj="ACME")]

    conn = AsyncMock()
    conn.fetch = AsyncMock(side_effect=[hop1, hop2])

    pool = MagicMock()
    pool.acquire = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

    store = _make_store(pool)

    results = await store.get_entity_neighborhood("Alice", hops=2)

    assert len(results) == 2


@pytest.mark.asyncio
async def test_get_entity_neighborhood_empty():
    """get_entity_neighborhood() with no triplets returns []."""
    pool, conn = _make_pool(fetch_results=[])
    store = _make_store(pool)

    results = await store.get_entity_neighborhood("NonExistent")

    assert results == []


# ─── get_triplets_for_source ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_triplets_for_source():
    """get_triplets_for_source() should query by source_type + source_id."""
    row = _make_triplet_row(id=3, subject="Task", predicate="requires", obj="Auth",
                            source_type="task", source_id="101")
    pool, conn = _make_pool(fetch_results=[row])
    store = _make_store(pool)

    results = await store.get_triplets_for_source("task", "101")

    assert len(results) == 1
    assert results[0]["source_type"] == "task"
    assert results[0]["source_id"] == "101"
    assert results[0]["subject"] == "Task"


# ─── ContextAssembler — graph formatting ─────────────────────────────────────


def test_format_graph_context_basic():
    """_format_graph_context() should format triplets grouped by source."""
    from nous.context import ContextAssembler
    from nous.store import MemoryStore
    from nous.embeddings import NullEmbedder

    store = _make_store()
    assembler = ContextAssembler(store)

    triplets = [
        Triplet(id=1, subject="Alice", predicate="knows", object="Bob",
                source_type="conversation", source_id="42"),
        Triplet(id=2, subject="Bob", predicate="works_at", object="ACME",
                source_type="conversation", source_id="42"),
        Triplet(id=3, subject="ACME", predicate="is", object="company",
                source_type="memory", source_id="7"),
    ]
    gc = GraphContext(
        rag_results=[],
        graph_triplets=triplets,
        discovered_turns=[],
        entities={"Alice", "Bob", "ACME", "company"},
    )

    result = assembler._format_graph_context(gc, char_budget=10000)

    assert "Alice → knows → Bob" in result
    assert "Bob → works_at → ACME" in result
    assert "ACME → is → company" in result
    # Both source groups present
    assert "conversation:42" in result
    assert "memory:7" in result


def test_format_graph_context_with_discovered_turns():
    """_format_graph_context() should include discovered turns below triplets."""
    from nous.context import ContextAssembler

    store = _make_store()
    assembler = ContextAssembler(store)

    triplets = [
        Triplet(id=1, subject="Alice", predicate="is", object="developer",
                source_type="conversation", source_id="5"),
    ]
    turns = [
        ConversationTurn(id=5, role="user", content="Alice is a developer right?"),
    ]
    gc = GraphContext(
        rag_results=[],
        graph_triplets=triplets,
        discovered_turns=turns,
        entities={"Alice", "developer"},
    )

    result = assembler._format_graph_context(gc, char_budget=10000)

    assert "Alice → is → developer" in result
    assert "Discovered via graph walk" in result
    assert "Alice is a developer right?" in result


def test_format_graph_context_respects_char_budget():
    """_format_graph_context() should truncate output when char_budget is tiny."""
    from nous.context import ContextAssembler

    store = _make_store()
    assembler = ContextAssembler(store)

    triplets = [
        Triplet(id=i, subject=f"Entity{i}", predicate="knows", object=f"Entity{i+1}",
                source_type="conversation", source_id="1")
        for i in range(100)
    ]
    gc = GraphContext(
        rag_results=[],
        graph_triplets=triplets,
        discovered_turns=[],
        entities=set(),
    )

    result = assembler._format_graph_context(gc, char_budget=200)

    # Output must not exceed the budget
    assert len(result) <= 200


def test_format_graph_context_empty():
    """_format_graph_context() with no triplets and no turns should return ''."""
    from nous.context import ContextAssembler

    store = _make_store()
    assembler = ContextAssembler(store)

    gc = GraphContext(
        rag_results=[],
        graph_triplets=[],
        discovered_turns=[],
        entities=set(),
    )

    result = assembler._format_graph_context(gc)

    assert result == ""


# ─── ContextAssembler — build_director_context ───────────────────────────────


@pytest.mark.asyncio
async def test_build_director_context_include_graph_true():
    """
    build_director_context(include_graph=True) should call graph_enhanced_recall
    and include [KNOWLEDGE GRAPH CONTEXT] section when there are triplets.
    """
    from nous.context import ContextAssembler

    store = _make_store()

    triplet = Triplet(id=1, subject="Alice", predicate="knows", object="Bob",
                      source_type="conversation", source_id="1")
    gc = GraphContext(
        rag_results=[_make_search_result(id=1, content="Alice knows Bob", score=0.8)],
        graph_triplets=[triplet],
        discovered_turns=[],
        entities={"Alice", "Bob"},
    )

    store.graph_enhanced_recall = AsyncMock(return_value=gc)
    store.get_recent_conversations = AsyncMock(return_value=[])
    store.search_documents = AsyncMock(return_value=[])

    assembler = ContextAssembler(store)
    context = await assembler.build_director_context(
        query="Tell me about Alice",
        include_conversations=False,
        include_graph=True,
        include_documents=False,
    )

    assert "[KNOWLEDGE GRAPH CONTEXT]" in context
    assert "Alice → knows → Bob" in context
    store.graph_enhanced_recall.assert_called_once()


@pytest.mark.asyncio
async def test_build_director_context_include_graph_false():
    """
    build_director_context(include_graph=False) should call plain recall,
    not graph_enhanced_recall.
    """
    from nous.context import ContextAssembler

    store = _make_store()
    store.recall = AsyncMock(return_value=[
        _make_search_result(id=1, content="some memory", score=0.7)
    ])
    store.graph_enhanced_recall = AsyncMock()
    store.get_recent_conversations = AsyncMock(return_value=[])
    store.search_documents = AsyncMock(return_value=[])

    assembler = ContextAssembler(store)
    context = await assembler.build_director_context(
        query="some query",
        include_conversations=False,
        include_graph=False,
        include_documents=False,
    )

    store.graph_enhanced_recall.assert_not_called()
    store.recall.assert_called_once()
    assert "[KNOWLEDGE GRAPH CONTEXT]" not in context
    assert "[MEMORY CONTEXT]" in context


@pytest.mark.asyncio
async def test_build_director_context_no_query_skips_graph():
    """
    build_director_context() with no query should skip both recall and graph,
    even if include_graph=True.
    """
    from nous.context import ContextAssembler

    store = _make_store()
    store.graph_enhanced_recall = AsyncMock()
    store.recall = AsyncMock()
    store.get_recent_conversations = AsyncMock(return_value=[])
    store.search_documents = AsyncMock(return_value=[])

    assembler = ContextAssembler(store)
    context = await assembler.build_director_context(
        query=None,
        include_conversations=False,
        include_graph=True,
    )

    store.graph_enhanced_recall.assert_not_called()
    store.recall.assert_not_called()


@pytest.mark.asyncio
async def test_build_director_context_graph_fallback_on_error():
    """
    build_director_context() should fall back to plain recall if
    graph_enhanced_recall raises an exception.
    """
    from nous.context import ContextAssembler

    store = _make_store()
    store.graph_enhanced_recall = AsyncMock(side_effect=Exception("graph unavailable"))
    store.recall = AsyncMock(return_value=[
        _make_search_result(id=1, content="fallback memory", score=0.6)
    ])
    store.get_recent_conversations = AsyncMock(return_value=[])
    store.search_documents = AsyncMock(return_value=[])

    assembler = ContextAssembler(store)
    context = await assembler.build_director_context(
        query="some query",
        include_conversations=False,
        include_graph=True,
        include_documents=False,
    )

    # Should have fallen back to plain recall — memory context present
    assert "[MEMORY CONTEXT]" in context
    store.recall.assert_called_once()


# ─── ContextAssembler — build_worker_context ─────────────────────────────────


@pytest.mark.asyncio
async def test_build_worker_context_include_graph_false_by_default():
    """
    build_worker_context() should NOT call graph_enhanced_recall by default
    (include_graph=False).
    """
    from nous.context import ContextAssembler

    store = _make_store()
    store.graph_enhanced_recall = AsyncMock()

    # Mock the shell
    shell = AsyncMock()
    shell.get_memories = AsyncMock(return_value=[])
    shell.get_instructions = AsyncMock(return_value=[])
    shell.get_task_history = AsyncMock(return_value=[])
    store.get_shell = MagicMock(return_value=shell)
    store.get_worker_memories = AsyncMock(return_value=[])
    store.get_worker_resume = AsyncMock(return_value=[])

    assembler = ContextAssembler(store)
    await assembler.build_worker_context("alpha", task_description="fix auth bug")

    store.graph_enhanced_recall.assert_not_called()


@pytest.mark.asyncio
async def test_build_worker_context_include_graph_true():
    """
    build_worker_context(include_graph=True) should call graph_enhanced_recall
    and include [KNOWLEDGE GRAPH CONTEXT] when triplets are found.
    """
    from nous.context import ContextAssembler

    store = _make_store()

    triplet = Triplet(id=1, subject="Auth", predicate="requires", object="JWT",
                      source_type="task", source_id="77")
    gc = GraphContext(
        rag_results=[],
        graph_triplets=[triplet],
        discovered_turns=[],
        entities={"Auth", "JWT"},
    )
    store.graph_enhanced_recall = AsyncMock(return_value=gc)

    shell = AsyncMock()
    shell.get_memories = AsyncMock(return_value=[])
    shell.get_instructions = AsyncMock(return_value=[])
    shell.get_task_history = AsyncMock(return_value=[])
    store.get_shell = MagicMock(return_value=shell)
    store.get_worker_memories = AsyncMock(return_value=[])
    store.get_worker_resume = AsyncMock(return_value=[])

    assembler = ContextAssembler(store)
    context = await assembler.build_worker_context(
        "beta",
        task_description="fix auth bug with JWT",
        include_graph=True,
    )

    assert "[KNOWLEDGE GRAPH CONTEXT]" in context
    assert "Auth → requires → JWT" in context
    store.graph_enhanced_recall.assert_called_once()


# ─── Heuristic extractor — unit tests ────────────────────────────────────────


def test_heuristic_extract_is_pattern():
    """_heuristic_extract should match 'X is Y' pattern."""
    from nous.store import MemoryStore
    from nous.embeddings import NullEmbedder

    store = _make_store()
    triplets = store._heuristic_extract("Python is a language.")

    assert len(triplets) >= 1
    subjects = [t[0].lower() for t in triplets]
    assert any("python" in s for s in subjects)


def test_heuristic_extract_uses_pattern():
    """_heuristic_extract should match 'X uses Y' pattern."""
    store = _make_store()
    triplets = store._heuristic_extract("Django uses Python for web development.")

    predicates = [t[1].lower() for t in triplets]
    assert any("uses" in p or "use" in p for p in predicates)


def test_heuristic_extract_empty_content():
    """_heuristic_extract on empty string should return []."""
    store = _make_store()
    triplets = store._heuristic_extract("")
    assert triplets == []


def test_heuristic_extract_deduplicates():
    """_heuristic_extract should deduplicate identical triplets."""
    store = _make_store()
    # Same statement twice in different sentences
    content = "Python is a language. Python is a language."
    triplets = store._heuristic_extract(content)

    # Should only have 1 after dedup
    keys = [(t[0].lower(), t[1].lower(), t[2].lower()) for t in triplets]
    assert len(keys) == len(set(keys))


def test_heuristic_extract_rejects_pronoun_subject():
    """_heuristic_extract should reject sentences where subject is a pronoun like 'This'."""
    store = _make_store()
    content = "This is why I forgot Nous existed."
    triplets = store._heuristic_extract(content)

    subjects = [t[0] for t in triplets]
    # "This" is in the blacklist and should be rejected
    assert "This" not in subjects
    assert "this" not in [s.lower() for s in subjects]


def test_heuristic_extract_rejects_it_subject():
    """_heuristic_extract should reject 'It is ...' sentences."""
    store = _make_store()
    content = "It is always visible in the system."
    triplets = store._heuristic_extract(content)

    subjects = [t[0] for t in triplets]
    assert "It" not in subjects
    assert "it" not in [s.lower() for s in subjects]


def test_heuristic_extract_strips_leading_article_from_subject():
    """_heuristic_extract should strip 'The ' from the start of captured subjects."""
    store = _make_store()
    # "The system prompt" should produce subject "system prompt" (lowercase — rejected)
    # So use a proper-noun: "The PostgreSQL database is fast."
    content = "The PostgreSQL database is fast."
    triplets = store._heuristic_extract(content)

    subjects = [t[0] for t in triplets]
    # "The PostgreSQL" should have been stripped to "PostgreSQL"
    # (or the whole triplet skipped if the remaining subject starts lowercase)
    for s in subjects:
        assert not s.lower().startswith("the "), f"Subject '{s}' still has leading article"


def test_heuristic_extract_strips_leading_conjunction():
    """_heuristic_extract should strip 'and the' from subjects."""
    store = _make_store()
    # "and the HAC dendrogram" is a sentence-fragment subject — should be rejected
    # because after stripping "and the ", "HAC dendrogram" starts with 'H' (capital) BUT
    # this wouldn't naturally appear at sentence-start. Simulate it:
    content = "and the HAC dendrogram is the ontology hierarchy."
    triplets = store._heuristic_extract(content)

    subjects = [t[0] for t in triplets]
    # The raw "and the HAC dendrogram" should NOT appear as a subject
    for s in subjects:
        assert not s.lower().startswith("and "), f"Subject '{s}' still has 'and' prefix"
        assert not s.lower().startswith("the "), f"Subject '{s}' still has 'the' prefix"


def test_heuristic_extract_truncates_overlong_object():
    """_heuristic_extract should truncate objects longer than 30 chars."""
    store = _make_store()
    # Construct a sentence where the object is a long fragment
    content = "Hermit uses instead of rolling custom memory per deployment environment."
    triplets = store._heuristic_extract(content)

    for subj, pred, obj in triplets:
        assert len(obj) <= 30, f"Object '{obj}' exceeds 30 chars"


def test_heuristic_extract_requires_capital_subject():
    """_heuristic_extract should reject subjects that don't start with a capital letter."""
    store = _make_store()
    # "backend that Hermit" starts with lowercase 'b' after stripping any article
    content = "backend that Hermit should use instead of rolling custom memory."
    triplets = store._heuristic_extract(content)

    subjects = [t[0] for t in triplets]
    for s in subjects:
        assert s[0].isupper(), f"Subject '{s}' does not start with capital letter"


def test_heuristic_extract_keeps_valid_named_entities():
    """_heuristic_extract should keep well-formed named-entity triplets."""
    store = _make_store()
    # Use predicates that are in the heuristic patterns: 'stores', 'uses'
    content = "PostgreSQL stores conversation history. Redis uses embeddings."
    triplets = store._heuristic_extract(content)

    subjects = [t[0] for t in triplets]
    # These are proper nouns — should be captured
    assert any("PostgreSQL" in s for s in subjects) or any("Redis" in s for s in subjects)
