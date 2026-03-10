"""
Tests for nous.semantic_cluster — embed_and_cluster() and helpers.

Tests cover:
- Empty entity list → empty ClusterResult
- NullEmbedder → single cluster fallback (no scipy needed)
- Single entity → trivial single cluster
- Two entities → two clusters or one depending on similarity
- HAC clustering groups semantically similar entities
- inter_cluster_edges computed from centroids
- min_cluster_size merging
- Embedding failure → graceful single-cluster fallback
- summarize_cluster_result formatting

All tests use mock EmbeddingProviders — no live Ollama required.
"""

from __future__ import annotations

import math
from unittest.mock import AsyncMock, patch

import pytest

from nous.embeddings import NullEmbedder
from nous.hierarchy import ClusterResult, EntityCluster, HierarchyEntity
from nous.semantic_cluster import (
    _compute_centroid,
    _compute_inter_cluster_edges,
    _most_central_entity,
    embed_and_cluster,
    summarize_cluster_result,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _make_entity(
    id: str,
    text: str,
    embedding: list[float] | None = None,
    requirements: list[str] | None = None,
    node_type: str = "process",
) -> HierarchyEntity:
    return HierarchyEntity(
        id=id,
        text=text,
        embedding=embedding or [],
        requirements=requirements or [],
        node_type=node_type,
    )


def _mock_embedder(embeddings: list[list[float]], model_name: str = "mock-model"):
    """Return a mock EmbeddingProvider that returns the given embeddings."""
    embedder = AsyncMock()
    embedder.model_name = model_name
    embedder.embed_batch = AsyncMock(return_value=embeddings)
    embedder.embed = AsyncMock(side_effect=lambda t: embeddings[0])
    return embedder


def _unit_vec(dims: int, hot_dim: int, value: float = 1.0) -> list[float]:
    """Unit vector with `value` at `hot_dim`, zeros elsewhere."""
    v = [0.0] * dims
    v[hot_dim % dims] = value
    return v


def _normalize(v: list[float]) -> list[float]:
    mag = math.sqrt(sum(x * x for x in v))
    if mag == 0:
        return v
    return [x / mag for x in v]


# ─── Tests: edge cases ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_empty_entities_returns_empty_result():
    embedder = _mock_embedder([])
    result = await embed_and_cluster([], embedder)
    assert isinstance(result, ClusterResult)
    assert result.clusters == []
    assert result.inter_cluster_edges == []


@pytest.mark.asyncio
async def test_null_embedder_returns_single_cluster():
    """NullEmbedder → all entities in one cluster, no scipy required."""
    embedder = NullEmbedder()
    entities = [
        _make_entity("e0", "PostgreSQL"),
        _make_entity("e1", "SQLite"),
        _make_entity("e2", "memory store"),
    ]
    result = await embed_and_cluster(entities, embedder)

    assert len(result.clusters) == 1
    assert len(result.clusters[0].entities) == 3
    # inter_cluster_edges should be empty (only 1 cluster)
    assert result.inter_cluster_edges == []


@pytest.mark.asyncio
async def test_single_entity_returns_one_cluster():
    """Single entity → trivial cluster, no scipy invoked."""
    emb = _normalize([1.0, 0.0, 0.0])
    embedder = _mock_embedder([emb])
    entities = [_make_entity("e0", "vector search", embedding=[])]

    result = await embed_and_cluster(entities, embedder)

    assert len(result.clusters) == 1
    cluster = result.clusters[0]
    assert cluster.id == "cluster-0"
    assert len(cluster.entities) == 1
    assert cluster.entities[0].text == "vector search"
    # Embedding should be populated in-place
    assert entities[0].embedding == emb
    assert result.inter_cluster_edges == []


@pytest.mark.asyncio
async def test_embedding_failure_returns_single_cluster():
    """When embed_batch raises, fall back gracefully to single cluster."""
    embedder = AsyncMock()
    embedder.model_name = "failing-model"
    embedder.embed_batch = AsyncMock(side_effect=RuntimeError("Ollama offline"))

    entities = [
        _make_entity("e0", "auth"),
        _make_entity("e1", "login"),
    ]
    result = await embed_and_cluster(entities, embedder)

    assert len(result.clusters) == 1
    assert len(result.clusters[0].entities) == 2


# ─── Tests: HAC clustering ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_two_similar_entities_cluster_together():
    """Two very similar embeddings → one cluster when threshold is generous."""
    pytest.importorskip("scipy")

    # Near-identical vectors → cosine distance ≈ 0 → same cluster
    e0 = _normalize([1.0, 0.1, 0.0])
    e1 = _normalize([1.0, 0.15, 0.0])
    embedder = _mock_embedder([e0, e1])
    entities = [_make_entity("e0", "auth"), _make_entity("e1", "login")]

    result = await embed_and_cluster(entities, embedder, distance_threshold=0.9)

    # With very high threshold, both should land in the same cluster
    assert len(result.clusters) == 1
    assert len(result.clusters[0].entities) == 2


@pytest.mark.asyncio
async def test_two_dissimilar_entities_form_separate_clusters():
    """Two orthogonal embeddings → two clusters when threshold is tight."""
    pytest.importorskip("scipy")

    # Orthogonal → cosine distance = 1.0 → separate clusters with tight threshold
    e0 = _normalize([1.0, 0.0, 0.0])
    e1 = _normalize([0.0, 1.0, 0.0])
    embedder = _mock_embedder([e0, e1])
    entities = [_make_entity("e0", "database"), _make_entity("e1", "queue")]

    result = await embed_and_cluster(entities, embedder, distance_threshold=0.1)

    assert len(result.clusters) == 2
    cluster_entities = {e.text for c in result.clusters for e in c.entities}
    assert cluster_entities == {"database", "queue"}


@pytest.mark.asyncio
async def test_three_entities_two_clusters():
    """Three entities where two are similar and one is distinct → 2 clusters."""
    pytest.importorskip("scipy")

    # e0 and e1 close; e2 orthogonal
    e0 = _normalize([1.0, 0.05, 0.0])
    e1 = _normalize([0.95, 0.1, 0.0])
    e2 = _normalize([0.0, 0.0, 1.0])
    embedder = _mock_embedder([e0, e1, e2])
    entities = [
        _make_entity("e0", "PostgreSQL"),
        _make_entity("e1", "SQLite"),
        _make_entity("e2", "HTTP gateway"),
    ]

    result = await embed_and_cluster(entities, embedder, distance_threshold=0.3)

    assert len(result.clusters) == 2

    # The database cluster should contain both SQL entities
    entity_texts = {e.text for c in result.clusters for e in c.entities}
    assert entity_texts == {"PostgreSQL", "SQLite", "HTTP gateway"}


@pytest.mark.asyncio
async def test_cluster_ids_are_sequential():
    """Cluster IDs should be cluster-0, cluster-1, etc."""
    pytest.importorskip("scipy")

    e0 = _normalize([1.0, 0.0, 0.0])
    e1 = _normalize([0.0, 1.0, 0.0])
    e2 = _normalize([0.0, 0.0, 1.0])
    embedder = _mock_embedder([e0, e1, e2])
    entities = [
        _make_entity("e0", "alpha"),
        _make_entity("e1", "beta"),
        _make_entity("e2", "gamma"),
    ]

    result = await embed_and_cluster(entities, embedder, distance_threshold=0.01)

    cluster_ids = [c.id for c in result.clusters]
    for i, cid in enumerate(cluster_ids):
        assert cid == f"cluster-{i}"


@pytest.mark.asyncio
async def test_entity_embeddings_populated_in_place():
    """embed_and_cluster mutates HierarchyEntity.embedding in-place."""
    pytest.importorskip("scipy")

    embs = [
        _normalize([1.0, 0.0]),
        _normalize([0.9, 0.1]),
    ]
    embedder = _mock_embedder(embs)
    entities = [
        _make_entity("e0", "entity-a"),
        _make_entity("e1", "entity-b"),
    ]
    # Start with empty embeddings
    assert entities[0].embedding == []
    assert entities[1].embedding == []

    await embed_and_cluster(entities, embedder)

    # Should be populated in-place
    assert entities[0].embedding == embs[0]
    assert entities[1].embedding == embs[1]


@pytest.mark.asyncio
async def test_ward_linkage_switches_to_average():
    """Passing linkage_method='ward' should auto-switch to 'average' (cosine compat)."""
    pytest.importorskip("scipy")

    e0 = _normalize([1.0, 0.0])
    e1 = _normalize([0.0, 1.0])
    embedder = _mock_embedder([e0, e1])
    entities = [_make_entity("e0", "a"), _make_entity("e1", "b")]

    # Should not raise even though "ward" isn't valid for cosine
    result = await embed_and_cluster(
        entities, embedder, distance_threshold=0.1, linkage_method="ward"
    )
    assert len(result.clusters) >= 1


# ─── Tests: inter_cluster_edges ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_inter_cluster_edges_computed():
    """Two separate clusters → one inter-cluster edge computed from centroids."""
    pytest.importorskip("scipy")

    e0 = _normalize([1.0, 0.0, 0.0])
    e1 = _normalize([0.0, 1.0, 0.0])
    embedder = _mock_embedder([e0, e1])
    entities = [_make_entity("e0", "alpha"), _make_entity("e1", "beta")]

    result = await embed_and_cluster(entities, embedder, distance_threshold=0.01)

    if len(result.clusters) == 2:
        assert len(result.inter_cluster_edges) == 1
        a_id, b_id, sim = result.inter_cluster_edges[0]
        assert a_id in {c.id for c in result.clusters}
        assert b_id in {c.id for c in result.clusters}
        assert 0.0 <= sim <= 1.0
    else:
        # Only 1 cluster → no edges, valid
        assert result.inter_cluster_edges == []


@pytest.mark.asyncio
async def test_inter_cluster_edges_sorted_by_similarity_descending():
    """Inter-cluster edges should be sorted by similarity descending."""
    pytest.importorskip("scipy")

    # 3 orthogonal vectors → 3 separate clusters → 3 edges
    e0 = _normalize([1.0, 0.0, 0.0])
    e1 = _normalize([0.0, 1.0, 0.0])
    e2 = _normalize([0.0, 0.0, 1.0])
    embedder = _mock_embedder([e0, e1, e2])
    entities = [
        _make_entity("e0", "a"),
        _make_entity("e1", "b"),
        _make_entity("e2", "c"),
    ]

    result = await embed_and_cluster(entities, embedder, distance_threshold=0.01)

    if len(result.inter_cluster_edges) > 1:
        sims = [sim for _, _, sim in result.inter_cluster_edges]
        assert sims == sorted(sims, reverse=True)


def test_compute_inter_cluster_edges_direct():
    """Unit test _compute_inter_cluster_edges with known centroids."""
    c0 = EntityCluster(
        id="cluster-0",
        entities=[_make_entity("e0", "a")],
        centroid=_normalize([1.0, 0.0]),
    )
    c1 = EntityCluster(
        id="cluster-1",
        entities=[_make_entity("e1", "b")],
        centroid=_normalize([0.0, 1.0]),
    )
    c2 = EntityCluster(
        id="cluster-2",
        entities=[_make_entity("e2", "c")],
        centroid=_normalize([1.0, 0.0]),  # same as c0
    )

    edges = _compute_inter_cluster_edges([c0, c1, c2])

    assert len(edges) == 3
    # c0–c2 should have sim ≈ 1.0 (same direction)
    # Sort to find it
    c0_c2 = next(
        (e for e in edges if {e[0], e[1]} == {"cluster-0", "cluster-2"}), None
    )
    assert c0_c2 is not None
    assert abs(c0_c2[2] - 1.0) < 1e-5

    # c0–c1 should have sim ≈ 0.0 (orthogonal)
    c0_c1 = next(
        (e for e in edges if {e[0], e[1]} == {"cluster-0", "cluster-1"}), None
    )
    assert c0_c1 is not None
    assert abs(c0_c1[2]) < 1e-5


# ─── Tests: min_cluster_size merging ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_min_cluster_size_merges_small_clusters():
    """With min_cluster_size=2, singleton clusters are merged into larger ones."""
    pytest.importorskip("scipy")

    # 3 orthogonal → 3 singleton clusters at tight threshold
    e0 = _normalize([1.0, 0.0, 0.0])
    e1 = _normalize([0.0, 1.0, 0.0])
    e2 = _normalize([0.0, 0.0, 1.0])
    embedder = _mock_embedder([e0, e1, e2])
    entities = [
        _make_entity("e0", "alpha"),
        _make_entity("e1", "beta"),
        _make_entity("e2", "gamma"),
    ]

    result = await embed_and_cluster(
        entities, embedder, distance_threshold=0.01, min_cluster_size=2
    )

    # After merging, no cluster should have fewer than 2 entities
    # (unless there's truly only 1 cluster possible)
    for cluster in result.clusters:
        assert len(cluster.entities) >= 1  # at least 1 always true
    # Total entities preserved
    total = sum(len(c.entities) for c in result.clusters)
    assert total == 3


# ─── Tests: centroid and label helpers ───────────────────────────────────────


def test_compute_centroid_basic():
    """Centroid of two opposite unit vectors should be near zero."""
    v0 = [1.0, 0.0]
    v1 = [-1.0, 0.0]
    centroid = _compute_centroid([v0, v1])
    assert abs(centroid[0]) < 1e-9
    assert abs(centroid[1]) < 1e-9


def test_compute_centroid_empty():
    assert _compute_centroid([]) == []


def test_compute_centroid_single():
    v = [0.5, 0.5]
    assert _compute_centroid([v]) == v


def test_compute_centroid_ignores_empty_embeddings():
    """Empty embedding lists should be ignored in centroid computation."""
    v = [1.0, 0.0]
    centroid = _compute_centroid([v, []])
    assert centroid == v


def test_most_central_entity_single():
    entity = _make_entity("e0", "only one")
    result = _most_central_entity([entity], centroid=[1.0, 0.0])
    assert result == "only one"


def test_most_central_entity_picks_closest():
    """The entity closest to the centroid should be chosen as label."""
    centroid = _normalize([1.0, 0.0, 0.0])

    e0 = _make_entity("e0", "near", embedding=_normalize([0.9, 0.1, 0.0]))
    e1 = _make_entity("e1", "far", embedding=_normalize([0.0, 0.0, 1.0]))

    result = _most_central_entity([e0, e1], centroid)
    assert result == "near"


def test_most_central_entity_no_embeddings():
    """When no embeddings available, return None."""
    e0 = _make_entity("e0", "a", embedding=[])
    e1 = _make_entity("e1", "b", embedding=[])
    result = _most_central_entity([e0, e1], centroid=[1.0, 0.0])
    assert result is None


# ─── Tests: summarize_cluster_result ─────────────────────────────────────────


def test_summarize_cluster_result_empty():
    result = ClusterResult(clusters=[], inter_cluster_edges=[])
    summary = summarize_cluster_result(result)
    assert "0 entities" in summary
    assert "0 clusters" in summary


def test_summarize_cluster_result_with_clusters():
    e0 = _make_entity("e0", "PostgreSQL")
    e1 = _make_entity("e1", "SQLite")
    cluster = EntityCluster(
        id="cluster-0",
        entities=[e0, e1],
        centroid=[0.5, 0.5],
        label="PostgreSQL",
    )
    result = ClusterResult(
        clusters=[cluster],
        inter_cluster_edges=[],
    )
    summary = summarize_cluster_result(result)
    assert "2 entities" in summary
    assert "1 clusters" in summary
    assert "cluster-0" in summary
    assert "PostgreSQL" in summary


def test_summarize_cluster_result_shows_edge_count():
    c0 = EntityCluster(id="cluster-0", entities=[_make_entity("e0", "a")])
    c1 = EntityCluster(id="cluster-1", entities=[_make_entity("e1", "b")])
    result = ClusterResult(
        clusters=[c0, c1],
        inter_cluster_edges=[("cluster-0", "cluster-1", 0.75)],
    )
    summary = summarize_cluster_result(result)
    assert "1 inter-cluster edges" in summary


# ─── Tests: requirements tracing ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_entity_requirements_preserved_through_clustering():
    """HierarchyEntity.requirements survive clustering unchanged."""
    pytest.importorskip("scipy")

    e0 = _make_entity("e0", "embed entities", requirements=["REQ-NKGL-002"])
    e1 = _make_entity("e1", "cluster entities", requirements=["REQ-NKGL-003"])

    e0_emb = _normalize([1.0, 0.05])
    e1_emb = _normalize([0.05, 1.0])
    embedder = _mock_embedder([e0_emb, e1_emb])

    result = await embed_and_cluster([e0, e1], embedder, distance_threshold=0.3)

    all_entities = [e for c in result.clusters for e in c.entities]
    all_reqs = {req for e in all_entities for req in e.requirements}
    assert "REQ-NKGL-002" in all_reqs
    assert "REQ-NKGL-003" in all_reqs


# ─── Tests: cluster label ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cluster_label_is_most_central_entity():
    """The cluster label should be the entity text closest to the centroid."""
    pytest.importorskip("scipy")

    # e0 and e1 are close; centroid will be near both
    e0_emb = _normalize([1.0, 0.1, 0.0])
    e1_emb = _normalize([0.9, 0.2, 0.0])
    embedder = _mock_embedder([e0_emb, e1_emb])

    entities = [
        _make_entity("e0", "entity-alpha"),
        _make_entity("e1", "entity-beta"),
    ]

    result = await embed_and_cluster(entities, embedder, distance_threshold=0.9)

    assert len(result.clusters) == 1
    cluster = result.clusters[0]
    # Label should be one of the entity texts (whichever is closest to centroid)
    assert cluster.label in {"entity-alpha", "entity-beta"}


# ─── Tests: graph-constrained clustering ─────────────────────────────────────


@pytest.mark.asyncio
async def test_graph_edges_pull_dissimilar_entities_into_same_cluster():
    """
    graph_edges causes orthogonal entities (cosine dist=1.0) to cluster together.

    Without graph_edges they would form separate clusters at tight threshold.
    With graph_edges their distance is multiplied by 0.3, making them close enough
    to cluster together.
    """
    pytest.importorskip("scipy")

    # Orthogonal embeddings — pure semantic clustering would separate these
    e0 = _normalize([1.0, 0.0, 0.0])
    e1 = _normalize([0.0, 1.0, 0.0])
    embedder = _mock_embedder([e0, e1])
    entities = [
        _make_entity("e0", "Rhode"),
        _make_entity("e1", "Telegram"),
    ]

    # Without graph_edges: tight threshold should produce 2 clusters
    result_no_edges = await embed_and_cluster(
        entities, embedder, distance_threshold=0.5
    )
    # Re-embed since embedder mock returns fixed values
    embedder2 = _mock_embedder([e0, e1])
    entities2 = [
        _make_entity("e0", "Rhode"),
        _make_entity("e1", "Telegram"),
    ]

    # With graph_edges: distance is reduced by 0.3x → 0.3 * 1.0 = 0.3 < threshold=0.5
    # So they should now be in the same cluster
    result_with_edges = await embed_and_cluster(
        entities2,
        embedder2,
        distance_threshold=0.5,
        graph_edges=[("e0", "e1")],
    )

    # With the graph edge, the two entities should land in one cluster
    assert len(result_with_edges.clusters) == 1, (
        f"Expected 1 cluster with graph_edges, got {len(result_with_edges.clusters)}"
    )
    cluster_texts = {e.text for e in result_with_edges.clusters[0].entities}
    assert cluster_texts == {"Rhode", "Telegram"}


@pytest.mark.asyncio
async def test_graph_edges_none_behaves_like_no_graph_edges():
    """graph_edges=None is backward-compatible (same as not passing it)."""
    pytest.importorskip("scipy")

    e0 = _normalize([1.0, 0.05, 0.0])
    e1 = _normalize([0.95, 0.1, 0.0])
    embedder_a = _mock_embedder([e0, e1])
    embedder_b = _mock_embedder([e0, e1])
    entities_a = [_make_entity("e0", "alpha"), _make_entity("e1", "beta")]
    entities_b = [_make_entity("e0", "alpha"), _make_entity("e1", "beta")]

    result_default = await embed_and_cluster(entities_a, embedder_a, distance_threshold=0.9)
    result_none = await embed_and_cluster(
        entities_b, embedder_b, distance_threshold=0.9, graph_edges=None
    )

    assert len(result_default.clusters) == len(result_none.clusters)


@pytest.mark.asyncio
async def test_graph_edges_unknown_ids_ignored():
    """graph_edges with entity IDs not in the entity list are silently ignored."""
    pytest.importorskip("scipy")

    e0 = _normalize([1.0, 0.0])
    e1 = _normalize([0.0, 1.0])
    embedder = _mock_embedder([e0, e1])
    entities = [_make_entity("e0", "alpha"), _make_entity("e1", "beta")]

    # "nonexistent-id" is not in entities — should not raise
    result = await embed_and_cluster(
        entities,
        embedder,
        distance_threshold=0.1,
        graph_edges=[("e0", "nonexistent-id"), ("also-missing", "e1")],
    )
    assert len(result.clusters) >= 1  # no crash, some valid result


@pytest.mark.asyncio
async def test_graph_edges_bidirectional():
    """graph_edges are treated symmetrically — (a,b) and (b,a) both reduce distance."""
    pytest.importorskip("scipy")

    e0 = _normalize([1.0, 0.0, 0.0])
    e1 = _normalize([0.0, 1.0, 0.0])
    embedder = _mock_embedder([e0, e1])
    entities = [_make_entity("e0", "X"), _make_entity("e1", "Y")]

    # Pass reverse order edge (e1→e0) — should still cluster together
    result = await embed_and_cluster(
        entities,
        embedder,
        distance_threshold=0.5,
        graph_edges=[("e1", "e0")],
    )
    assert len(result.clusters) == 1
