"""
Semantic clustering for the Nous hierarchy generator.

Takes a list of HierarchyEntity objects (entity text + optional requirements),
embeds them via any EmbeddingProvider, then clusters via Hierarchical
Agglomerative Clustering (HAC) using scipy's average linkage on cosine distance.
Produces a ClusterResult (from nous.hierarchy) consumed by BridgeDetector and
OlogicEmitter.

Implements:
    REQ-NKGL-002 — Embed entities via configurable EmbeddingProvider (OllamaEmbedder)
    REQ-NKGL-003 — Cluster entities via HAC (average linkage on cosine distance)

Usage:
    from nous import OllamaEmbedder
    from nous.hierarchy import HierarchyEntity
    from nous.semantic_cluster import embed_and_cluster

    embedder = OllamaEmbedder(model="nomic-embed-text")
    entities = [
        HierarchyEntity(id="e0", text="PostgreSQL connection pool", requirements=["REQ-NKGL-002"]),
        HierarchyEntity(id="e1", text="SQLite worker shell", requirements=["REQ-NKGL-002"]),
        HierarchyEntity(id="e2", text="vector cosine similarity", requirements=["REQ-NKGL-003"]),
    ]
    result = await embed_and_cluster(entities, embedder, distance_threshold=0.6)

    for cluster in result.clusters:
        print(f"Cluster {cluster.id} [{cluster.display_label()}]: "
              f"{[e.text for e in cluster.entities]}")

Optional dependency:
    pip install nous-memory[cluster]
    # which installs scipy>=1.11.0 and numpy>=1.24.0
"""

from __future__ import annotations

import logging

from nous.embeddings import EmbeddingProvider, NullEmbedder
from nous.hierarchy import ClusterResult, EntityCluster, HierarchyEntity
from nous.vectors import cosine_similarity

logger = logging.getLogger(__name__)


# ─── Internal helpers ─────────────────────────────────────────────────────────


def _require_scipy():
    """
    Import scipy.cluster.hierarchy lazily — only required when clustering.

    Raises ImportError with a helpful message if scipy is missing.
    Returns (linkage, fcluster) functions.
    """
    try:
        from scipy.cluster.hierarchy import fcluster, linkage

        return linkage, fcluster
    except ImportError:
        raise ImportError(
            "scipy is required for HAC clustering. "
            "Install with: pip install nous-memory[cluster]"
        )


def _require_numpy():
    """Import numpy lazily. Raises ImportError with a helpful message if missing."""
    try:
        import numpy as np

        return np
    except ImportError:
        raise ImportError(
            "numpy is required for HAC clustering. "
            "Install with: pip install nous-memory[cluster]"
        )


def _cosine_distance_matrix_numpy(embeddings: list[list[float]]):
    """
    Compute pairwise cosine distance matrix using numpy.

    Returns a numpy ndarray of shape (n, n), where entry [i][j] = 1 - cosine_sim(i, j).
    Distance is clamped to [0, 2].
    """
    np = _require_numpy()
    mat = np.array(embeddings, dtype=np.float64)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    mat_normed = mat / norms
    sim = mat_normed @ mat_normed.T
    sim = np.clip(sim, -1.0, 1.0)
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    return dist


def _compute_centroid(embeddings: list[list[float]]) -> list[float]:
    """Compute the mean vector (centroid) of a list of embedding vectors."""
    if not embeddings:
        return []
    real = [e for e in embeddings if e]
    if not real:
        return []
    dims = len(real[0])
    centroid = [0.0] * dims
    for v in real:
        for i, x in enumerate(v):
            centroid[i] += x
    n = len(real)
    return [x / n for x in centroid]


def _most_central_entity(
    entities: list[HierarchyEntity],
    centroid: list[float],
) -> str | None:
    """
    Return the text of the entity whose embedding is closest to the centroid.

    Used as the cluster label. Returns None if no embeddings available.
    """
    if not entities:
        return None
    if len(entities) == 1:
        return entities[0].text

    best_entity = None
    best_sim = -2.0

    for entity in entities:
        if not entity.embedding or not centroid:
            continue
        sim = cosine_similarity(entity.embedding, centroid)
        if sim > best_sim:
            best_sim = sim
            best_entity = entity.text

    return best_entity


def _compute_inter_cluster_edges(
    clusters: list[EntityCluster],
) -> list[tuple[str, str, float]]:
    """
    Compute inter-cluster similarity edges from cluster centroids.

    For each pair of clusters, computes cosine similarity between centroids.
    Returns list of (cluster_id_a, cluster_id_b, similarity) tuples, sorted
    by similarity descending. Only non-empty centroids are used.
    """
    edges: list[tuple[str, str, float]] = []
    n = len(clusters)

    for i in range(n):
        for j in range(i + 1, n):
            c_a = clusters[i]
            c_b = clusters[j]
            if not c_a.centroid or not c_b.centroid:
                continue
            sim = cosine_similarity(c_a.centroid, c_b.centroid)
            # Clamp to [0, 1] — cosine sim can be negative for opposite vectors
            sim = max(0.0, min(1.0, sim))
            edges.append((c_a.id, c_b.id, sim))

    edges.sort(key=lambda e: e[2], reverse=True)
    return edges


def _single_cluster(entities: list[HierarchyEntity]) -> ClusterResult:
    """Return a ClusterResult with all entities in one cluster (no edges)."""
    if not entities:
        return ClusterResult(clusters=[], inter_cluster_edges=[])

    centroid = _compute_centroid([e.embedding for e in entities])
    label = _most_central_entity(entities, centroid) if centroid else None

    cluster = EntityCluster(
        id="cluster-0",
        entities=list(entities),
        centroid=centroid,
        label=label,
    )
    return ClusterResult(clusters=[cluster], inter_cluster_edges=[])


# ─── Public API ───────────────────────────────────────────────────────────────


async def embed_and_cluster(
    entities: list[HierarchyEntity],
    embedder: EmbeddingProvider,
    distance_threshold: float = 0.7,
    linkage_method: str = "average",
    min_cluster_size: int = 1,
) -> ClusterResult:
    """
    Embed a list of HierarchyEntity objects and cluster them via HAC.

    Implements REQ-NKGL-002 (embedding via OllamaEmbedder) and REQ-NKGL-003
    (clustering via Hierarchical Agglomerative Clustering).

    This function mutates entities in-place to populate their `embedding` field.
    Embedding is done via the provided EmbeddingProvider (e.g. OllamaEmbedder).
    Clustering uses scipy HAC with average linkage on cosine distance.

    After clustering, inter-cluster similarity edges are computed from centroids
    and stored in ClusterResult.inter_cluster_edges for BridgeDetector.

    Args:
        entities:           HierarchyEntity objects to embed and cluster.
                            Each entity's `embedding` field is populated in-place.
        embedder:           Any EmbeddingProvider. NullEmbedder skips semantic
                            clustering and returns all entities in one cluster.
        distance_threshold: HAC dendrogram cut height (cosine distance, [0.0, 2.0]).
                            Lower = more clusters. Default 0.7 gives moderate
                            granularity with nomic-embed-text.
        linkage_method:     Scipy linkage: "average" (default, works with cosine
                            distance), "complete", or "single". Avoid "ward" —
                            it requires Euclidean distance.
        min_cluster_size:   Minimum entities per cluster. Smaller clusters are
                            merged into the nearest (by centroid) larger cluster.
                            Default 1 (no merging).

    Returns:
        ClusterResult with:
            clusters:            List of EntityCluster objects (each has id, entities,
                                 centroid, label).
            inter_cluster_edges: List of (cluster_id_a, cluster_id_b, similarity)
                                 tuples for BridgeDetector, sorted by similarity desc.

    Raises:
        ImportError: if scipy/numpy are not installed. Use: pip install nous-memory[cluster]
    """
    if not entities:
        return ClusterResult(clusters=[], inter_cluster_edges=[])

    # ── Step 1: Embed all entities ────────────────────────────────────────────
    logger.debug("Embedding %d entities via %s", len(entities), embedder.model_name)

    if isinstance(embedder, NullEmbedder):
        logger.warning(
            "NullEmbedder in use — semantic clustering unavailable, "
            "returning all entities as one cluster"
        )
        # Leave embeddings as empty — single cluster fallback
        return _single_cluster(entities)

    texts = [e.text for e in entities]
    try:
        raw_embeddings = await embedder.embed_batch(texts)
        for entity, emb in zip(entities, raw_embeddings):
            entity.embedding = emb
    except Exception as exc:
        logger.error(
            "Embedding failed (%s) — falling back to single cluster", exc
        )
        return _single_cluster(entities)

    embeddings = [e.embedding for e in entities]

    # ── Step 2: Degenerate cases ──────────────────────────────────────────────

    if len(entities) == 1:
        entity = entities[0]
        centroid = list(entity.embedding) if entity.embedding else []
        cluster = EntityCluster(
            id="cluster-0",
            entities=[entity],
            centroid=centroid,
            label=entity.text,
        )
        return ClusterResult(clusters=[cluster], inter_cluster_edges=[])

    # ── Step 3: HAC via scipy ─────────────────────────────────────────────────

    linkage_fn, fcluster_fn = _require_scipy()
    np = _require_numpy()

    logger.debug(
        "Running HAC (%s linkage, distance_threshold=%.2f) on %d entities",
        linkage_method,
        distance_threshold,
        len(entities),
    )

    try:
        dist_matrix = _cosine_distance_matrix_numpy(embeddings)
    except Exception as exc:
        logger.error("Distance matrix computation failed: %s — single cluster", exc)
        return _single_cluster(entities)

    # Convert full distance matrix to condensed upper-triangle form for scipy
    n = len(entities)
    condensed = []
    for i in range(n):
        for j in range(i + 1, n):
            condensed.append(float(dist_matrix[i, j]))

    condensed_np = np.array(condensed, dtype=np.float64)

    # If caller requests "ward", switch to "average" — ward requires Euclidean
    actual_method = linkage_method
    if linkage_method == "ward":
        logger.debug(
            "Ward linkage requires Euclidean distance; switching to 'average' "
            "for cosine distance"
        )
        actual_method = "average"

    try:
        Z = linkage_fn(condensed_np, method=actual_method)
    except Exception as exc:
        logger.error("scipy linkage failed: %s — single cluster", exc)
        return _single_cluster(entities)

    # Cut the dendrogram to get flat cluster labels
    raw_labels = fcluster_fn(Z, t=distance_threshold, criterion="distance")
    # scipy labels are 1-indexed; convert to 0-indexed
    labels = [int(lbl) - 1 for lbl in raw_labels]

    # ── Step 4: Build EntityCluster objects ───────────────────────────────────

    # Group entities by cluster label
    cluster_groups: dict[int, list[HierarchyEntity]] = {}
    for entity, label in zip(entities, labels):
        cluster_groups.setdefault(label, []).append(entity)

    clusters: list[EntityCluster] = []

    for new_idx, (_, members) in enumerate(sorted(cluster_groups.items())):
        member_embeddings = [m.embedding for m in members]
        centroid = _compute_centroid(member_embeddings)
        label_str = _most_central_entity(members, centroid)

        cluster = EntityCluster(
            id=f"cluster-{new_idx}",
            entities=members,
            centroid=centroid,
            label=label_str,
        )
        clusters.append(cluster)

    # ── Step 5: Merge tiny clusters if min_cluster_size > 1 ──────────────────

    if min_cluster_size > 1 and len(clusters) > 1:
        clusters = _merge_small_clusters(clusters, min_cluster_size)

    # ── Step 6: Compute inter-cluster similarity edges ────────────────────────

    inter_cluster_edges = _compute_inter_cluster_edges(clusters)

    logger.info(
        "Clustered %d entities into %d clusters (model=%s, threshold=%.2f, edges=%d)",
        len(entities),
        len(clusters),
        embedder.model_name,
        distance_threshold,
        len(inter_cluster_edges),
    )

    return ClusterResult(clusters=clusters, inter_cluster_edges=inter_cluster_edges)


def _merge_small_clusters(
    clusters: list[EntityCluster],
    min_size: int,
) -> list[EntityCluster]:
    """
    Merge clusters smaller than min_size into the nearest large cluster.

    "Nearest" is defined by centroid cosine similarity. Cluster IDs are
    reassigned sequentially after merging.
    """
    large = [c for c in clusters if len(c.entities) >= min_size]
    small = [c for c in clusters if len(c.entities) < min_size]

    if not large:
        # All clusters are tiny — cannot merge into nothing, keep as-is
        return clusters

    for small_cluster in small:
        if not large:
            break
        if not small_cluster.centroid:
            large[0].entities.extend(small_cluster.entities)
            continue

        best_idx = 0
        best_sim = -2.0
        for idx, lc in enumerate(large):
            if not lc.centroid:
                continue
            sim = cosine_similarity(small_cluster.centroid, lc.centroid)
            if sim > best_sim:
                best_sim = sim
                best_idx = idx

        large[best_idx].entities.extend(small_cluster.entities)

    # Rebuild clusters with fresh centroids, labels, and sequential IDs
    rebuilt: list[EntityCluster] = []
    for new_idx, lc in enumerate(large):
        member_embeddings = [m.embedding for m in lc.entities]
        centroid = _compute_centroid(member_embeddings)
        label = _most_central_entity(lc.entities, centroid)

        rebuilt.append(
            EntityCluster(
                id=f"cluster-{new_idx}",
                entities=list(lc.entities),
                centroid=centroid,
                label=label,
            )
        )

    return rebuilt


# ─── Utility: cluster summary ─────────────────────────────────────────────────


def summarize_cluster_result(result: ClusterResult) -> str:
    """
    Return a human-readable summary of a ClusterResult for logging/debugging.

    Example output:
        ClusterResult: 3 clusters, 5 inter-cluster edges
          cluster-0 [PostgreSQL] (3 entities)
          cluster-1 [embeddings] (4 entities)
          cluster-2 [SQLite] (2 entities)
    """
    total_entities = sum(len(c.entities) for c in result.clusters)
    lines = [
        f"ClusterResult: {total_entities} entities → {len(result.clusters)} clusters, "
        f"{len(result.inter_cluster_edges)} inter-cluster edges"
    ]
    for c in result.clusters:
        lines.append(
            f"  {c.id} [{c.display_label()!r}] ({len(c.entities)} entities)"
        )
    return "\n".join(lines)
