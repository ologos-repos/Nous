"""
visual_recall — orchestration function for visual memory recall.

Chains:
    store.graph_enhanced_recall() → entity extraction → embed_and_cluster()
    → build_ologic() → .ologic YAML

Returns a tuple of (ologic_yaml_str, summary_text) suitable for rendering
via the Ordinal render endpoint.

Usage:
    from nous.visual import visual_recall
    from nous.embeddings import OllamaEmbedder

    embedder = OllamaEmbedder()
    yaml_str, summary = await visual_recall("authentication system", store, embedder)
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from nous.hierarchy import HierarchyEntity, build_ologic
from nous.semantic_cluster import embed_and_cluster
from nous.types import GraphContext, SearchResult, Triplet

if TYPE_CHECKING:
    from nous.embeddings import EmbeddingProvider
    from nous.store import MemoryStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Predicate → node_type mapping
# ---------------------------------------------------------------------------

_PREDICATE_NODE_TYPES: dict[str, str] = {
    "uses": "process",
    "depends_on": "process",
    "stores": "database",
    "persists": "database",
    "serves": "api",
    "exposes": "api",
    "monitors": "monitor",
    "observes": "monitor",
    "decides": "decision",
    "routes": "decision",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slugify(text: str) -> str:
    """Convert a string to a URL/ID-safe slug (lowercase, hyphens)."""
    slug = text.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    slug = slug.strip("-")
    return slug or "entity"


def _infer_node_type(entity_text: str, triplets: list[Triplet]) -> str:
    """
    Infer node_type for an entity from triplet predicates.

    Checks predicates where entity_text appears as the subject first.
    Also checks when entity_text is the object — e.g., if something
    "stores" this entity, it should be type "database".
    Falls back to 'process' if no matching predicate is found.
    """
    # Object-role hints: when this entity is the object of a predicate,
    # the predicate implies something about what this entity IS.
    _OBJECT_PREDICATE_NODE_TYPES: dict[str, str] = {
        "stores": "database",
        "persists": "database",
        "serves": "api",
        "exposes": "api",
        "monitors": "monitor",
        "observes": "monitor",
        "uses": "process",
        "depends_on": "process",
        "decides": "decision",
        "routes": "decision",
    }

    # Subject role takes priority
    for triplet in triplets:
        if triplet.subject == entity_text:
            pred = triplet.predicate.lower()
            if pred in _PREDICATE_NODE_TYPES:
                return _PREDICATE_NODE_TYPES[pred]

    # Object role as fallback
    for triplet in triplets:
        obj = getattr(triplet, "object", None)
        if obj == entity_text:
            pred = triplet.predicate.lower()
            if pred in _OBJECT_PREDICATE_NODE_TYPES:
                return _OBJECT_PREDICATE_NODE_TYPES[pred]

    return "process"


# Leading determiners/articles to strip from raw entity text scraped from triplets.
_ENTITY_LEADING_ARTICLES = (
    "and the ", "but the ", "or the ", "if the ", "since the ",
    "when the ", "that the ", "as the ",
    "and a ", "but a ", "or a ",
    "and an ", "but an ",
    "and ", "but ", "or ", "since ", "when ", "if ", "that ",
    "the ", "a ", "an ",
    "this ", "that ", "these ", "those ", "it ", "its ",
)

# Non-entity words that should never appear as standalone entity names.
_ENTITY_BLACKLIST: frozenset[str] = frozenset({
    "This", "It", "That", "Here", "There", "Everything", "Nothing",
    "Something", "Always", "Never", "Each", "Every", "These", "Those",
    "The", "A", "An", "We", "They", "You", "He", "She", "My", "Your",
    "His", "Her", "Its", "Our", "Their", "What", "Which", "Who", "Whom",
    "How", "When", "Where", "Why", "Just", "Also", "Since", "Because",
    "Although", "However", "Therefore", "Thus", "Then", "Now", "Still",
    "And", "But", "Or", "So", "Yet",
})


def _normalize_entity_text(text: str) -> str:
    """
    Normalize raw entity text from triplets:
    - Strip leading articles/conjunctions/determiners.
    - Strip trailing punctuation.
    - Collapse internal whitespace.
    """
    text = text.strip().rstrip('.,;:')
    lowered = text.lower()
    for prefix in _ENTITY_LEADING_ARTICLES:
        if lowered.startswith(prefix):
            text = text[len(prefix):].strip()
            break
    return text


def _is_valid_entity(text: str) -> bool:
    """
    Return True if the entity text is a valid named entity.

    Rejects:
    - Text shorter than 2 chars or longer than 40 chars.
    - Blacklisted pronouns/function words.
    - Text that doesn't start with a capital letter (sentence fragments).
    """
    if len(text) < 2 or len(text) > 40:
        return False
    if text in _ENTITY_BLACKLIST:
        return False
    # Entity names should start with a capital letter (proper noun or technical term)
    if not text[0].isupper():
        return False
    return True


def _extract_ner_entities(content: str) -> list[str]:
    """
    Simple NER: extract capitalized phrases and quoted terms from text.

    Capitalized phrases: 2+ consecutive words starting with uppercase.
    Quoted terms: anything in single or double quotes.
    """
    entities: list[str] = []

    # Quoted terms (single or double quotes)
    quoted = re.findall(r'["\']([^"\']{2,50})["\']', content)
    entities.extend(quoted)

    # Capitalized multi-word phrases (e.g. "PostgreSQL Connection Pool")
    cap_phrases = re.findall(r"\b([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)+)\b", content)
    entities.extend(cap_phrases)

    # Single capitalized words that look like proper nouns (all-caps acronyms or
    # PascalCase identifiers like PostgreSQL, SQLite, OAuth, etc.)
    cap_words = re.findall(r"\b([A-Z][a-zA-Z]{2,}[A-Z][a-zA-Z]*|[A-Z]{2,})\b", content)
    entities.extend(cap_words)

    # Deduplicate while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for e in entities:
        e = e.strip()
        if e and e not in seen:
            seen.add(e)
            result.append(e)
    return result


def _triplet_to_dict(triplet: Triplet) -> dict[str, Any]:
    """Convert a Triplet to a plain dict for metadata storage."""
    return {
        "id": triplet.id,
        "subject": triplet.subject,
        "predicate": triplet.predicate,
        "object": getattr(triplet, "object", None),
        "source_type": triplet.source_type,
        "source_id": triplet.source_id,
        "confidence": triplet.confidence,
    }


# ---------------------------------------------------------------------------
# Entity extraction from GraphContext
# ---------------------------------------------------------------------------


def extract_entities_from_graph(graph_context: GraphContext) -> list[HierarchyEntity]:
    """
    Convert a GraphContext into a list of HierarchyEntity objects.

    Step 1: Extract unique entity strings from graph_triplets (subjects + objects).
    Step 2: Normalize entity text (strip leading articles, validate length/blacklist).
    Step 3: Case-insensitive deduplication of near-matches.
    Step 4: Infer node_type from predicate patterns.
    Step 5: Collect source_triplets for metadata.
    Step 6: Supplement with entities from rag_results via simple NER.

    Returns a list of HierarchyEntity objects ready for embed_and_cluster().
    """
    triplets: list[Triplet] = graph_context.graph_triplets or []
    rag_results: list[SearchResult] = graph_context.rag_results or []

    # Map: normalized_entity_text → list of triplets involving this entity
    entity_triplets: dict[str, list[Triplet]] = {}

    # Case-insensitive dedup index: lowercase_text → canonical_text
    _lower_to_canonical: dict[str, str] = {}

    def _register_entity(raw_text: str, triplet: Triplet) -> None:
        """Normalize, validate, and register an entity from a triplet."""
        text = _normalize_entity_text(raw_text)
        if not _is_valid_entity(text):
            return
        # Case-insensitive dedup: prefer the first seen casing
        lower = text.lower()
        if lower in _lower_to_canonical:
            canonical = _lower_to_canonical[lower]
        else:
            _lower_to_canonical[lower] = text
            canonical = text

        if canonical not in entity_triplets:
            entity_triplets[canonical] = []
        entity_triplets[canonical].append(triplet)

    for triplet in triplets:
        subj = triplet.subject
        obj = getattr(triplet, "object", None)

        if subj:
            _register_entity(subj, triplet)
        if obj:
            _register_entity(obj, triplet)

    # Build HierarchyEntity for each unique entity from triplets
    # Track seen slugs to avoid ID collisions
    seen_ids: dict[str, str] = {}  # slug → entity_text
    entities_map: dict[str, HierarchyEntity] = {}

    for entity_text, related_triplets in entity_triplets.items():
        eid = _slugify(entity_text)
        base_eid = eid
        suffix = 1
        while eid in seen_ids and seen_ids[eid] != entity_text:
            eid = f"{base_eid}-{suffix}"
            suffix += 1
        seen_ids[eid] = entity_text

        node_type = _infer_node_type(entity_text, triplets)

        entities_map[entity_text] = HierarchyEntity(
            id=eid,
            text=entity_text,
            embedding=[],
            requirements=[],
            node_type=node_type,
            metadata={"source_triplets": [_triplet_to_dict(t) for t in related_triplets]},
        )

    # Track seen entity texts (canonical + lowercase) for NER dedup
    seen_lowers: set[str] = {t.lower() for t in entities_map}

    # Supplement with entities from rag_results (simple NER)
    for result in rag_results:
        try:
            content = result.memory.content
        except AttributeError:
            continue
        ner_entities = _extract_ner_entities(content)
        for ner_text in ner_entities:
            # Normalize and validate NER entity
            ner_text = _normalize_entity_text(ner_text)
            if not _is_valid_entity(ner_text):
                continue
            if ner_text.lower() in seen_lowers:
                continue
            seen_lowers.add(ner_text.lower())

            eid = _slugify(ner_text)
            base_eid = eid
            suffix = 1
            while eid in seen_ids and seen_ids[eid] != ner_text:
                eid = f"{base_eid}-{suffix}"
                suffix += 1
            seen_ids[eid] = ner_text

            entities_map[ner_text] = HierarchyEntity(
                id=eid,
                text=ner_text,
                embedding=[],
                requirements=[],
                node_type="process",
                metadata={"source_triplets": [], "from_ner": True},
            )

    return list(entities_map.values())


# ---------------------------------------------------------------------------
# Edge extraction: build outputs on entities from triplets
# ---------------------------------------------------------------------------


def apply_triplet_edges(
    entities_map: dict[str, HierarchyEntity],
    triplets: list[Triplet],
) -> None:
    """
    For each triplet, set subject entity's outputs to include the object entity's id.

    Mutates entities in-place by updating metadata with 'outputs' list.
    """
    for triplet in triplets:
        subj = triplet.subject
        obj = getattr(triplet, "object", None)
        if subj in entities_map and obj and obj in entities_map:
            subject_entity = entities_map[subj]
            outputs = subject_entity.metadata.get("outputs", [])
            obj_id = entities_map[obj].id
            if obj_id not in outputs:
                outputs.append(obj_id)
            subject_entity.metadata["outputs"] = outputs


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------


async def visual_recall(
    query: str,
    store: "MemoryStore",
    embedder: "EmbeddingProvider",
    **kwargs: Any,
) -> tuple[str, str]:
    """
    Visual memory recall pipeline.

    Chains:
        graph_enhanced_recall → entity extraction → embed_and_cluster
        → build_ologic → .ologic YAML

    Args:
        query:    Natural language query to search memory.
        store:    MemoryStore instance (must support graph_enhanced_recall).
        embedder: EmbeddingProvider for semantic embedding.
        **kwargs: Additional kwargs (unused, for forward compatibility).

    Returns:
        (ologic_yaml_str, summary_text)
        ologic_yaml_str is empty string if not enough graph data.
    """
    logger.info("visual_recall: query=%r", query)

    # ── Step 1: Graph-enhanced recall ─────────────────────────────────────────
    graph_context: GraphContext = await store.graph_enhanced_recall(
        query, limit=15, hops=2
    )

    triplets: list[Triplet] = graph_context.graph_triplets or []
    rag_results: list[SearchResult] = graph_context.rag_results or []

    logger.debug(
        "visual_recall: %d triplets, %d rag_results",
        len(triplets),
        len(rag_results),
    )

    # ── Step 2 & 3: Extract entities ──────────────────────────────────────────
    entities_list = extract_entities_from_graph(graph_context)

    # ── Step 4: Edge case — fewer than 2 entities ─────────────────────────────
    if len(entities_list) < 2:
        summary = f"Not enough graph data for query: {query}"
        logger.info("visual_recall: %s (found %d entities)", summary, len(entities_list))
        return ("", summary)

    # ── Step 5: Apply triplet edges to entities ────────────────────────────────
    entities_map: dict[str, HierarchyEntity] = {e.text: e for e in entities_list}
    apply_triplet_edges(entities_map, triplets)

    # ── Step 6: Embed and cluster (graph-aware) ────────────────────────────────
    # Extract graph edges from entity metadata for graph-constrained clustering.
    # Graph-connected entities get their cosine distance reduced so HAC prefers
    # keeping them in the same cluster — prevents the 24k-wide horizontal strip.
    graph_edges: list[tuple[str, str]] = []
    for entity in entities_list:
        for target_id in entity.metadata.get("outputs", []):
            graph_edges.append((entity.id, target_id))

    logger.info(
        "visual_recall: embedding %d entities, %d graph edges",
        len(entities_list),
        len(graph_edges),
    )
    cluster_result = await embed_and_cluster(
        entities_list, embedder, distance_threshold=0.7,
        graph_edges=graph_edges if graph_edges else None,
    )

    # ── Step 7: Build .ologic YAML ────────────────────────────────────────────
    network_id = f"memory-{_slugify(query)}"
    bridges, yaml_str = build_ologic(
        cluster_result,
        similarity_threshold=0.35,
        network_id=network_id,
    )

    # ── Step 8: Generate summary ──────────────────────────────────────────────
    n_entities = len(entities_list)
    n_clusters = len(cluster_result.clusters)
    n_bridges = len(bridges)

    # Top 5 entities: sort by number of source_triplets (most connected first)
    sorted_entities = sorted(
        entities_list,
        key=lambda e: len(e.metadata.get("source_triplets", [])),
        reverse=True,
    )
    top_5 = [e.text for e in sorted_entities[:5]]
    top_5_str = ", ".join(top_5) if top_5 else "(none)"

    summary = (
        f"{n_entities} entities across {n_clusters} clusters, "
        f"{n_bridges} bridges detected. "
        f"Top entities: {top_5_str}"
    )

    logger.info("visual_recall: %s", summary)

    # ── Step 9: Return ────────────────────────────────────────────────────────
    return (yaml_str, summary)
