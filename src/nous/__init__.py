"""
Nous — Persistent Agent Memory Architecture

Three-tier memory, hybrid PostgreSQL+SQLite storage, embedded vector retrieval,
multi-agent isolation, context injection, and crash recovery for always-on agent systems.

Quick start:
    from nous import MemoryStore, ContextAssembler

    store = await MemoryStore.connect(postgres_url="postgresql://...", shell_dir="./shells")
    assembler = ContextAssembler(store)

    await store.remember("project uses Python 3.12", category="fact")
    context = await assembler.build_context("what Python version?")
"""

from nous.store import MemoryStore, extract_search_queries
from nous.context import ContextAssembler
from nous.embeddings import EmbeddingProvider, OllamaEmbedder, NullEmbedder
from nous.types import (
    GraphContext,
    Memory,
    MemoryCategory,
    MemoryTier,
    SearchResult,
    Triplet,
    WorkerShell,
)
from nous.semantic_cluster import embed_and_cluster, summarize_cluster_result
from nous.hierarchy import (
    Bridge,
    BridgeDetector,
    ClusterResult,
    EntityCluster,
    HierarchyEntity,
    OlogicEmitter,
    build_ologic,
    build_ologic_from_cluster_result,
)
from nous.ologic import (
    REQUIREMENTS,
    VALID_NODE_TYPES,
    OlogicError,
    OlogicValidationResult,
    OlogicValidator,
    ReconciliationResult,
    reconcile_requirements,
    validate_ologic_yaml,
)

__version__ = "0.1.0"

__all__ = [
    # Core memory
    "MemoryStore",
    "extract_search_queries",
    "ContextAssembler",
    # Embeddings
    "EmbeddingProvider",
    "OllamaEmbedder",
    "NullEmbedder",
    # Memory types
    "GraphContext",
    "Memory",
    "MemoryCategory",
    "MemoryTier",
    "SearchResult",
    "Triplet",
    "WorkerShell",
    # Semantic clustering (REQ-NKGL-002, REQ-NKGL-003)
    "embed_and_cluster",
    "summarize_cluster_result",
    # Semantic hierarchy / .ologic generation (REQ-NKGL-004 through 007)
    "Bridge",
    "BridgeDetector",
    "ClusterResult",
    "EntityCluster",
    "HierarchyEntity",
    "OlogicEmitter",
    "build_ologic",
    "build_ologic_from_cluster_result",
    # .ologic validator and requirement reconciler (REQ-NKGL-007)
    "REQUIREMENTS",
    "VALID_NODE_TYPES",
    "OlogicError",
    "OlogicValidationResult",
    "OlogicValidator",
    "ReconciliationResult",
    "reconcile_requirements",
    "validate_ologic_yaml",
]
