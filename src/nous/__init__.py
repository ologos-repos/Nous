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
from nous.types import Memory, WorkerShell, SearchResult, MemoryCategory, MemoryTier

__version__ = "0.1.0"

__all__ = [
    "MemoryStore",
    "extract_search_queries",
    "ContextAssembler",
    "EmbeddingProvider",
    "OllamaEmbedder",
    "NullEmbedder",
    "Memory",
    "WorkerShell",
    "SearchResult",
    "MemoryCategory",
    "MemoryTier",
]
