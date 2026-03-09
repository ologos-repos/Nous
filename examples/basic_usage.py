"""
Basic Nous usage example.

Demonstrates:
- Connecting to the memory store
- Director memory (Tier 1) — remember, recall, forget
- Worker shared memory (Tier 2)
- Worker private shell (Tier 3)
- Context assembly for prompt injection

Prerequisites:
- PostgreSQL running with a 'nous' database
- Ollama running with nomic-embed-text (or use NullEmbedder for keyword-only)
"""

import asyncio
from nous import MemoryStore, ContextAssembler, OllamaEmbedder, NullEmbedder


async def main():
    # ── Connect ────────────────────────────────────────────────────────
    # Use OllamaEmbedder for semantic search, or NullEmbedder for keyword-only
    try:
        embedder = OllamaEmbedder(model="nomic-embed-text")
        if not await embedder.is_available():
            print("Ollama not available — falling back to keyword search")
            embedder = NullEmbedder()
    except ImportError:
        print("httpx not installed — using keyword search only")
        embedder = NullEmbedder()

    store = await MemoryStore.connect(
        postgres_url="postgresql://nous:nous@localhost:5432/nous",
        shell_dir="./example_shells",
        embedder=embedder,
        run_migrations=True,
    )
    print("Connected to memory store\n")

    # ── Tier 1: Director Memory ────────────────────────────────────────
    print("=== Tier 1: Director Memory ===")

    mem1 = await store.remember(
        "Project deploys to K3s on a single-node cluster",
        category="decision",
    )
    mem2 = await store.remember(
        "Always use uv instead of pip for Python projects",
        category="preference",
    )
    mem3 = await store.remember(
        "Authentication uses JWT with 1-hour expiry",
        category="fact",
    )
    print(f"Stored 3 director memories (IDs: {mem1.id}, {mem2.id}, {mem3.id})")

    # Search
    results = await store.recall("how do we deploy?")
    print(f"\nSearch 'how do we deploy?' → {len(results)} results:")
    for r in results:
        print(f"  [{r.score:.0%}] [{r.memory.category}] {r.memory.content}")

    # Category-filtered search
    results = await store.recall("Python", category="preference")
    print(f"\nSearch 'Python' (category=preference) → {len(results)} results:")
    for r in results:
        print(f"  [{r.score:.0%}] {r.memory.content}")

    # ── Tier 2: Worker Shared Memory ───────────────────────────────────
    print("\n=== Tier 2: Worker Shared Memory ===")

    await store.worker_remember("alpha", "API rate limit is 100 req/min", category="fact")
    await store.worker_remember("alpha", "Use retry with exponential backoff", category="lesson")
    await store.worker_remember("beta", "Database uses connection pooling", category="fact")

    alpha_memories = await store.worker_recall("alpha", "rate limit")
    print(f"Alpha search 'rate limit' → {len(alpha_memories)} results:")
    for r in alpha_memories:
        print(f"  {r.memory.content}")

    # ── Tier 3: Worker Shell ───────────────────────────────────────────
    print("\n=== Tier 3: Worker Shell (Alpha) ===")

    shell = store.get_shell("alpha")

    # Private memories with importance
    await shell.remember("prefer functional style over OOP", importance=0.7, category="preference")
    await shell.remember("temporary debug note", importance=0.1, category="general")

    # Knowledge
    await shell.learn("error handling", "Always catch specific exceptions, never bare except")
    await shell.learn("testing", "Use pytest fixtures for database setup", source="code review")

    # Instructions
    await shell.add_instruction("Run tests before committing", priority=10)
    await shell.add_instruction("Use type hints on all function signatures", priority=5)

    # Stats
    stats = await shell.stats()
    print(f"Shell stats: {stats}")

    # ── Context Assembly ───────────────────────────────────────────────
    print("\n=== Context Assembly ===")

    assembler = ContextAssembler(store)

    # Build worker context (what a worker agent would see at task launch)
    worker_ctx = await assembler.build_worker_context(
        worker_name="alpha",
        task_description="fix the rate limiter",
        extra_sections={
            "KANBAN BOARD": "Alpha: fixing rate limiter | Beta: idle | Gamma: idle",
            "TASK": "Fix the rate limiter middleware — it's not resetting the counter correctly",
        },
    )
    print("Worker context preview (first 500 chars):")
    print(worker_ctx[:500])
    print(f"... ({len(worker_ctx)} chars total)")

    # ── Conversation Log ───────────────────────────────────────────────
    print("\n=== Conversation Log ===")

    await store.log_conversation("user", "Can you fix the rate limiter?")
    await store.log_conversation("assistant", "I'll look into the rate limiter middleware.")
    await store.log_conversation("user", "The counter doesn't reset at the window boundary.")

    turns = await store.get_recent_conversations(limit=5)
    print(f"Recent conversations ({len(turns)} turns):")
    for t in turns:
        print(f"  [{t.role}]: {t.content[:80]}")

    # ── Cleanup ────────────────────────────────────────────────────────
    # Clean up example data
    await store.forget(mem1.id)
    await store.forget(mem2.id)
    await store.forget(mem3.id)
    print("\nCleaned up example memories")

    await store.close()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
