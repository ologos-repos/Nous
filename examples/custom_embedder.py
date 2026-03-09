"""
Example: Custom embedding provider.

Shows how to implement the EmbeddingProvider protocol for any backend —
a local model, a cloud API, or even a mock for testing.
"""

import asyncio
import random
from nous import MemoryStore


class MockEmbedder:
    """
    Mock embedding provider for testing.

    Generates deterministic pseudo-random vectors based on text hash.
    Not semantically meaningful — just demonstrates the interface.
    """

    @property
    def dimensions(self) -> int:
        return 128

    @property
    def model_name(self) -> str:
        return "mock-128d"

    async def embed(self, text: str) -> list[float]:
        # Deterministic pseudo-random vector from text hash
        rng = random.Random(hash(text))
        return [rng.gauss(0, 1) for _ in range(self.dimensions)]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]

    async def is_available(self) -> bool:
        return True


async def main():
    embedder = MockEmbedder()

    store = await MemoryStore.connect(
        postgres_url="postgresql://nous:nous@localhost:5432/nous",
        shell_dir="./example_shells",
        embedder=embedder,
        run_migrations=True,
    )

    # Store with mock embeddings
    await store.remember("cats are great pets", category="fact")
    await store.remember("dogs are loyal companions", category="fact")
    await store.remember("Python 3.12 added better error messages", category="fact")

    # Search — mock embeddings won't give meaningful semantic results,
    # but the pipeline works end-to-end
    results = await store.recall("animals as pets", threshold=0.0)
    print(f"Search results ({len(results)}):")
    for r in results:
        print(f"  [{r.score:.2f}] {r.memory.content}")

    await store.close()


if __name__ == "__main__":
    asyncio.run(main())
