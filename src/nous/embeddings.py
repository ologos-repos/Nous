"""
Embedding providers for Nous.

Supports pluggable backends — local (Ollama), cloud (OpenAI), or custom.
The EmbeddingProvider protocol defines the interface; bring your own implementation
or use the built-in ones.

Usage:
    # Local (Ollama)
    embedder = OllamaEmbedder(model="nomic-embed-text")
    vector = await embedder.embed("some text")

    # No embeddings (keyword fallback only)
    embedder = NullEmbedder()

    # Custom
    class MyEmbedder(EmbeddingProvider):
        async def embed(self, text: str) -> list[float]: ...
        async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding providers. Implement this to add a custom backend."""

    @property
    def dimensions(self) -> int:
        """Number of dimensions in the embedding vector."""
        ...

    @property
    def model_name(self) -> str:
        """Name of the embedding model (used for version tracking)."""
        ...

    async def embed(self, text: str) -> list[float]:
        """Embed a single text string. Returns a float vector."""
        ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts. Default: sequential calls to embed()."""
        ...

    async def is_available(self) -> bool:
        """Check if the embedding service is reachable."""
        ...


class NullEmbedder:
    """No-op embedder — returns empty vectors. Use when you only want keyword search."""

    @property
    def dimensions(self) -> int:
        return 0

    @property
    def model_name(self) -> str:
        return "null"

    async def embed(self, text: str) -> list[float]:
        return []

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[] for _ in texts]

    async def is_available(self) -> bool:
        return True


class OllamaEmbedder:
    """
    Embedding provider using a local Ollama instance.

    Requires: httpx (install with `pip install nous-memory[ollama]`)

    Args:
        model: Ollama model name (default: "nomic-embed-text")
        base_url: Ollama API base URL (default: "http://localhost:11434")
        dimensions: Vector dimensions for the model (default: 768 for nomic-embed-text)
        timeout: Request timeout in seconds (default: 30)
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        dimensions: int = 768,
        timeout: float = 30.0,
    ):
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._dimensions = dimensions
        self._timeout = timeout
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import httpx
            except ImportError:
                raise ImportError(
                    "httpx is required for OllamaEmbedder. "
                    "Install with: pip install nous-memory[ollama]"
                )
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
            )
        return self._client

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self._model

    async def embed(self, text: str) -> list[float]:
        """Embed a single text using Ollama's /api/embed endpoint."""
        client = self._get_client()
        response = await client.post(
            "/api/embed",
            json={"model": self._model, "input": text},
        )
        response.raise_for_status()
        data = response.json()
        # Ollama returns {"embeddings": [[...]], "model": "..."}
        embeddings = data.get("embeddings", [])
        if embeddings and len(embeddings) > 0:
            return embeddings[0]
        raise ValueError(f"Ollama returned no embeddings for model {self._model}")

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts. Ollama's /api/embed supports batch input."""
        if not texts:
            return []
        client = self._get_client()
        response = await client.post(
            "/api/embed",
            json={"model": self._model, "input": texts},
        )
        response.raise_for_status()
        data = response.json()
        embeddings = data.get("embeddings", [])
        if len(embeddings) != len(texts):
            logger.warning(
                f"Ollama returned {len(embeddings)} embeddings for {len(texts)} texts"
            )
        return embeddings

    async def is_available(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            client = self._get_client()
            response = await client.get("/api/tags")
            if response.status_code != 200:
                return False
            data = response.json()
            models = [m.get("name", "").split(":")[0] for m in data.get("models", [])]
            return self._model in models
        except Exception:
            return False

    async def close(self):
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None


class OpenAIEmbedder:
    """
    Embedding provider using OpenAI's API.

    Requires: openai (install with `pip install nous-memory[openai]`)

    Args:
        model: OpenAI model name (default: "text-embedding-3-small")
        api_key: OpenAI API key (default: reads OPENAI_API_KEY env var)
        dimensions: Vector dimensions (default: 1536 for text-embedding-3-small)
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        dimensions: int = 1536,
    ):
        self._model = model
        self._api_key = api_key
        self._dimensions = dimensions
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import openai
            except ImportError:
                raise ImportError(
                    "openai is required for OpenAIEmbedder. "
                    "Install with: pip install nous-memory[openai]"
                )
            kwargs = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            self._client = openai.AsyncOpenAI(**kwargs)
        return self._client

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self._model

    async def embed(self, text: str) -> list[float]:
        client = self._get_client()
        response = await client.embeddings.create(model=self._model, input=text)
        return response.data[0].embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._get_client()
        response = await client.embeddings.create(model=self._model, input=texts)
        # Sort by index to maintain order
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [d.embedding for d in sorted_data]

    async def is_available(self) -> bool:
        try:
            client = self._get_client()
            await client.models.retrieve(self._model)
            return True
        except Exception:
            return False

    async def close(self):
        """Close the OpenAI client."""
        if self._client is not None:
            await self._client.close()
            self._client = None
