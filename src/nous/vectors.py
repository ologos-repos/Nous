"""
Vector math utilities for Nous.

Pure-Python cosine similarity and vector serialization — zero dependencies.
Uses struct.pack/unpack for binary serialization (compatible with PostgreSQL BYTEA
and SQLite BLOB columns).

This deliberately avoids numpy to keep the dependency footprint at zero for
the core library. For production systems processing >10K vectors, consider
using pgvector or a dedicated ANN index instead.
"""

from __future__ import annotations

import math
import struct


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Returns:
        Float between -1.0 and 1.0. Higher = more similar.
        Returns 0.0 if either vector is zero-length or all zeros.
    """
    if not a or not b or len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))

    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0

    return dot / (mag_a * mag_b)


def serialize_vector(vector: list[float]) -> bytes:
    """
    Serialize a float vector to binary (little-endian floats).

    This format is directly storable in PostgreSQL BYTEA or SQLite BLOB columns.

    Args:
        vector: List of floats (e.g., 768-dimensional embedding)

    Returns:
        Binary bytes — len(vector) * 4 bytes
    """
    if not vector:
        return b""
    return struct.pack(f"<{len(vector)}f", *vector)


def deserialize_vector(data: bytes) -> list[float]:
    """
    Deserialize binary data back to a float vector.

    Args:
        data: Binary bytes from serialize_vector()

    Returns:
        List of floats
    """
    if not data:
        return []
    count = len(data) // 4
    return list(struct.unpack(f"<{count}f", data))


def top_k_similar(
    query_vector: list[float],
    candidates: list[tuple[any, list[float]]],
    k: int = 10,
    threshold: float = 0.0,
) -> list[tuple[any, float]]:
    """
    Find the top-k most similar vectors to a query.

    Args:
        query_vector: The query embedding
        candidates: List of (id, vector) tuples to search
        k: Maximum results to return
        threshold: Minimum similarity score (0.0-1.0) to include

    Returns:
        List of (id, score) tuples, sorted by descending similarity
    """
    scored = []
    for item_id, vector in candidates:
        score = cosine_similarity(query_vector, vector)
        if score >= threshold:
            scored.append((item_id, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]
