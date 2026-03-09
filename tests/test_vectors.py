"""Tests for the vector math module — zero-dependency, runs without PostgreSQL."""

from nous.vectors import cosine_similarity, serialize_vector, deserialize_vector, top_k_similar


def test_cosine_similarity_identical():
    v = [1.0, 2.0, 3.0]
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-6


def test_cosine_similarity_orthogonal():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert abs(cosine_similarity(a, b)) < 1e-6


def test_cosine_similarity_opposite():
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    assert abs(cosine_similarity(a, b) - (-1.0)) < 1e-6


def test_cosine_similarity_empty():
    assert cosine_similarity([], []) == 0.0
    assert cosine_similarity([1.0], []) == 0.0


def test_cosine_similarity_zero_vector():
    assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


def test_serialize_deserialize_roundtrip():
    original = [1.0, -2.5, 3.14159, 0.0, 99.99]
    data = serialize_vector(original)
    restored = deserialize_vector(data)
    assert len(restored) == len(original)
    for a, b in zip(original, restored):
        assert abs(a - b) < 1e-5


def test_serialize_empty():
    assert serialize_vector([]) == b""
    assert deserialize_vector(b"") == []


def test_top_k_similar():
    query = [1.0, 0.0, 0.0]
    candidates = [
        ("a", [1.0, 0.0, 0.0]),  # identical
        ("b", [0.0, 1.0, 0.0]),  # orthogonal
        ("c", [0.9, 0.1, 0.0]),  # similar
        ("d", [-1.0, 0.0, 0.0]),  # opposite
    ]

    results = top_k_similar(query, candidates, k=2, threshold=0.5)
    assert len(results) == 2
    assert results[0][0] == "a"  # most similar
    assert results[1][0] == "c"  # second most


def test_top_k_similar_threshold():
    query = [1.0, 0.0]
    candidates = [
        ("a", [1.0, 0.0]),
        ("b", [0.0, 1.0]),
    ]

    results = top_k_similar(query, candidates, threshold=0.9)
    assert len(results) == 1
    assert results[0][0] == "a"


def test_top_k_similar_empty():
    assert top_k_similar([1.0], [], k=5) == []
