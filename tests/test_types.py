"""Tests for types and retention policy — no external dependencies."""

from datetime import datetime, timedelta, timezone
from nous.types import Memory, MemoryTier, RetentionPolicy


def _utcnow():
    """Helper to get timezone-naive UTC now (matching PostgreSQL default behavior)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def test_memory_age_days():
    m = Memory(
        id=1,
        content="test",
        created_at=_utcnow() - timedelta(days=10),
    )
    assert 9.9 < m.age_days < 10.1


def test_memory_age_no_created_at():
    m = Memory(id=1, content="test")
    assert m.age_days == 0.0


def test_retention_low_importance_old():
    policy = RetentionPolicy()
    m = Memory(
        id=1,
        content="ephemeral note",
        importance=0.1,
        created_at=_utcnow() - timedelta(days=45),
    )
    assert policy.should_prune(m) is True


def test_retention_low_importance_recent():
    policy = RetentionPolicy()
    m = Memory(
        id=1,
        content="ephemeral note",
        importance=0.1,
        created_at=_utcnow() - timedelta(days=5),
    )
    assert policy.should_prune(m) is False


def test_retention_high_importance_old():
    policy = RetentionPolicy()
    m = Memory(
        id=1,
        content="important decision",
        importance=0.9,
        created_at=_utcnow() - timedelta(days=100),
    )
    assert policy.should_prune(m) is False  # 100 < 180


def test_retention_preserved_category():
    policy = RetentionPolicy(preserve_categories={"decision"})
    m = Memory(
        id=1,
        content="old decision",
        category="decision",
        importance=0.0,
        created_at=_utcnow() - timedelta(days=365),
    )
    assert policy.should_prune(m) is False


def test_retention_medium_importance():
    policy = RetentionPolicy()
    m = Memory(
        id=1,
        content="medium note",
        importance=0.5,
        created_at=_utcnow() - timedelta(days=95),
    )
    assert policy.should_prune(m) is True  # 95 > 90


def test_memory_tier_enum():
    assert MemoryTier.DIRECTOR.value == "director"
    assert MemoryTier.SHARED.value == "shared"
    assert MemoryTier.PRIVATE.value == "private"
