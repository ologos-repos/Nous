"""Core types for the Nous memory system."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


class MemoryTier(str, enum.Enum):
    """Which tier a memory belongs to."""

    DIRECTOR = "director"  # Tier 1: curated, embedded
    SHARED = "shared"  # Tier 2: worker shared, name-scoped
    PRIVATE = "private"  # Tier 3: worker shell, importance-weighted


class MemoryCategory(str, enum.Enum):
    """Standard memory categories. Extensible — users can define custom categories."""

    PREFERENCE = "preference"
    LESSON = "lesson"
    FACT = "fact"
    DECISION = "decision"
    PROJECT = "project"
    PERSON = "person"
    GENERAL = "general"


@dataclass
class Memory:
    """A single memory entry."""

    id: int | str
    content: str
    category: str = "general"
    tier: MemoryTier = MemoryTier.DIRECTOR
    importance: float = 0.5  # 0.0–1.0, used for Tier 3 retention
    worker_name: str | None = None  # Set for Tier 2 and 3
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @property
    def age_days(self) -> float:
        """Days since creation."""
        if self.created_at is None:
            return 0.0
        if self.created_at.tzinfo:
            now = datetime.now(timezone.utc)
        else:
            now = datetime.now(timezone.utc).replace(tzinfo=None)
        return (now - self.created_at).total_seconds() / 86400


@dataclass
class SearchResult:
    """A memory search result with relevance score."""

    memory: Memory
    score: float  # 0.0–1.0 relevance (cosine similarity or keyword match rank)
    match_type: str = "semantic"  # "semantic" or "keyword"


@dataclass
class WorkerShell:
    """Represents a worker's private SQLite shell."""

    worker_name: str
    db_path: str
    memories_count: int = 0
    knowledge_count: int = 0
    instructions_count: int = 0
    tasks_completed: int = 0


@dataclass
class ConversationTurn:
    """A single turn in the conversation log."""

    id: int | str
    role: str  # "user", "assistant", "system"
    content: str
    created_at: datetime | None = None


@dataclass
class WorkerResume:
    """A worker's task completion record."""

    task_id: str
    description: str
    outcome: str  # "completed", "failed"
    skills_used: str | None = None
    summary: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


@dataclass
class Triplet:
    """
    A knowledge graph triplet (subject, predicate, object).

    Universal — not scoped to conversations. source_type identifies the origin
    system ('conversation', 'memory', 'project', 'task', 'note', 'worker_resume',
    etc.) and source_id is the ID within that system.
    """

    id: int
    subject: str
    predicate: str
    object: str
    source_type: str = "conversation"
    source_id: str = ""
    confidence: float = 1.0
    created_at: Optional[str] = None


@dataclass
class GraphContext:
    """Result of a graph-enhanced recall."""

    rag_results: list  # SearchResult objects
    graph_triplets: list  # Triplet objects from walk
    discovered_turns: list  # ConversationTurn objects found via graph (source_type='conversation')
    entities: set  # All entities encountered in the walk


@dataclass
class RetentionPolicy:
    """Importance-weighted retention rules for worker shell memories."""

    low_threshold: float = 0.3
    medium_threshold: float = 0.7
    low_retention_days: int = 30
    medium_retention_days: int = 90
    high_retention_days: int = 180
    preserve_categories: set[str] = field(default_factory=lambda: {"decision", "lesson"})

    def should_prune(self, memory: Memory) -> bool:
        """Check if a memory should be pruned based on importance and age."""
        if memory.category in self.preserve_categories:
            return False
        age = memory.age_days
        if memory.importance < self.low_threshold:
            return age > self.low_retention_days
        elif memory.importance < self.medium_threshold:
            return age > self.medium_retention_days
        else:
            return age > self.high_retention_days
