"""
Context Assembler — builds agent context windows from all memory tiers.

This is the glue between the storage layer and the agent runtime. Before each
agent invocation, the assembler pulls relevant memories from all three tiers
and formats them for injection into the system prompt or message context.

Usage:
    assembler = ContextAssembler(store)

    # Director context (startup or per-message)
    context = await assembler.build_director_context(
        query="what's the deployment strategy?",
        include_conversations=True,
        conversation_limit=5,
    )

    # Worker context (task launch)
    context = await assembler.build_worker_context(
        worker_name="alpha",
        task_description="fix the auth bug",
    )
"""

from __future__ import annotations

import logging
from typing import Any

from nous.store import MemoryStore
from nous.types import GraphContext, Memory, SearchResult, Triplet

logger = logging.getLogger(__name__)


class ContextAssembler:
    """
    Builds context windows from the memory store for injection into agent prompts.

    The assembler understands the three-tier memory model and knows how to pull
    the right information at the right time for each agent tier.
    """

    def __init__(
        self,
        store: MemoryStore,
        max_context_chars: int = 50000,
        memory_search_limit: int = 15,
        conversation_window_hours: float = 2.0,
    ):
        """
        Args:
            store: The MemoryStore to pull context from
            max_context_chars: Maximum total characters in assembled context
            memory_search_limit: Maximum memories to include from semantic search
            conversation_window_hours: Time window for recent conversation inclusion
        """
        self._store = store
        self._max_chars = max_context_chars
        self._search_limit = memory_search_limit
        self._conv_window_hours = conversation_window_hours

    async def build_director_context(
        self,
        query: str | None = None,
        include_conversations: bool = True,
        conversation_limit: int = 5,
        include_memories: bool = True,
        include_documents: bool = True,
        include_graph: bool = True,
        extra_sections: dict[str, str] | None = None,
    ) -> str:
        """
        Build context for the director agent.

        This assembles:
        1. Recent conversations (sliding window)
        2. Relevant director memories (semantic/hybrid search, or graph-enhanced)
        3. Relevant document chunks (RAG search)
        4. Knowledge graph context (when include_graph=True and a query is provided)
        5. Any extra sections provided by the caller

        Args:
            query: Current query for semantic search (None = skip memory/doc search)
            include_conversations: Whether to include recent conversation history
            conversation_limit: Max conversation turns to include
            include_memories: Whether to search and include relevant memories
            include_documents: Whether to search and include relevant documents
            include_graph: Whether to use graph-enhanced recall (default True).
                           When True and a query is provided, calls
                           graph_enhanced_recall() instead of plain recall(),
                           and adds a [KNOWLEDGE GRAPH CONTEXT] section.
            extra_sections: Additional named sections to include (e.g., {"KANBAN": "..."})

        Returns:
            Formatted context string ready for prompt injection
        """
        sections = []
        char_budget = self._max_chars

        # Recent conversations
        if include_conversations:
            turns = await self._store.get_recent_conversations(
                limit=conversation_limit,
                hours_window=self._conv_window_hours,
            )
            if turns:
                conv_text = self._format_conversations(turns)
                if len(conv_text) <= char_budget:
                    sections.append(("[RECENT CONTEXT]", conv_text))
                    char_budget -= len(conv_text)

        # Memory search — graph-enhanced or plain
        if include_memories and query:
            if include_graph:
                try:
                    graph_context = await self._store.graph_enhanced_recall(
                        query, limit=self._search_limit
                    )
                    # Include RAG hits as the memory section
                    if graph_context.rag_results:
                        mem_text = self._format_memories(graph_context.rag_results)
                        if len(mem_text) <= char_budget:
                            sections.append(("[MEMORY CONTEXT]", mem_text))
                            char_budget -= len(mem_text)
                    # Include graph context section
                    if graph_context.graph_triplets or graph_context.discovered_turns:
                        graph_text = self._format_graph_context(graph_context, char_budget)
                        if graph_text and len(graph_text) <= char_budget:
                            sections.append(("[KNOWLEDGE GRAPH CONTEXT]", graph_text))
                            char_budget -= len(graph_text)
                except Exception as e:
                    logger.warning(
                        f"graph_enhanced_recall failed, falling back to plain recall: {e}"
                    )
                    results = await self._store.recall(query, limit=self._search_limit)
                    if results:
                        mem_text = self._format_memories(results)
                        if len(mem_text) <= char_budget:
                            sections.append(("[MEMORY CONTEXT]", mem_text))
                            char_budget -= len(mem_text)
            else:
                results = await self._store.recall(query, limit=self._search_limit)
                if results:
                    mem_text = self._format_memories(results)
                    if len(mem_text) <= char_budget:
                        sections.append(("[MEMORY CONTEXT]", mem_text))
                        char_budget -= len(mem_text)

        # RAG document search
        if include_documents and query:
            doc_results = await self._store.search_documents(query, limit=3)
            if doc_results:
                doc_text = self._format_documents(doc_results)
                if len(doc_text) <= char_budget:
                    sections.append(("[DOCUMENT CONTEXT]", doc_text))
                    char_budget -= len(doc_text)

        # Extra sections
        if extra_sections:
            for name, content in extra_sections.items():
                if len(content) <= char_budget:
                    sections.append((f"[{name}]", content))
                    char_budget -= len(content)

        return self._assemble(sections)

    async def build_worker_context(
        self,
        worker_name: str,
        task_description: str | None = None,
        include_shell_memories: bool = True,
        shell_memory_limit: int = 50,
        include_instructions: bool = True,
        include_shared_memories: bool = True,
        shared_memory_limit: int = 20,
        include_resume: bool = True,
        resume_limit: int = 8,
        include_task_history: bool = True,
        task_history_limit: int = 10,
        include_graph: bool = False,
        extra_sections: dict[str, str] | None = None,
    ) -> str:
        """
        Build context for a worker agent at task launch.

        This assembles:
        1. Shell memories (private, grouped by category)
        2. Standing instructions
        3. Shared memories (PostgreSQL, name-scoped)
        4. Worker resume (recent task completions)
        5. Shell task history
        6. Knowledge graph context (optional, default off for workers)
        7. Any extra sections (e.g., kanban board, task-specific instructions)

        Args:
            worker_name: The worker's identity
            task_description: Current task (used for relevance, not injected here)
            include_shell_memories: Include private shell memories
            shell_memory_limit: Max shell memories
            include_instructions: Include standing instructions
            include_shared_memories: Include shared (Tier 2) memories
            shared_memory_limit: Max shared memories
            include_resume: Include worker resume from PostgreSQL
            resume_limit: Max resume entries
            include_task_history: Include shell task history
            task_history_limit: Max task history entries
            include_graph: Whether to include graph context from shared memory
                           (default False — off by default for workers to keep context lean)
            extra_sections: Additional sections (e.g., {"KANBAN BOARD": "...", "TASK": "..."})

        Returns:
            Formatted context string for worker prompt injection
        """
        sections = []
        char_budget = self._max_chars

        shell = self._store.get_shell(worker_name)

        # Shell memories (grouped by category)
        if include_shell_memories:
            memories = await shell.get_memories(limit=shell_memory_limit)
            if memories:
                grouped = self._group_by_category(memories)
                mem_text = self._format_grouped_memories(grouped)
                if len(mem_text) <= char_budget:
                    sections.append(("[PRIVATE MEMORIES]", mem_text))
                    char_budget -= len(mem_text)

        # Standing instructions
        if include_instructions:
            instructions = await shell.get_instructions()
            if instructions:
                inst_text = "\n".join(
                    f"- [{i['priority']}] {i['content']}" for i in instructions
                )
                if len(inst_text) <= char_budget:
                    sections.append(("[STANDING INSTRUCTIONS]", inst_text))
                    char_budget -= len(inst_text)

        # Shared memories (Tier 2)
        if include_shared_memories:
            shared = await self._store.get_worker_memories(
                worker_name, limit=shared_memory_limit
            )
            if shared:
                grouped = self._group_by_category(shared)
                shared_text = self._format_grouped_memories(grouped)
                if len(shared_text) <= char_budget:
                    sections.append(("[SHARED MEMORIES]", shared_text))
                    char_budget -= len(shared_text)

        # Worker resume (from PostgreSQL)
        if include_resume:
            resume = await self._store.get_worker_resume(worker_name, limit=resume_limit)
            if resume:
                resume_text = self._format_resume(resume)
                if len(resume_text) <= char_budget:
                    sections.append(("[RESUME — RECENT TASKS]", resume_text))
                    char_budget -= len(resume_text)

        # Shell task history
        if include_task_history:
            history = await shell.get_task_history(limit=task_history_limit)
            if history:
                hist_text = "\n".join(
                    f"- [{h['outcome']}] {h['description']}" +
                    (f": {h['summary'][:200]}" if h.get("summary") else "")
                    for h in history
                )
                if len(hist_text) <= char_budget:
                    sections.append(("[TASK HISTORY]", hist_text))
                    char_budget -= len(hist_text)

        # Graph context (optional, default off for workers)
        if include_graph and task_description:
            try:
                graph_context = await self._store.graph_enhanced_recall(
                    task_description, limit=10
                )
                if graph_context.graph_triplets or graph_context.discovered_turns:
                    graph_text = self._format_graph_context(graph_context, char_budget)
                    if graph_text and len(graph_text) <= char_budget:
                        sections.append(("[KNOWLEDGE GRAPH CONTEXT]", graph_text))
                        char_budget -= len(graph_text)
            except Exception as e:
                logger.warning(f"build_worker_context: graph_enhanced_recall failed: {e}")

        # Extra sections
        if extra_sections:
            for name, content in extra_sections.items():
                if len(content) <= char_budget:
                    sections.append((f"[{name}]", content))
                    char_budget -= len(content)

        return self._assemble(sections)

    async def build_startup_context(
        self,
        conversation_limit: int = 20,
        session_summaries: list[str] | None = None,
        reboot_prompt: str | None = None,
    ) -> str:
        """
        Build context for director agent startup (boot/reboot).

        Assembles:
        1. Recent conversation history (from DB)
        2. Session summaries (last N shutdown digests)
        3. Reboot prompt (if recovering from restart)
        """
        sections = []

        # Conversation history
        turns = await self._store.get_recent_conversations(limit=conversation_limit)
        if turns:
            sections.append(
                ("[CONVERSATION HISTORY]", self._format_conversations(turns))
            )

        # Session summaries
        if session_summaries:
            for i, summary in enumerate(session_summaries):
                sections.append(
                    (f"[SESSION SUMMARY {i + 1}]", summary)
                )

        # Reboot prompt
        if reboot_prompt:
            sections.append(("[REBOOT CONTINUATION]", reboot_prompt))

        return self._assemble(sections)

    # ─── Formatting Helpers ────────────────────────────────────────────

    def _assemble(self, sections: list[tuple[str, str]]) -> str:
        """Assemble sections into a single context string."""
        if not sections:
            return ""
        parts = []
        for header, content in sections:
            parts.append(f"{header}\n{content}")
        return "\n\n".join(parts)

    def _format_conversations(self, turns) -> str:
        """Format conversation turns for context injection."""
        lines = []
        for turn in turns:
            role = turn.role.upper()
            content = turn.content
            if len(content) > 500:
                content = content[:500] + "..."
            lines.append(f"[{role}]: {content}")
        return "\n".join(lines)

    def _format_memories(self, results: list[SearchResult]) -> str:
        """Format search results for context injection."""
        lines = []
        for r in results:
            score_pct = int(r.score * 100)
            lines.append(f"- [{r.memory.category}] ({score_pct}%) {r.memory.content}")
        return "\n".join(lines)

    def _format_documents(self, results: list[tuple[str, str, float]]) -> str:
        """Format document search results."""
        lines = []
        for doc_path, content, score in results:
            score_pct = int(score * 100)
            lines.append(f"--- {doc_path} ({score_pct}% match) ---\n{content[:1000]}")
        return "\n\n".join(lines)

    def _format_resume(self, resume) -> str:
        """Format worker resume entries."""
        lines = []
        for entry in resume:
            line = f"- [{entry.outcome}] {entry.description}"
            if entry.summary:
                line += f": {entry.summary[:200]}"
            if entry.skills_used:
                line += f" (skills: {entry.skills_used})"
            lines.append(line)
        return "\n".join(lines)

    def _format_graph_context(
        self, graph_context: GraphContext, char_budget: int = 10000
    ) -> str:
        """
        Format a GraphContext for prompt injection.

        Triplets are grouped by source (source_type + source_id) for readability.
        Discovered conversation turns are appended below the triplets.
        Total output is kept within char_budget.

        Args:
            graph_context: The GraphContext returned by graph_enhanced_recall()
            char_budget: Maximum characters to output

        Returns:
            Formatted string, or "" if nothing to show or budget exhausted
        """
        parts: list[str] = []
        running_len = 0

        # Group triplets by (source_type, source_id)
        grouped: dict[tuple[str, str], list[Triplet]] = {}
        for t in graph_context.graph_triplets:
            key = (t.source_type, t.source_id)
            grouped.setdefault(key, []).append(t)

        if grouped:
            parts.append("Triplets from knowledge graph:")
            running_len += len(parts[-1]) + 1

            for (source_type, source_id), triplets in sorted(grouped.items()):
                label = f"  [{source_type}:{source_id}]" if source_id else f"  [{source_type}]"
                label_line = label + "\n"
                if running_len + len(label_line) > char_budget:
                    break
                parts.append(label)
                running_len += len(label_line)

                for t in triplets:
                    line = f"    {t.subject} → {t.predicate} → {getattr(t, 'object')}"
                    if running_len + len(line) + 1 > char_budget:
                        break
                    parts.append(line)
                    running_len += len(line) + 1

        # Discovered turns (from graph walk, not in original RAG hits)
        if graph_context.discovered_turns:
            header = "Discovered via graph walk:"
            if running_len + len(header) + 1 <= char_budget:
                parts.append(header)
                running_len += len(header) + 1

                for turn in graph_context.discovered_turns:
                    content = turn.content
                    if len(content) > 300:
                        content = content[:300] + "..."
                    line = f"  [{turn.role.upper()}]: {content}"
                    if running_len + len(line) + 1 > char_budget:
                        break
                    parts.append(line)
                    running_len += len(line) + 1

        if not parts:
            return ""

        return "\n".join(parts)

    def _group_by_category(self, memories: list[Memory]) -> dict[str, list[Memory]]:
        """Group memories by category."""
        grouped: dict[str, list[Memory]] = {}
        for m in memories:
            grouped.setdefault(m.category, []).append(m)
        return grouped

    def _format_grouped_memories(self, grouped: dict[str, list[Memory]]) -> str:
        """Format grouped memories with category headers."""
        parts = []
        for category, memories in sorted(grouped.items()):
            parts.append(f"## {category}")
            for m in memories:
                imp = f" (importance: {m.importance:.1f})" if m.importance != 0.5 else ""
                parts.append(f"- {m.content}{imp}")
        return "\n".join(parts)
