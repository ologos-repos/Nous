# Nous

**Persistent memory architecture for always-on AI agent systems.**

Nous gives your agents real memory — the kind that survives reboots, spans conversations, isolates workers, and retrieves context semantically. It's the memory layer extracted from a production 24/7 multi-agent system and generalized for any agent runtime.

## What It Does

- **Three-tier memory model** — Director (curated, embedded), Worker Shared (name-scoped), Worker Private (importance-weighted SQLite shells)
- **Hybrid storage** — PostgreSQL for shared state + per-worker SQLite for isolated private state
- **Embedded vector retrieval** — Local embeddings (Ollama, OpenAI, or custom) with cosine similarity search and automatic keyword fallback
- **Context injection** — Assembles agent context windows from all three tiers before every invocation
- **Multi-agent isolation** — Workers can't read each other's memories. The director sees everything.
- **Crash recovery** — WAL mode, conversation replay, session continuity across restarts
- **Zero cloud dependency** — Everything runs locally. No vector DB service, no per-query pricing.

## What It Is NOT

Nous is **not** an agent framework. No task queue, no orchestrator, no chat interface. It's the memory layer. You bring your own agent runtime and plug Nous in.

## Install

```bash
pip install nous-memory

# With Ollama embedding support (recommended)
pip install nous-memory[ollama]

# With OpenAI embedding support
pip install nous-memory[openai]

# Everything
pip install nous-memory[all]
```

## Quick Start

```python
import asyncio
from nous import MemoryStore, ContextAssembler, OllamaEmbedder

async def main():
    # Connect with local embeddings
    embedder = OllamaEmbedder(model="nomic-embed-text")
    store = await MemoryStore.connect(
        postgres_url="postgresql://nous:nous@localhost:5432/nous",
        shell_dir="./shells",
        embedder=embedder,
    )

    # Director memory (Tier 1) — curated, embedded
    await store.remember("Project uses Python 3.12 with uv", category="decision")
    await store.remember("Deploy to K3s on masternode", category="decision")

    # Search semantically
    results = await store.recall("what Python version?")
    for r in results:
        print(f"  [{r.score:.0%}] {r.memory.content}")

    # Worker shared memory (Tier 2) — name-scoped
    await store.worker_remember("alpha", "auth module uses JWT", category="fact")

    # Worker private shell (Tier 3) — importance-weighted
    shell = store.get_shell("alpha")
    await shell.remember("prefer async patterns", importance=0.8, category="preference")
    await shell.learn("auth", "JWT tokens expire after 1 hour", source="code review")

    # Build context for agent prompt injection
    assembler = ContextAssembler(store)

    # Director context (for your orchestrator agent)
    director_ctx = await assembler.build_director_context(
        query="deployment strategy",
        include_conversations=True,
    )

    # Worker context (for a task agent)
    worker_ctx = await assembler.build_worker_context(
        worker_name="alpha",
        task_description="fix the auth token refresh",
        extra_sections={"KANBAN": "Beta: running DB migration | Gamma: idle"},
    )

    print(worker_ctx)  # Ready to inject into your agent's system prompt

    await store.close()

asyncio.run(main())
```

## Setup

### Option 1: Interactive CLI

```bash
nous-setup init
```

Walks you through PostgreSQL connection, embedding backend, and shell directory configuration. Creates `nous.toml` and runs migrations.

### Option 2: Manual

1. **PostgreSQL** — Create a database:
   ```bash
   bash scripts/setup-postgres.sh
   # Or manually:
   createdb nous
   ```

2. **Ollama** (optional, for semantic search):
   ```bash
   bash scripts/setup-ollama.sh
   # Or manually:
   ollama pull nomic-embed-text
   ```

3. **Run migrations**:
   ```bash
   nous-setup migrate
   ```

### Option 3: Programmatic

```python
store = await MemoryStore.connect(
    postgres_url="postgresql://...",
    run_migrations=True,  # Creates tables automatically
)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Your Agent Runtime                       │
│  (LangChain, Claude SDK, OpenAI, custom, whatever you use)     │
├─────────────────────────────────────────────────────────────────┤
│                      ContextAssembler                           │
│  Pulls from all 3 tiers → formats → injects into agent prompt  │
├──────────────────┬──────────────────┬───────────────────────────┤
│   Tier 1         │   Tier 2         │   Tier 3                  │
│   Director       │   Worker Shared  │   Worker Shell            │
│   (PostgreSQL)   │   (PostgreSQL)   │   (SQLite per worker)     │
│   Curated +      │   Name-scoped    │   Private + portable      │
│   embedded       │   keyword search │   importance-weighted     │
├──────────────────┴──────────────────┴───────────────────────────┤
│                     EmbeddingProvider                            │
│  Ollama (local) │ OpenAI (cloud) │ Custom │ Null (keyword only) │
└─────────────────────────────────────────────────────────────────┘
```

## Memory Tiers

| Tier | Storage | Access | Search | Use Case |
|------|---------|--------|--------|----------|
| **1. Director** | PostgreSQL | Director only | Semantic + keyword fallback | Curated knowledge, decisions, lessons |
| **2. Worker Shared** | PostgreSQL (name-scoped) | Each worker: own only. Director: all | Keyword | Cross-task knowledge workers discover |
| **3. Worker Shell** | SQLite (per-worker file) | Worker-exclusive write, director admin-read | Keyword (embedding column ready) | Private state, instructions, training data |

## Embedding Providers

```python
# Local — free, fast, no cloud dependency (recommended)
from nous import OllamaEmbedder
embedder = OllamaEmbedder(model="nomic-embed-text")

# Cloud — higher quality, requires API key
from nous import OpenAIEmbedder  # needs nous-memory[openai]
embedder = OpenAIEmbedder(model="text-embedding-3-small")

# None — keyword search only, zero dependencies
from nous import NullEmbedder
embedder = NullEmbedder()

# Custom — implement the EmbeddingProvider protocol
class MyEmbedder:
    @property
    def dimensions(self) -> int: return 768
    @property
    def model_name(self) -> str: return "my-model"
    async def embed(self, text: str) -> list[float]: ...
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
    async def is_available(self) -> bool: ...
```

## Worker Shells

Each worker gets an independent SQLite database — their portable identity:

```python
shell = store.get_shell("alpha")

# Private memories with importance scoring
await shell.remember("prefer async patterns", importance=0.8)

# Structured knowledge
await shell.learn("auth", "JWT tokens expire in 1 hour", source="code review")

# Standing instructions (persist across tasks)
await shell.add_instruction("Always run tests before committing", priority=10)

# Self-training
session_id = await shell.start_training("error handling patterns")
await shell.add_training_pair(session_id, "raw try/except", "specific exception types")
await shell.complete_training(session_id)

# Retention pruning (importance-weighted)
from nous.types import RetentionPolicy
policy = RetentionPolicy(
    low_retention_days=30,    # importance < 0.3: keep 30 days
    medium_retention_days=90, # 0.3-0.7: keep 90 days
    high_retention_days=180,  # > 0.7: keep 180 days
    preserve_categories={"decision", "lesson"},  # never prune these
)
pruned = await shell.prune(policy)
```

## RAG / Document Search

Index long-form documents for semantic retrieval:

```python
# Index a document (chunk it yourself or use your preferred chunker)
chunks = ["chunk 1 text...", "chunk 2 text...", "chunk 3 text..."]
await store.index_document("docs/architecture.md", chunks)

# Search
results = await store.search_documents("deployment strategy")
for doc_path, content, score in results:
    print(f"  [{score:.0%}] {doc_path}: {content[:100]}...")
```

## Context Injection

The `ContextAssembler` builds formatted context strings ready for prompt injection:

```python
assembler = ContextAssembler(
    store,
    max_context_chars=50000,         # Budget for total context
    memory_search_limit=15,          # Max memories per search
    conversation_window_hours=2.0,   # Recent conversation window
)

# Director startup context (boot/reboot recovery)
startup_ctx = await assembler.build_startup_context(
    conversation_limit=20,
    session_summaries=["Yesterday: deployed v2.1, fixed auth bug"],
    reboot_prompt="Continue monitoring deployment",
)

# Director per-message context
msg_ctx = await assembler.build_director_context(
    query="what's the auth setup?",
    include_conversations=True,
    include_memories=True,
    include_documents=True,
)

# Worker launch context
worker_ctx = await assembler.build_worker_context(
    worker_name="alpha",
    task_description="fix token refresh",
    extra_sections={
        "KANBAN BOARD": "Beta: running migration | Gamma: idle",
        "TASK INSTRUCTIONS": "Focus on the refresh endpoint in auth.py",
    },
)

# Inject into your agent's system prompt
system_prompt = f"""You are a coding agent.

{worker_ctx}

Your current task: fix the token refresh bug."""
```

## Configuration

Nous reads from `nous.toml` (or set `NOUS_CONFIG` env var):

```toml
postgres_url = "postgresql://nous:nous@localhost:5432/nous"
shell_dir = "./shells"

[embedding]
backend = "ollama"
model = "nomic-embed-text"
base_url = "http://localhost:11434"
dimensions = 768

[retrieval]
similarity_threshold = 0.4
max_results = 10
keyword_fallback = true

[retention]
low_threshold = 0.3
medium_threshold = 0.7
low_retention_days = 30
medium_retention_days = 90
high_retention_days = 180
preserve_categories = ["decision", "lesson"]

[context]
max_context_chars = 50000
conversation_window_hours = 2.0
memory_search_limit = 15
```

## CLI Tools

```bash
nous-setup init       # Interactive setup wizard
nous-setup check      # Verify PostgreSQL connection + table status
nous-setup migrate    # Create/update database tables
nous-setup shell alpha  # Initialize a worker shell
nous-setup stats      # Show memory statistics across all tiers
nous-setup config     # Print current configuration
```

## Adapting Nous to Your Agent System

Nous is designed to drop into whatever agent runtime you already use. Here's how to wire it in depending on your architecture.

### Single Agent (Simplest Case)

If you have one persistent agent (chatbot, assistant, coding agent), use Tier 1 only:

```python
from nous import MemoryStore, ContextAssembler, OllamaEmbedder

class MyAgent:
    def __init__(self):
        self.store = None
        self.assembler = None

    async def start(self):
        embedder = OllamaEmbedder()  # or NullEmbedder() for keyword-only
        self.store = await MemoryStore.connect(
            postgres_url="postgresql://...",
            shell_dir="./shells",
            embedder=embedder,
        )
        self.assembler = ContextAssembler(self.store)

    async def handle_message(self, user_message: str) -> str:
        # 1. Log the conversation
        await self.store.log_conversation("user", user_message)

        # 2. Build context from memory
        context = await self.assembler.build_director_context(
            query=user_message,
            include_conversations=True,
            include_memories=True,
        )

        # 3. Call your LLM with memory-enriched prompt
        response = await self.call_llm(
            system_prompt=f"You are a helpful assistant.\n\n{context}",
            message=user_message,
        )

        # 4. Log the response
        await self.store.log_conversation("assistant", response)

        # 5. Let the agent decide what's worth remembering
        #    (or do this inside your LLM tool calls)
        if self.is_worth_remembering(response):
            await self.store.remember(response, category="lesson")

        return response
```

### Multi-Agent System (Director + Workers)

If you have an orchestrator that spawns worker agents for tasks, use all three tiers:

```python
class Orchestrator:
    """Your director/orchestrator agent."""

    async def dispatch_task(self, worker_name: str, task: str):
        # Build worker context — pulls from all 3 tiers automatically
        worker_ctx = await self.assembler.build_worker_context(
            worker_name=worker_name,
            task_description=task,
            extra_sections={
                "KANBAN": self.get_kanban_board(),
                "TASK": task,
            },
        )

        # Launch your worker with memory-enriched context
        result = await self.run_worker(
            system_prompt=f"You are worker '{worker_name}'.\n\n{worker_ctx}",
            task=task,
        )

        # Record the completion in the worker's resume
        await self.store.record_task_completion(
            worker_name=worker_name,
            task_id=result.id,
            description=task,
            outcome="completed",
            summary=result.summary,
        )

class Worker:
    """Your task/worker agent — gets Nous tools for its own memory."""

    def __init__(self, name: str, store: MemoryStore):
        self.name = name
        self.shell = store.get_shell(name)
        self.store = store

    async def remember_shared(self, content: str, category: str = "general"):
        """Store something other workers/director can see (Tier 2)."""
        await self.store.worker_remember(self.name, content, category)

    async def remember_private(self, content: str, importance: float = 0.5):
        """Store something only this worker can see (Tier 3)."""
        await self.shell.remember(content, importance=importance)

    async def get_instructions(self):
        """Read standing instructions that persist across tasks."""
        return await self.shell.get_instructions()
```

### Framework Integration Examples

**LangChain / LangGraph:**
```python
from langchain_core.messages import SystemMessage
from nous import MemoryStore, ContextAssembler

# In your graph setup
store = await MemoryStore.connect(...)
assembler = ContextAssembler(store)

async def inject_memory(state):
    """Node that enriches state with Nous memory before LLM call."""
    context = await assembler.build_director_context(
        query=state["messages"][-1].content,
    )
    state["messages"].insert(0, SystemMessage(content=context))
    return state
```

**Claude Agent SDK / Anthropic:**
```python
import anthropic
from nous import MemoryStore, ContextAssembler

store = await MemoryStore.connect(...)
assembler = ContextAssembler(store)

# Build memory-enriched system prompt
context = await assembler.build_director_context(query=user_input)
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    system=f"You are a persistent assistant.\n\n{context}",
    messages=[{"role": "user", "content": user_input}],
)
```

**OpenAI:**
```python
from openai import AsyncOpenAI
from nous import MemoryStore, ContextAssembler

store = await MemoryStore.connect(...)
assembler = ContextAssembler(store)

context = await assembler.build_director_context(query=user_input)
response = await client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": f"You are a persistent assistant.\n\n{context}"},
        {"role": "user", "content": user_input},
    ],
)
```

### Exposing Memory as Agent Tools

Most agent frameworks support tool/function calling. Expose Nous operations as tools so your agent can manage its own memory:

```python
# Define tools your agent can call
async def tool_remember(content: str, category: str = "general") -> str:
    """Store a memory for future recall."""
    mem = await store.remember(content, category=category)
    return f"Stored memory #{mem.id} in category '{category}'"

async def tool_recall(query: str, category: str | None = None) -> str:
    """Search memories by semantic similarity or keywords."""
    results = await store.recall(query, category=category, limit=5)
    if not results:
        return "No matching memories found."
    return "\n".join(f"- [{r.memory.category}] {r.memory.content}" for r in results)

async def tool_forget(memory_id: int) -> str:
    """Delete a specific memory."""
    deleted = await store.forget(memory_id)
    return f"Memory #{memory_id} {'deleted' if deleted else 'not found'}"

# Register these with your framework's tool system
# (LangChain @tool decorator, OpenAI function_call schema, Claude tool_use, etc.)
```

### Startup / Reboot Recovery

For agents that run as persistent services (systemd, Docker, etc.):

```python
async def agent_startup():
    store = await MemoryStore.connect(...)
    assembler = ContextAssembler(store)

    # Recover context from before the restart
    startup_ctx = await assembler.build_startup_context(
        conversation_limit=20,           # Last 20 conversation turns
        session_summaries=load_summaries(),  # Your shutdown digests
        reboot_prompt=load_reboot_prompt(),  # Why we restarted
    )

    # Agent wakes up knowing what happened before the restart
    return startup_ctx

async def agent_shutdown():
    # Generate a session summary before shutting down
    summary = await generate_summary(recent_conversations)
    save_summary(summary)

    await store.close()
```

### Key Integration Points

| Your System Has... | Wire Into Nous... |
|---------------------|-------------------|
| Conversation handler | `store.log_conversation()` on every turn |
| Agent system prompt | `assembler.build_*_context()` before each LLM call |
| Worker/task spawner | `assembler.build_worker_context()` at launch |
| Task completion hook | `store.record_task_completion()` for resumes |
| Tool/function system | Expose `remember`, `recall`, `forget` as callable tools |
| Startup routine | `assembler.build_startup_context()` for recovery |
| Shutdown routine | Generate and save a session summary |
| Worker identity | `store.get_shell(name)` — one SQLite DB per worker |

The pattern is always the same: **ingest as a side effect of doing work, retrieve before every LLM call, let the agent curate what's worth keeping.**

## Design Principles

1. **Memory is not a feature — it's woven into every interaction.** There's no separate "memory service." Ingestion happens as a side effect of doing work.

2. **The persistent agent IS the consolidation layer.** No background batch synthesis. The director curates in real-time with full conversational context.

3. **Graceful degradation.** Embedding service down? Keyword fallback. PostgreSQL down? Worker shells still work. Crash mid-task? Conversation replay on restart.

4. **Zero cloud dependency for memory.** Local embeddings, local databases, local compute. Your agent's memory doesn't phone home.

5. **Worker shells are portable identity.** The SQLite file IS the worker. Back it up, migrate it, inspect it independently.

## Requirements

- Python 3.11+
- PostgreSQL (for shared state)
- Ollama (optional, for semantic search — or use OpenAI, or keyword-only)

## License

MIT — see [LICENSE](LICENSE).
