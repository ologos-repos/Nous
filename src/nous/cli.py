"""
Nous CLI — setup and configuration helper.

Usage:
    nous-setup init            # Interactive setup — creates config, DB tables, shell dir
    nous-setup check           # Verify PostgreSQL connection and table status
    nous-setup migrate         # Run migrations (create/update tables)
    nous-setup shell <name>    # Initialize a new worker shell
    nous-setup stats           # Show memory statistics
    nous-setup config          # Print current configuration
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path


NOUS_CONFIG_FILE = "nous.toml"
NOUS_CONFIG_ENV = "NOUS_CONFIG"

DEFAULT_CONFIG = {
    "postgres_url": "postgresql://nous:nous@localhost:5432/nous",
    "shell_dir": "./shells",
    "embedding": {
        "backend": "ollama",
        "model": "nomic-embed-text",
        "base_url": "http://localhost:11434",
        "dimensions": 768,
    },
    "retrieval": {
        "similarity_threshold": 0.4,
        "max_results": 10,
        "keyword_fallback": True,
    },
    "retention": {
        "low_threshold": 0.3,
        "medium_threshold": 0.7,
        "low_retention_days": 30,
        "medium_retention_days": 90,
        "high_retention_days": 180,
        "preserve_categories": ["decision", "lesson"],
    },
    "context": {
        "max_context_chars": 50000,
        "conversation_window_hours": 2.0,
        "memory_search_limit": 15,
    },
}


def write_toml_config(path: Path, config: dict):
    """Write configuration as TOML (minimal, no toml dependency)."""
    lines = []
    for key, value in config.items():
        if isinstance(value, dict):
            lines.append(f"\n[{key}]")
            for k, v in value.items():
                if isinstance(v, str):
                    lines.append(f'{k} = "{v}"')
                elif isinstance(v, bool):
                    lines.append(f"{k} = {'true' if v else 'false'}")
                elif isinstance(v, list):
                    items = ", ".join(f'"{i}"' for i in v)
                    lines.append(f"{k} = [{items}]")
                else:
                    lines.append(f"{k} = {v}")
        else:
            if isinstance(value, str):
                lines.append(f'{key} = "{value}"')
            else:
                lines.append(f"{key} = {value}")

    path.write_text("\n".join(lines) + "\n")


def load_config(path: str | None = None) -> dict:
    """Load configuration from file or environment."""
    config_path = path or os.environ.get(NOUS_CONFIG_ENV) or NOUS_CONFIG_FILE

    if not Path(config_path).exists():
        return DEFAULT_CONFIG.copy()

    # Try tomllib (Python 3.11+)
    try:
        import tomllib
        with open(config_path, "rb") as f:
            return tomllib.load(f)
    except ImportError:
        pass

    # Fallback: try tomli
    try:
        import tomli
        with open(config_path, "rb") as f:
            return tomli.load(f)
    except ImportError:
        print(f"Warning: Cannot parse {config_path} — need Python 3.11+ or tomli package")
        return DEFAULT_CONFIG.copy()


async def cmd_init(args):
    """Interactive setup — creates config file, database tables, and shell directory."""
    config_path = Path(args.config or NOUS_CONFIG_FILE)

    print("╔══════════════════════════════════════╗")
    print("║       Nous Memory Architecture       ║")
    print("║            Initial Setup              ║")
    print("╚══════════════════════════════════════╝")
    print()

    config = DEFAULT_CONFIG.copy()

    # PostgreSQL URL
    print(f"PostgreSQL URL [{config['postgres_url']}]: ", end="")
    url = input().strip()
    if url:
        config["postgres_url"] = url

    # Shell directory
    print(f"Worker shell directory [{config['shell_dir']}]: ", end="")
    shell_dir = input().strip()
    if shell_dir:
        config["shell_dir"] = shell_dir

    # Embedding backend
    print(f"\nEmbedding backend (ollama/openai/none) [{config['embedding']['backend']}]: ", end="")
    backend = input().strip().lower()
    if backend in ("ollama", "openai", "none"):
        config["embedding"]["backend"] = backend

    if config["embedding"]["backend"] == "ollama":
        print(f"Ollama model [{config['embedding']['model']}]: ", end="")
        model = input().strip()
        if model:
            config["embedding"]["model"] = model
        print(f"Ollama URL [{config['embedding']['base_url']}]: ", end="")
        base_url = input().strip()
        if base_url:
            config["embedding"]["base_url"] = base_url

    # Write config
    write_toml_config(config_path, config)
    print(f"\n✓ Configuration written to {config_path}")

    # Create shell directory
    shell_path = Path(config["shell_dir"])
    shell_path.mkdir(parents=True, exist_ok=True)
    print(f"✓ Shell directory created: {shell_path.resolve()}")

    # Test PostgreSQL connection and run migrations
    print(f"\nConnecting to PostgreSQL...")
    try:
        from nous.store import MemoryStore
        store = await MemoryStore.connect(
            postgres_url=config["postgres_url"],
            shell_dir=config["shell_dir"],
            run_migrations=True,
        )
        print("✓ PostgreSQL connected")
        print("✓ Tables created (if they didn't exist)")
        await store.close()
    except Exception as e:
        print(f"✗ PostgreSQL connection failed: {e}")
        print("  You can run 'nous-setup migrate' later once PostgreSQL is available.")

    # Test embedding backend
    if config["embedding"]["backend"] == "ollama":
        print(f"\nChecking Ollama ({config['embedding']['base_url']})...")
        try:
            from nous.embeddings import OllamaEmbedder
            embedder = OllamaEmbedder(
                model=config["embedding"]["model"],
                base_url=config["embedding"]["base_url"],
            )
            available = await embedder.is_available()
            if available:
                print(f"✓ Ollama running, model '{config['embedding']['model']}' available")
            else:
                print(f"⚠ Ollama running but model '{config['embedding']['model']}' not found")
                print(f"  Pull it with: ollama pull {config['embedding']['model']}")
            await embedder.close()
        except Exception as e:
            print(f"⚠ Ollama not reachable: {e}")
            print("  Semantic search will fall back to keyword matching.")

    print("\n✓ Setup complete. You can now use Nous in your agent system.")
    print(f"  Config: {config_path.resolve()}")
    print(f"  Shells: {shell_path.resolve()}")


async def cmd_check(args):
    """Verify PostgreSQL connection and table status."""
    config = load_config(args.config)

    print(f"Checking PostgreSQL at {config['postgres_url']}...")
    try:
        import asyncpg
        conn = await asyncpg.connect(config["postgres_url"])

        tables = await conn.fetch("""
            SELECT tablename FROM pg_tables
            WHERE schemaname = 'public'
            ORDER BY tablename
        """)

        nous_tables = {
            "director_memory", "conversations", "worker_memory",
            "worker_history", "document_chunks", "document_embeddings",
        }

        found = {row["tablename"] for row in tables}
        present = nous_tables & found
        missing = nous_tables - found

        print(f"✓ Connected to PostgreSQL")
        if present:
            print(f"✓ Nous tables found: {', '.join(sorted(present))}")
        if missing:
            print(f"✗ Missing tables: {', '.join(sorted(missing))}")
            print("  Run 'nous-setup migrate' to create them.")

        # Count records
        for table in sorted(present):
            row = await conn.fetchrow(f"SELECT COUNT(*) as c FROM {table}")
            print(f"  {table}: {row['c']} records")

        await conn.close()
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        sys.exit(1)


async def cmd_migrate(args):
    """Run database migrations."""
    config = load_config(args.config)

    print(f"Running migrations on {config['postgres_url']}...")
    try:
        from nous.store import MemoryStore
        store = await MemoryStore.connect(
            postgres_url=config["postgres_url"],
            shell_dir=config.get("shell_dir", "./shells"),
            run_migrations=True,
        )
        print("✓ Migrations complete")
        await store.close()
    except Exception as e:
        print(f"✗ Migration failed: {e}")
        sys.exit(1)


async def cmd_shell(args):
    """Initialize a new worker shell."""
    config = load_config(args.config)
    name = args.name

    shell_dir = Path(config.get("shell_dir", "./shells"))
    shell_dir.mkdir(parents=True, exist_ok=True)
    db_path = shell_dir / f"{name}.db"

    if db_path.exists():
        print(f"Shell '{name}' already exists at {db_path}")
    else:
        from nous.store import ShellStore
        shell = ShellStore(name, str(db_path))
        await shell._ensure_init()
        await shell.close()
        print(f"✓ Shell '{name}' created at {db_path}")


async def cmd_stats(args):
    """Show memory statistics."""
    config = load_config(args.config)

    try:
        from nous.store import MemoryStore
        store = await MemoryStore.connect(
            postgres_url=config["postgres_url"],
            shell_dir=config.get("shell_dir", "./shells"),
            run_migrations=False,
        )

        # PostgreSQL stats
        async with store._pool.acquire() as conn:
            for table in ["director_memory", "conversations", "worker_memory", "worker_history",
                          "document_chunks", "document_embeddings"]:
                try:
                    row = await conn.fetchrow(f"SELECT COUNT(*) as c FROM {table}")
                    print(f"  {table}: {row['c']} records")
                except Exception:
                    print(f"  {table}: (table not found)")

        # Shell stats
        shells = await store.list_shells()
        if shells:
            print(f"\nWorker Shells ({len(shells)}):")
            for s in shells:
                print(
                    f"  {s.worker_name}: {s.memories_count} memories, "
                    f"{s.knowledge_count} knowledge, {s.instructions_count} instructions, "
                    f"{s.tasks_completed} tasks"
                )
        else:
            print("\nNo worker shells found.")

        await store.close()
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)


async def cmd_config(args):
    """Print current configuration."""
    config = load_config(args.config)
    # Print as formatted JSON (safe for display)
    sanitized = config.copy()
    # Mask password in postgres URL
    url = sanitized.get("postgres_url", "")
    if "@" in url:
        parts = url.split("@")
        sanitized["postgres_url"] = parts[0].rsplit(":", 1)[0] + ":***@" + parts[1]
    print(json.dumps(sanitized, indent=2))


def main():
    parser = argparse.ArgumentParser(
        prog="nous-setup",
        description="Nous Memory Architecture — setup and configuration",
    )
    parser.add_argument("--config", "-c", help="Path to nous.toml config file")

    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("init", help="Interactive setup")
    subparsers.add_parser("check", help="Verify PostgreSQL connection")
    subparsers.add_parser("migrate", help="Run database migrations")

    shell_parser = subparsers.add_parser("shell", help="Initialize a worker shell")
    shell_parser.add_argument("name", help="Worker name")

    subparsers.add_parser("stats", help="Show memory statistics")
    subparsers.add_parser("config", help="Print current configuration")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    commands = {
        "init": cmd_init,
        "check": cmd_check,
        "migrate": cmd_migrate,
        "shell": cmd_shell,
        "stats": cmd_stats,
        "config": cmd_config,
    }

    asyncio.run(commands[args.command](args))


if __name__ == "__main__":
    main()
