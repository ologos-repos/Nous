#!/bin/bash
# setup-ollama.sh — Install Ollama and pull the embedding model for Nous.
#
# Usage:
#   bash scripts/setup-ollama.sh                          # default model
#   NOUS_EMBED_MODEL=mxbai-embed-large bash scripts/setup-ollama.sh  # custom model

set -euo pipefail

MODEL="${NOUS_EMBED_MODEL:-nomic-embed-text}"

echo "╔══════════════════════════════════════╗"
echo "║     Nous — Ollama Embedding Setup    ║"
echo "╚══════════════════════════════════════╝"
echo ""

# Check if Ollama is installed
if command -v ollama &> /dev/null; then
    echo "✓ Ollama is installed: $(ollama --version 2>/dev/null || echo 'unknown version')"
else
    echo "Ollama not found. Installing..."
    echo ""
    echo "Option 1 — Official installer (recommended):"
    echo "  curl -fsSL https://ollama.com/install.sh | sh"
    echo ""
    echo "Option 2 — Manual download:"
    echo "  https://ollama.com/download"
    echo ""
    read -p "Run the official installer now? [y/N] " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        curl -fsSL https://ollama.com/install.sh | sh
    else
        echo "Please install Ollama manually and re-run this script."
        exit 1
    fi
fi

# Check if Ollama service is running
echo ""
echo "Checking Ollama service..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✓ Ollama is running"
else
    echo "Starting Ollama..."
    ollama serve &
    sleep 2
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✓ Ollama started"
    else
        echo "⚠ Could not start Ollama. Start it manually with: ollama serve"
    fi
fi

# Pull the embedding model
echo ""
echo "Pulling embedding model '${MODEL}'..."
ollama pull "${MODEL}"

echo ""
echo "✓ Embedding setup complete."
echo "  Model: ${MODEL}"
echo "  URL:   http://localhost:11434"
echo ""
echo "Test it with:"
echo "  curl http://localhost:11434/api/embed -d '{\"model\": \"${MODEL}\", \"input\": \"hello world\"}'"
