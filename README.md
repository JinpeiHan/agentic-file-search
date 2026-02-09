# Agentic File Search

> **Based on**: [run-llama/fs-explorer](https://github.com/run-llama/fs-explorer) — The original CLI agent for filesystem exploration.

An AI-powered document search agent that explores files like a human would — scanning, reasoning, and following cross-references. Unlike traditional RAG systems that rely on pre-computed embeddings, this agent dynamically navigates documents to find answers.

## Why Agentic Search?

Traditional RAG (Retrieval-Augmented Generation) has limitations:
- **Chunks lose context** — Splitting documents destroys relationships between sections
- **Cross-references are invisible** — "See Exhibit B" means nothing to embeddings
- **Similarity ≠ Relevance** — Semantic matching misses logical connections

This system uses a **three-phase strategy**:
1. **Parallel Scan** — Preview all documents in a folder at once
2. **Deep Dive** — Full extraction on relevant documents only
3. **Backtrack** — Follow cross-references to previously skipped documents

## Watch the video
This video explains the architecture of the project and how to run it. 
[![Watch the demo on YouTube](https://img.youtube.com/vi/rMADSuus6jg/maxresdefault.jpg)](https://www.youtube.com/watch?v=rMADSuus6jg)

## Features

- 🔍 **6 Tools**: `scan_folder`, `preview_file`, `parse_file`, `read`, `grep`, `glob`
- 📄 **Document Support**: PDF, DOCX, PPTX, XLSX, HTML, Markdown (via Docling)
- 🤖 **Powered by**: Qwen3 32B running locally via Ollama (no cloud, no API keys)
- 💰 **100% Free**: Local inference on your own hardware
- 🌐 **Web UI**: Real-time WebSocket streaming interface
- 📊 **Citations**: Answers include source references
- ⚡ **Optimized**: Native Ollama API with thinking tokens disabled for 15x faster responses

## Quick Start

```bash
# Clone this branch
git clone -b feat/ollama-qwen3-native-api https://github.com/PromtEngineer/agentic-file-search.git
cd agentic-file-search

# Install dependencies (requires uv: https://docs.astral.sh/uv/)
uv sync

# Set up environment
cp .env.example .env
```

### Prerequisites

1. **Install Ollama**: https://ollama.com/download
2. **Pull the model**:
```bash
ollama pull qwen3:32b
```
3. **Verify Ollama is running**:
```bash
curl http://localhost:11434/api/tags
```

### Configuration

Edit `.env` in the project root:

```bash
OLLAMA_HOST=http://localhost:11434
MODEL_NAME=qwen3:32b
# OLLAMA_NUM_CTX=65536      # Context window (default: 65536)
# WORKFLOW_TIMEOUT=600       # Max query time in seconds (default: 600)
```

## Usage

### Web UI

```bash
# Start the server
uv run explore-ui

# Open http://localhost:8000 in your browser
```

The web UI provides:
- Folder browser to select target directory
- Real-time step-by-step execution log
- Final answer with citations
- Token usage statistics

### CLI

```bash
# Basic query
uv run explore --task "What is the purchase price?" --folder ./data/test_acquisition

# Multi-document query
uv run explore --task "What are all the financial terms including adjustments and escrow?" --folder ./data/large_acquisition
```

### Concurrent Benchmark

Measure your hardware's performance with parallel queries:

```bash
# Run 6 queries at concurrency levels 1, 2, 4, 8
uv run python -u benchmarks/concurrent_benchmark.py --concurrency 1,2,4,8 --warmup

# Quick test with 3 queries
uv run python -u benchmarks/concurrent_benchmark.py -c 1,4 -n 3
```

## Architecture

```
User Query
    ↓
┌─────────────────┐
│ Workflow Engine │ ←→ LlamaIndex Workflows (event-driven)
└────────┬────────┘
         ↓
┌─────────────────┐
│     Agent       │ ←→ Qwen3 32B via Ollama (structured JSON)
└────────┬────────┘
         ↓
┌─────────────────────────────────────────┐
│ scan_folder │ preview │ parse │ read │ grep │ glob │
└─────────────────────────────────────────┘
                    ↓
              Document Parser (Docling - local)
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed diagrams.

## Test Documents

The repo includes test document sets for evaluation:

- `data/test_acquisition/` — 10 interconnected legal documents
- `data/large_acquisition/` — 25 documents with extensive cross-references

Example queries:
```bash
# Simple (single doc)
uv run explore --task "Look in data/test_acquisition/. Who is the CTO?"

# Cross-reference required
uv run explore --task "Look in data/test_acquisition/. What is the adjusted purchase price?"

# Multi-document synthesis
uv run explore --task "Look in data/large_acquisition/. What happens to employees after the acquisition?"
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Qwen3 32B via Ollama (local) |
| Document Parsing | Docling (local, open-source) |
| Orchestration | LlamaIndex Workflows |
| CLI | Typer + Rich |
| Web Server | FastAPI + WebSocket |
| Package Manager | uv |

## Project Structure

```
src/fs_explorer/
├── agent.py      # Ollama client, token tracking
├── workflow.py   # LlamaIndex workflow engine
├── fs.py         # File tools: scan, parse, grep
├── models.py     # Pydantic models for actions
├── main.py       # CLI entry point
├── server.py     # FastAPI + WebSocket server
└── ui.html       # Single-file web interface
```

## Development

```bash
# Install dev dependencies
uv sync --group dev

# Run tests
make test

# Lint + format
make lint
make format

# Type check
make typecheck
```

## License

MIT

## Acknowledgments

- Original concept from [run-llama/fs-explorer](https://github.com/run-llama/fs-explorer)
- Document parsing by [Docling](https://github.com/DS4SD/docling)
- Local inference by [Ollama](https://ollama.com/) + [Qwen3](https://github.com/QwenLM/Qwen3)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=PromtEngineer/agentic-file-search&type=Date)](https://star-history.com/#PromtEngineer/agentic-file-search&Date)
