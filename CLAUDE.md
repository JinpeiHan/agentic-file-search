# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Agentic File Search — an AI agent that explores filesystems to answer questions about documents. It uses Qwen3 32B via local Ollama and follows a three-phase strategy: scan folders in parallel, deep-dive into relevant documents, then backtrack to resolve cross-references. Supports PDF, DOCX, PPTX, XLSX, HTML, and Markdown via Docling (all local, no cloud).

## Commands

```bash
make test          # uv run pytest tests
make lint          # uv run pre-commit run -a
make format        # uv run ruff format
make format-check  # uv run ruff format --check
make typecheck     # uv run ty check src/fs_explorer/
make build         # uv build
make all           # test + lint + format + typecheck
```

Run a single test file: `uv run pytest tests/test_models.py -v`

Run the CLI: `uv run explore --task "What is the purchase price?" --folder ./data/`

Run the web UI: `uv run explore-ui` (serves on `0.0.0.0:8000`)

## Architecture

The system has four layers connected by an event-driven workflow:

**Entry points** → `main.py` (Typer CLI) and `server.py` (FastAPI + WebSocket at `/ws/explore`) both drive the same workflow. The web UI is a single-file SPA (`ui.html`) served by FastAPI.

**Workflow orchestration** → `workflow.py` uses LlamaIndex Workflows (not traditional async chains). Events (`InputEvent`, `ToolCallEvent`, `GoDeeperEvent`, `AskHumanEvent`, `ExplorationEndEvent`) flow through `FsExplorerWorkflow` step methods. Both CLI and web UI consume these events via `handler.stream_events()`.

**Agent** → `agent.py` contains `FsExplorerAgent` which maintains a chat history list with Ollama (AsyncOpenAI client pointing at Ollama's `/v1` endpoint). Each iteration: send history → receive JSON action → execute tool → append result as user message. The `SYSTEM_PROMPT` defines the three-phase exploration strategy and citation format. `TokenUsage` dataclass tracks all usage metrics.

**Action model** → `models.py` defines a discriminated union: `Action.action` is one of `ToolCallAction`, `StopAction`, `GoDeeperAction`, or `AskHumanAction`. The agent always returns JSON matching this schema. `Action.to_action_type()` returns the discriminator.

**Tools** → `fs.py` provides 6 tools registered in the `TOOLS` dict: `scan_folder` (parallel preview with ThreadPoolExecutor), `preview_file`, `parse_file` (full extraction via Docling), `read`, `grep`, `glob`. Documents are cached in-memory keyed by `path:mtime`.

## Environment

Requires Ollama running locally with the model pulled. Configuration via `.env` (see `.env.example`):
- `OLLAMA_HOST` — default `http://localhost:11434`
- `MODEL_NAME` — default `qwen3:32b`

## CI

GitHub Actions runs on PRs: `test.yaml` (Python 3.10–3.13), `lint.yaml`, `typecheck.yaml`, `build.yaml`. All use `astral-sh/setup-uv@v6`.

## Conventions

- Python ≥3.10 with full type hints; type checking via `ty` (not mypy)
- Formatting and linting via Ruff + pre-commit
- Tool functions in `fs.py` return strings (error messages as strings, not exceptions)
- Async throughout: agent uses `AsyncOpenAI`, workflow steps are `async def`
- Workflow timeout is 300 seconds
