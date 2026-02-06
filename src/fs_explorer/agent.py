"""
FsExplorer Agent for filesystem exploration using Qwen3 via Ollama.

This module contains the agent that interacts with the Qwen3 AI model
(via Ollama's native API) to make decisions about filesystem exploration actions.
"""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Any, cast

from dotenv import load_dotenv
import httpx

# Load .env file from project root
_env_path = Path(__file__).parent.parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

from .models import Action, ActionType, StopAction, ToolCallAction, Tools
from .fs import (
    read_file,
    grep_file_content,
    glob_paths,
    scan_folder,
    preview_file,
    parse_file,
)


# =============================================================================
# Token Usage Tracking
# =============================================================================

@dataclass
class TokenUsage:
    """
    Track token usage across the session.

    Maintains running totals of API calls and token counts.
    Since we're using a local model, no cost estimates are provided.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    api_calls: int = 0

    # Track content sizes
    tool_result_chars: int = 0
    documents_parsed: int = 0
    documents_scanned: int = 0

    def add_api_call(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Record token usage from an API call."""
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += prompt_tokens + completion_tokens
        self.api_calls += 1

    def add_tool_result(self, result: str, tool_name: str) -> None:
        """Record metrics from a tool execution."""
        self.tool_result_chars += len(result)
        if tool_name == "parse_file":
            self.documents_parsed += 1
        elif tool_name == "scan_folder":
            # Count documents in scan result by counting document markers
            self.documents_scanned += result.count("│ [")
        elif tool_name == "preview_file":
            self.documents_parsed += 1

    def summary(self) -> str:
        """Generate a formatted summary of token usage."""
        return f"""
═══════════════════════════════════════════════════════════════
                      TOKEN USAGE SUMMARY
═══════════════════════════════════════════════════════════════
  API Calls:           {self.api_calls}
  Prompt Tokens:       {self.prompt_tokens:,}
  Completion Tokens:   {self.completion_tokens:,}
  Total Tokens:        {self.total_tokens:,}
───────────────────────────────────────────────────────────────
  Documents Scanned:   {self.documents_scanned}
  Documents Parsed:    {self.documents_parsed}
  Tool Result Chars:   {self.tool_result_chars:,}
───────────────────────────────────────────────────────────────
  Model: Qwen3 32B (Local via Ollama)
  Cost:  $0.00 (Local inference)
═══════════════════════════════════════════════════════════════
"""


# =============================================================================
# Tool Registry
# =============================================================================

TOOLS: dict[Tools, Callable[..., str]] = {
    "read": read_file,
    "grep": grep_file_content,
    "glob": glob_paths,
    "scan_folder": scan_folder,
    "preview_file": preview_file,
    "parse_file": parse_file,
}


# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT = """
You are FsExplorer, an AI agent that answers user questions by reading documents in a folder.

## Tools

- `scan_folder(directory)` — Scan all documents in a folder (gives previews of each)
- `parse_file(file_path)` — Read complete content of a document
- `preview_file(file_path)` — Quick preview of a document
- `read(file_path)` — Read a plain text file
- `grep(file_path, pattern)` — Search for a pattern in a file
- `glob(directory, pattern)` — Find files matching a glob pattern

## IMPORTANT RULES

1. **ONLY use file paths returned by tool results.** After calling scan_folder, the result will contain exact file paths. Copy-paste those paths exactly into subsequent tool calls. NEVER make up a file path.
2. **Never ask the user about files.** If you need a file, just call parse_file or preview_file with the path from scan results.
3. **Cite your sources** in the final answer using `[Source: <filename>, <section>]` format.

## Workflow

1. Start with `scan_folder` on the target directory to see all documents
2. Read the scan results carefully — note the EXACT file paths listed
3. Call `parse_file` on documents relevant to the user's question (use the exact paths from step 2)
4. If a document references another document, parse that one too
5. When you have enough information, stop and provide your final answer with citations

## Response Format

Respond with ONLY valid JSON (no other text):

{"action": {"tool_name": "...", "tool_input": [{"parameter_name": "...", "parameter_value": "..."}]}, "reason": "..."}

OR to give the final answer:

{"action": {"final_result": "Your answer with [Source: filename, section] citations..."}, "reason": "..."}

OR to navigate into a subdirectory:

{"action": {"directory": "/path/to/dir"}, "reason": "..."}
"""


# =============================================================================
# Agent Implementation
# =============================================================================

class FsExplorerAgent:
    """
    AI agent for exploring filesystems using Qwen3 via Ollama.

    The agent maintains a conversation history with the LLM and uses
    structured JSON output to make decisions about which actions to take.

    Attributes:
        token_usage: Tracks API call statistics.
    """

    def __init__(self, api_key: str | None = None) -> None:
        """
        Initialize the agent with Ollama configuration.

        Args:
            api_key: Not used for Ollama (kept for API compatibility).

        Note:
            Uses OLLAMA_HOST and MODEL_NAME environment variables,
            defaulting to http://localhost:11434 and qwen3:32b.
        """
        self._ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self._model_name = os.getenv("MODEL_NAME", "qwen3:32b")
        self._num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "65536"))

        # Use native Ollama API via httpx for think=false support.
        # The OpenAI-compatible endpoint ignores think=false, causing
        # Qwen3 to spend ~15x longer generating unused thinking tokens.
        self._client = httpx.AsyncClient(
            base_url=self._ollama_host,
            timeout=httpx.Timeout(300.0, connect=10.0),
        )
        self._chat_history: list[dict[str, str]] = []
        self.token_usage = TokenUsage()

    def configure_task(self, task: str) -> None:
        """
        Add a task message to the conversation history.

        Args:
            task: The task or context to add to the conversation.
        """
        self._chat_history.append({
            "role": "user",
            "content": task
        })

    async def take_action(self) -> tuple[Action, ActionType] | None:
        """
        Request the next action from the AI model.

        Uses Ollama's native /api/chat endpoint with think=false to disable
        Qwen3's thinking mode, which otherwise adds ~15x latency per call.

        Returns:
            A tuple of (Action, ActionType) if successful, None otherwise.
        """
        # Build messages list with system prompt
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *self._chat_history
        ]

        # Call Ollama's native API with think=false and JSON format
        response = await self._client.post(
            "/api/chat",
            json={
                "model": self._model_name,
                "messages": messages,
                "format": "json",
                "think": False,
                "stream": False,
                "options": {"num_ctx": self._num_ctx},
            },
        )
        response.raise_for_status()
        data = response.json()

        # Track token usage from response
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)
        self.token_usage.add_api_call(prompt_tokens, completion_tokens)

        content = data.get("message", {}).get("content", "")
        if content:
            # Extract JSON from the response (should already be clean
            # with think=false, but handle edge cases)
            json_str = content.strip()
            if not json_str.startswith("{"):
                json_start = json_str.find("{")
                json_end = json_str.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = json_str[json_start:json_end]

            # Store compact JSON in history
            self._chat_history.append({
                "role": "assistant",
                "content": json_str
            })

            # Parse the JSON response
            try:
                action = Action.model_validate_json(json_str)
                if action.to_action_type() == "toolcall":
                    toolcall = cast(ToolCallAction, action.action)
                    self.call_tool(
                        tool_name=toolcall.tool_name,
                        tool_input=toolcall.to_fn_args(),
                    )
                return action, action.to_action_type()
            except Exception:
                # Recover from common LLM format mistakes.
                # E.g. model puts final answer in "reason" with
                # {"action":{"tool_name":"final_result",...}} instead
                # of {"action":{"final_result":"..."}}.
                recovered = self._try_recover_action(json_str)
                if recovered:
                    return recovered
                print(f"Failed to parse response: {json_str[:500]}...")

        return None

    @staticmethod
    def _try_recover_action(json_str: str) -> tuple[Action, ActionType] | None:
        """Attempt to recover a valid Action from malformed LLM JSON.

        Common mistake: the model puts the final answer in the "reason"
        field and sets tool_name to "final_result".
        """
        try:
            raw = json.loads(json_str)
        except json.JSONDecodeError:
            return None

        action_data = raw.get("action", {})
        reason = raw.get("reason", "")

        # Case: {"action": {"tool_name": "final_result", ...}, "reason": "actual answer"}
        if action_data.get("tool_name") == "final_result" and reason:
            action = Action(
                action=StopAction(final_result=reason),
                reason="Recovered from malformed response",
            )
            return action, "stop"

        return None

    def call_tool(self, tool_name: Tools, tool_input: dict[str, Any]) -> None:
        """
        Execute a tool and add the result to the conversation history.

        Args:
            tool_name: Name of the tool to execute.
            tool_input: Dictionary of arguments to pass to the tool.
        """
        try:
            result = TOOLS[tool_name](**tool_input)
        except Exception as e:
            result = (
                f"An error occurred while calling tool {tool_name} "
                f"with {tool_input}: {e}"
            )

        # Track tool result sizes
        self.token_usage.add_tool_result(result, tool_name)

        self._chat_history.append({
            "role": "user",
            "content": f"Tool result for {tool_name}:\n\n{result}"
        })

    def reset(self) -> None:
        """Reset the agent's conversation history and token tracking."""
        self._chat_history.clear()
        self.token_usage = TokenUsage()
