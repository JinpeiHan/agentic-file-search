"""
Concurrent Benchmark for FsExplorer on DGX Spark.

Runs multiple queries simultaneously against Ollama to measure:
- Per-query latency at different concurrency levels
- Aggregate throughput (queries/minute)
- Token usage and efficiency
- Ollama's actual parallelism under load

Usage:
    uv run python benchmarks/concurrent_benchmark.py [--concurrency 1,2,4,8] [--data-dir data/test_acquisition]
"""

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Suppress verbose Docling/httpx/OCR logging
logging.getLogger("docling").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("rapidocr").setLevel(logging.WARNING)
logging.getLogger("RapidOCR").setLevel(logging.WARNING)
# Catch-all for any other noisy loggers
for name in ("docling.document_converter", "docling.pipeline", "PIL"):
    logging.getLogger(name).setLevel(logging.WARNING)

# Add project root to path so we can import fs_explorer
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

# Load .env
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

from fs_explorer.agent import FsExplorerAgent, SYSTEM_PROMPT
from fs_explorer.fs import describe_dir_content, clear_document_cache
from fs_explorer.models import ToolCallAction, GoDeeperAction, StopAction


# =============================================================================
# Test Queries
# =============================================================================

# Queries ordered by complexity (simple → complex)
TEST_QUERIES = [
    # Simple (single document)
    "What is the total purchase price for the StartupXYZ acquisition?",
    "When was the Non-Disclosure Agreement between TechCorp and StartupXYZ signed?",
    "How many patents does StartupXYZ own?",
    # Medium (2-3 documents with cross-references)
    "What are the key risks identified in this acquisition and what mitigation measures were put in place?",
    "The original purchase price was $45M. Were there any adjustments? What is the final amount?",
    "Which customers required consent for the acquisition, and was consent obtained from all of them?",
    # Complex (multiple documents, deep cross-references)
    "Give me a complete picture of StartupXYZ's intellectual property - what do they own, is it properly certified, and are there any pending matters or risks?",
    "What did the due diligence process uncover, and how were any issues resolved before closing?",
    "Create a timeline of this acquisition from NDA signing to closing. What are the key milestones and their status?",
    "Is this acquisition ready to close? What items are complete and what's still pending?",
    # Adversarial (tests cross-reference discovery)
    "The Acquisition Agreement mentions 'Exhibit A - Financial Terms'. What are the detailed financial terms?",
    "How does the Legal Opinion Letter relate to other documents in this acquisition?",
    "Is there anything about MegaCorp in these documents? Why are they important to this deal?",
]


# =============================================================================
# Query Result Tracking
# =============================================================================

@dataclass
class QueryResult:
    """Result from a single query execution."""

    query_id: int
    query: str
    success: bool = False
    error: str | None = None
    final_answer: str | None = None
    steps: int = 0
    api_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    documents_parsed: int = 0
    documents_scanned: int = 0
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def duration_seconds(self) -> float:
        return self.end_time - self.start_time

    @property
    def duration_str(self) -> str:
        d = self.duration_seconds
        if d >= 60:
            return f"{int(d // 60)}m{int(d % 60)}s"
        return f"{d:.1f}s"


@dataclass
class BenchmarkResult:
    """Aggregate results for a concurrency level."""

    concurrency: int
    query_results: list[QueryResult] = field(default_factory=list)
    wall_clock_seconds: float = 0.0

    @property
    def total_queries(self) -> int:
        return len(self.query_results)

    @property
    def successful_queries(self) -> int:
        return sum(1 for r in self.query_results if r.success)

    @property
    def failed_queries(self) -> int:
        return self.total_queries - self.successful_queries

    @property
    def avg_latency(self) -> float:
        successful = [r for r in self.query_results if r.success]
        if not successful:
            return 0.0
        return sum(r.duration_seconds for r in successful) / len(successful)

    @property
    def min_latency(self) -> float:
        successful = [r for r in self.query_results if r.success]
        if not successful:
            return 0.0
        return min(r.duration_seconds for r in successful)

    @property
    def max_latency(self) -> float:
        successful = [r for r in self.query_results if r.success]
        if not successful:
            return 0.0
        return max(r.duration_seconds for r in successful)

    @property
    def queries_per_minute(self) -> float:
        if self.wall_clock_seconds == 0:
            return 0.0
        return (self.successful_queries / self.wall_clock_seconds) * 60

    @property
    def total_tokens(self) -> int:
        return sum(r.total_tokens for r in self.query_results)

    @property
    def tokens_per_second(self) -> float:
        if self.wall_clock_seconds == 0:
            return 0.0
        return self.total_tokens / self.wall_clock_seconds

    @property
    def speedup_vs_serial(self) -> float:
        """Estimated speedup: sum of individual durations / wall clock."""
        serial_time = sum(r.duration_seconds for r in self.query_results if r.success)
        if self.wall_clock_seconds == 0:
            return 0.0
        return serial_time / self.wall_clock_seconds


# =============================================================================
# Agent Runner (bypasses workflow singleton)
# =============================================================================

MAX_STEPS = 20


async def run_single_query(
    query_id: int,
    query: str,
    data_dir: str,
    semaphore: asyncio.Semaphore | None = None,
) -> QueryResult:
    """
    Run a single query using an independent agent instance.

    This bypasses the workflow's singleton agent to allow true concurrency.
    Each query gets its own FsExplorerAgent with its own chat history
    and httpx client.
    """
    result = QueryResult(query_id=query_id, query=query)
    result.start_time = time.time()

    # Each query gets its own agent instance
    agent = FsExplorerAgent()

    abs_dir = str(Path(data_dir).resolve())
    dir_desc = describe_dir_content(abs_dir)

    agent.configure_task(
        f"Given that the target directory ('{abs_dir}') looks like this:\n\n"
        f"```text\n{dir_desc}\n```\n\n"
        f"And that the user is giving you this task: '{query}', "
        f"what action should you take first?"
    )

    try:
        for step in range(MAX_STEPS):
            # Optionally limit concurrency at the Ollama call level
            if semaphore:
                async with semaphore:
                    action_result = await agent.take_action()
            else:
                action_result = await agent.take_action()

            if action_result is None:
                result.error = f"No action at step {step + 1}"
                break

            action, action_type = action_result
            result.steps = step + 1

            if action_type == "stop":
                stop = action.action
                if isinstance(stop, StopAction):
                    result.final_answer = stop.final_result
                    result.success = True
                break

            elif action_type == "godeeper":
                godeeper = action.action
                if isinstance(godeeper, GoDeeperAction):
                    new_dir_desc = describe_dir_content(godeeper.directory)
                    agent.configure_task(
                        f"Given that the current directory ('{godeeper.directory}') "
                        f"looks like this:\n\n```text\n{new_dir_desc}\n```\n\n"
                        f"And that the user is giving you this task: '{query}', "
                        f"what action should you take next?"
                    )

            elif action_type == "toolcall":
                # Tool already executed in take_action()
                agent.configure_task(
                    "Given the result from the tool call you just performed, "
                    "what action should you take next?"
                )

            elif action_type == "askhuman":
                # Auto-answer in benchmark mode
                agent.configure_task(
                    "The user says: 'Please answer based on the documents available.' "
                    f"Proceed with your exploration based on the original task: {query}"
                )

        else:
            # Hit MAX_STEPS without stopping
            result.error = f"Hit max steps ({MAX_STEPS})"

    except Exception as e:
        result.error = str(e)

    result.end_time = time.time()

    # Capture token usage
    usage = agent.token_usage
    result.api_calls = usage.api_calls
    result.prompt_tokens = usage.prompt_tokens
    result.completion_tokens = usage.completion_tokens
    result.total_tokens = usage.total_tokens
    result.documents_parsed = usage.documents_parsed
    result.documents_scanned = usage.documents_scanned

    # Clean up the httpx client
    await agent._client.aclose()

    return result


# =============================================================================
# Benchmark Runner
# =============================================================================

async def run_benchmark_batch(
    queries: list[tuple[int, str]],
    data_dir: str,
    concurrency: int,
) -> BenchmarkResult:
    """
    Run a batch of queries at a given concurrency level.

    Args:
        queries: List of (query_id, query_text) tuples.
        data_dir: Path to the data directory.
        concurrency: Max concurrent queries.
    """
    bench = BenchmarkResult(concurrency=concurrency)

    # Use a semaphore to limit actual concurrency if desired
    # (Ollama itself may also limit via OLLAMA_NUM_PARALLEL)
    sem = asyncio.Semaphore(concurrency) if concurrency > 0 else None

    async def throttled_query(qid: int, q: str) -> QueryResult:
        if sem:
            async with sem:
                return await run_single_query(qid, q, data_dir)
        return await run_single_query(qid, q, data_dir)

    start = time.time()
    tasks = [throttled_query(qid, q) for qid, q in queries]
    bench.query_results = await asyncio.gather(*tasks)
    bench.wall_clock_seconds = time.time() - start

    return bench


# =============================================================================
# Reporting
# =============================================================================

def format_duration(seconds: float) -> str:
    if seconds >= 3600:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h}h{m}m{s}s"
    if seconds >= 60:
        return f"{int(seconds // 60)}m{int(seconds % 60)}s"
    return f"{seconds:.1f}s"


def print_report(results: list[BenchmarkResult], data_dir: str) -> None:
    """Print a comprehensive benchmark report."""
    print()
    print("=" * 75)
    print("  DGX SPARK CONCURRENT BENCHMARK REPORT")
    print("  FsExplorer + Qwen3:32b via Ollama")
    print("=" * 75)
    print(f"  Data directory: {data_dir}")
    print(f"  Model: {os.getenv('MODEL_NAME', 'qwen3:32b')}")
    print(f"  Ollama host: {os.getenv('OLLAMA_HOST', 'http://localhost:11434')}")
    print(f"  Context window: {os.getenv('OLLAMA_NUM_CTX', '65536')}")
    print(f"  Max steps per query: {MAX_STEPS}")
    print("=" * 75)
    print()

    # Summary table
    print("┌" + "─" * 73 + "┐")
    print(f"│ {'Concurrency':>12} │ {'Queries':>7} │ {'OK':>4} │ {'Fail':>4} │ "
          f"{'Wall Clock':>10} │ {'Avg Latency':>11} │ {'Q/min':>6} │ {'Speedup':>7} │")
    print("├" + "─" * 73 + "┤")

    for bench in results:
        print(f"│ {bench.concurrency:>12} │ {bench.total_queries:>7} │ "
              f"{bench.successful_queries:>4} │ {bench.failed_queries:>4} │ "
              f"{format_duration(bench.wall_clock_seconds):>10} │ "
              f"{format_duration(bench.avg_latency):>11} │ "
              f"{bench.queries_per_minute:>6.1f} │ "
              f"{bench.speedup_vs_serial:>6.2f}x │")

    print("└" + "─" * 73 + "┘")
    print()

    # Token usage summary
    print("┌" + "─" * 73 + "┐")
    print(f"│ {'Concurrency':>12} │ {'Total Tokens':>12} │ {'Tok/sec':>8} │ "
          f"{'Docs Scanned':>12} │ {'Docs Parsed':>11} │ {'API Calls':>9} │")
    print("├" + "─" * 73 + "┤")

    for bench in results:
        total_scanned = sum(r.documents_scanned for r in bench.query_results)
        total_parsed = sum(r.documents_parsed for r in bench.query_results)
        total_api = sum(r.api_calls for r in bench.query_results)
        print(f"│ {bench.concurrency:>12} │ {bench.total_tokens:>12,} │ "
              f"{bench.tokens_per_second:>8.0f} │ "
              f"{total_scanned:>12} │ {total_parsed:>11} │ {total_api:>9} │")

    print("└" + "─" * 73 + "┘")
    print()

    # Latency distribution per concurrency level
    for bench in results:
        print(f"─── Concurrency = {bench.concurrency} "
              f"(wall clock: {format_duration(bench.wall_clock_seconds)}) ───")
        print()

        for r in sorted(bench.query_results, key=lambda x: x.query_id):
            status = "OK" if r.success else "FAIL"
            q_short = r.query[:60] + "..." if len(r.query) > 60 else r.query
            print(f"  Q{r.query_id:>2} [{status:>4}] {r.duration_str:>8} | "
                  f"{r.steps:>2} steps | {r.total_tokens:>6,} tok | {q_short}")
            if r.error:
                print(f"       ERROR: {r.error[:70]}")

        print()

    # Scaling analysis
    if len(results) >= 2:
        baseline = results[0]
        print("─── Scaling Analysis ───")
        print()
        print(f"  Baseline (concurrency={baseline.concurrency}): "
              f"{format_duration(baseline.wall_clock_seconds)}")
        print()
        for bench in results[1:]:
            ideal_speedup = bench.concurrency / baseline.concurrency
            actual_speedup = bench.speedup_vs_serial
            efficiency = (actual_speedup / ideal_speedup * 100) if ideal_speedup > 0 else 0
            print(f"  Concurrency {bench.concurrency:>3}: "
                  f"actual={actual_speedup:.2f}x, "
                  f"ideal={ideal_speedup:.1f}x, "
                  f"efficiency={efficiency:.0f}%")
        print()

    print("=" * 75)
    print("  NOTES:")
    print("  - Ollama's OLLAMA_NUM_PARALLEL controls actual GPU parallelism")
    print("  - Document parsing (Docling) is CPU-bound, cached after first parse")
    print("  - Speedup >1x indicates Ollama is processing requests in parallel")
    print("  - Efficiency <100% is expected due to GPU memory contention")
    print("=" * 75)
    print()


def save_results_json(results: list[BenchmarkResult], output_path: str) -> None:
    """Save detailed results as JSON for later analysis."""
    data = []
    for bench in results:
        bench_data = {
            "concurrency": bench.concurrency,
            "wall_clock_seconds": bench.wall_clock_seconds,
            "total_queries": bench.total_queries,
            "successful_queries": bench.successful_queries,
            "avg_latency": bench.avg_latency,
            "min_latency": bench.min_latency,
            "max_latency": bench.max_latency,
            "queries_per_minute": bench.queries_per_minute,
            "total_tokens": bench.total_tokens,
            "tokens_per_second": bench.tokens_per_second,
            "speedup_vs_serial": bench.speedup_vs_serial,
            "queries": [
                {
                    "query_id": r.query_id,
                    "query": r.query,
                    "success": r.success,
                    "error": r.error,
                    "steps": r.steps,
                    "api_calls": r.api_calls,
                    "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.completion_tokens,
                    "total_tokens": r.total_tokens,
                    "documents_parsed": r.documents_parsed,
                    "documents_scanned": r.documents_scanned,
                    "duration_seconds": r.duration_seconds,
                    "final_answer_length": len(r.final_answer) if r.final_answer else 0,
                }
                for r in bench.query_results
            ],
        }
        data.append(bench_data)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Detailed results saved to: {output_path}")


# =============================================================================
# Main
# =============================================================================

async def main():
    import argparse

    parser = argparse.ArgumentParser(description="DGX Spark Concurrent Benchmark")
    parser.add_argument(
        "--concurrency", "-c",
        default="1,2,4",
        help="Comma-separated concurrency levels to test (default: 1,2,4)",
    )
    parser.add_argument(
        "--data-dir", "-d",
        default="data/test_acquisition",
        help="Data directory with documents (default: data/test_acquisition)",
    )
    parser.add_argument(
        "--num-queries", "-n",
        type=int,
        default=0,
        help="Number of queries to run (0 = all 13, default: 0)",
    )
    parser.add_argument(
        "--output", "-o",
        default="benchmarks/results.json",
        help="Output JSON file for detailed results",
    )
    parser.add_argument(
        "--warmup", "-w",
        action="store_true",
        help="Run a single warmup query first (populates document cache)",
    )

    args = parser.parse_args()

    concurrency_levels = [int(x) for x in args.concurrency.split(",")]
    data_dir = args.data_dir
    num_queries = args.num_queries if args.num_queries > 0 else len(TEST_QUERIES)
    queries = list(enumerate(TEST_QUERIES[:num_queries], start=1))

    print(f"\nBenchmark Configuration:")
    print(f"  Data directory:      {data_dir}")
    print(f"  Queries:             {len(queries)}")
    print(f"  Concurrency levels:  {concurrency_levels}")
    print(f"  Model:               {os.getenv('MODEL_NAME', 'qwen3:32b')}")
    print(f"  Ollama host:         {os.getenv('OLLAMA_HOST', 'http://localhost:11434')}")
    print()

    # Optional warmup: parse all documents once to fill cache
    if args.warmup:
        print("Running warmup query to populate document cache...")
        warmup_result = await run_single_query(0, TEST_QUERIES[0], data_dir)
        print(f"  Warmup complete: {warmup_result.duration_str} "
              f"({'OK' if warmup_result.success else 'FAIL'})")
        print()

    all_results: list[BenchmarkResult] = []

    for level in concurrency_levels:
        print(f"{'=' * 50}")
        print(f"  Running {len(queries)} queries at concurrency = {level}")
        print(f"{'=' * 50}")

        bench = await run_benchmark_batch(queries, data_dir, level)
        all_results.append(bench)

        print(f"  Completed in {format_duration(bench.wall_clock_seconds)} "
              f"({bench.successful_queries}/{bench.total_queries} succeeded)")
        print()

        # Clear document cache between concurrency levels for fair comparison
        # (first level populates it, subsequent levels benefit from cache)
        # Actually, let's keep the cache — it simulates a real workload
        # where documents are parsed once and reused.

    # Print report
    print_report(all_results, data_dir)

    # Save JSON results
    save_results_json(all_results, args.output)


if __name__ == "__main__":
    asyncio.run(main())
