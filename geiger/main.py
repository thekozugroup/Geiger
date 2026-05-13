"""Geiger CLI - Tool-augmented LLM dataset generation."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

from geiger.config import Config
from geiger.agent.executor import AgentConfig, AgentExecutor
from geiger.types import Trace
from geiger.agent.review import Reviewer, ReviewResult, ReviewStatus, GradeBreakdown
from geiger.tools.parser import ToolParser
from geiger.dataset.generator import DatasetGenerator, DatasetConfig, DatasetStats, TraceStep
from geiger.types import DatasetTrace
from geiger.dataset.formatter import ShareGPTConversation

console = Console()


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="geiger",
        description="Tool-augmented LLM dataset generation",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", ""),
        help="OpenAI API key (or set OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--api-base",
        default="https://api.openai.com/v1",
        help="API base URL",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser(
        "generate",
        help="Run the full pipeline (parse tools, execute agents, review, generate dataset)",
    )
    generate_parser.add_argument(
        "--tools-md",
        required=True,
        help="Path to TOOLS.md file",
    )
    generate_parser.add_argument(
        "--tool-binaries",
        required=True,
        help="Path to tool binaries directory",
    )
    generate_parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for generated dataset",
    )
    generate_parser.add_argument(
        "--input-prompts",
        help="Path to JSON file containing prompts (optional)",
    )
    generate_parser.add_argument(
        "--model",
        default="gpt-4",
        help="Model to use for agents",
    )
    generate_parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Max tokens per response",
    )
    generate_parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation",
    )
    generate_parser.add_argument(
        "--max-agent-workers",
        type=int,
        default=5,
        help="Max concurrent agent workers",
    )
    generate_parser.add_argument(
        "--max-reviewer-workers",
        type=int,
        default=3,
        help="Max concurrent reviewer workers",
    )
    generate_parser.add_argument(
        "--min-grade-threshold",
        type=float,
        default=0.5,
        help="Minimum grade threshold for dataset inclusion",
    )

    review_parser = subparsers.add_parser(
        "review",
        help="Run the review step on existing traces",
    )
    review_parser.add_argument(
        "--traces-dir",
        required=True,
        help="Directory containing trace JSON files",
    )
    review_parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for reviewed traces",
    )
    review_parser.add_argument(
        "--min-grade-threshold",
        type=float,
        default=0.5,
        help="Minimum grade threshold",
    )
    review_parser.add_argument(
        "--max-reviewer-workers",
        type=int,
        default=3,
        help="Max concurrent reviewer workers",
    )

    stats_parser = subparsers.add_parser(
        "stats",
        help="Show dataset statistics",
    )
    stats_parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory containing dataset files",
    )

    return parser


@dataclass
class ReviewGrade:
    trace_id: str
    grade: float
    status: ReviewStatus
    comments: list[str]


class GeigerCLI:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.config = self._build_config()
        self.logger = logging.getLogger(__name__)

    def _build_config(self) -> Config:
        config = Config()
        config.api_key = self.args.api_key
        config.api_base_url = self.args.api_base
        config.model = getattr(self.args, "model", "gpt-4")
        config.max_tokens = getattr(self.args, "max_tokens", 2048)
        config.temperature = getattr(self.args, "temperature", 0.7)
        config.max_agent_workers = getattr(self.args, "max_agent_workers", 5)
        config.max_reviewer_workers = getattr(self.args, "max_reviewer_workers", 3)
        config.min_grade_threshold = getattr(self.args, "min_grade_threshold", 0.5)
        config.tools_md_path = getattr(self.args, "tools_md", "")
        config.tool_binaries_path = getattr(self.args, "tool_binaries", "")
        config.output_dir = getattr(self.args, "output_dir", "")
        return config

    async def run_generate(self) -> int:
        self.logger.info("Starting generate pipeline")
        try:
            self.config.validate()
        except ValueError as e:
            console.print(f"[red]Configuration error: {e}[/red]")
            return 1

        tools_md_path = Path(self.config.tools_md_path)
        if not tools_md_path.exists():
            console.print(f"[red]Tools file not found: {tools_md_path}[/red]")
            return 1

        tool_binaries_path = Path(self.config.tool_binaries_path)
        if not tool_binaries_path.exists():
            console.print(f"[red]Tool binaries directory not found: {tool_binaries_path}[/red]")
            return 1

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        console.print(Panel.fit(
            "[bold cyan]Geiger Dataset Generation[/bold cyan]",
            subtitle="Tool-augmented LLM dataset pipeline",
        ))

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            parse_task = progress.add_task("[cyan]Parsing TOOLS.md...", total=None)
            try:
                parser = ToolParser(tools_md_path)
                tool_definitions = parser.parse()
                progress.update(parse_task, completed=True)
            except Exception as e:
                progress.update(parse_task, completed=True)
                console.print(f"[red]Failed to parse tools: {e}[/red]")
                return 1

            if not tool_definitions:
                console.print("[yellow]No tools found in TOOLS.md[/yellow]")
                return 1

            console.print(f"[green]Parsed {len(tool_definitions)} tools[/green]")

            prompts = self._load_prompts()
            total_prompts = len(prompts)

            exec_task = progress.add_task(
                f"[cyan]Executing agents ({total_prompts} prompts)...",
                total=total_prompts,
            )

            agent_config = AgentConfig(
                base_url=self.config.api_base_url,
                api_key=self.config.api_key,
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )

            executor = AgentExecutor(
                config=agent_config,
                max_workers=self.config.max_agent_workers,
            )

            tool_executor = self._build_tool_executor(tool_binaries_path)

            agent_tool_defs = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": {"properties": {}, "type": "object"},
                    },
                }
                for t in tool_definitions
            ]

            traces: list[Trace] = []
            async for trace in executor.execute([
                (prompt, None, agent_tool_defs, tool_executor)
                for prompt in prompts
            ]):
                traces.append(trace)
                progress.update(exec_task, advance=1)

            console.print(f"[green]Generated {len(traces)} traces[/green]")

            traces_file = output_dir / "traces.json"
            with open(traces_file, "w") as f:
                json.dump([t.to_dict() for t in traces], f, indent=2)
            console.print(f"[dim]Traces saved to {traces_file}[/dim]")

            review_task = progress.add_task(
                f"[cyan]Reviewing traces ({len(traces)} traces)...",
                total=len(traces),
            )

            reviewer = Reviewer(
                base_url=self.config.api_base_url,
                api_key=self.config.api_key,
                min_score=self.config.min_grade_threshold,
                max_workers=self.config.max_reviewer_workers,
            )
            review_results: list[ReviewGrade] = []

            try:
                semaphore = asyncio.Semaphore(self.config.max_reviewer_workers)

                async def review_with_semaphore(trace):
                    async with semaphore:
                        result = await reviewer.review_result({
                            "success": True,
                            "result": trace.to_dict(),
                            "error": getattr(trace, "error", None),
                        })
                        return trace.session_id, result

                raw_results = await asyncio.gather(*[review_with_semaphore(t) for t in traces])
                review_results: list[ReviewGrade] = []
                for trace_id, raw in raw_results:
                    grade = ReviewGrade(
                        trace_id=trace_id,
                        grade=raw.grade,
                        status=raw.status,
                        comments=[raw.reasoning],
                    )
                    review_results.append(grade)
                progress.update(review_task, completed=len(traces))
            finally:
                await reviewer.close()

            approved = sum(1 for r in review_results if r.status == ReviewStatus.APPROVED)
            console.print(
                f"[green]Review complete: {approved}/{len(review_results)} approved[/green]"
            )

            dataset_task = progress.add_task(
                "[cyan]Generating ShareGPT dataset...",
                total=None,
            )

            dataset_traces: list[DatasetTrace] = []
            for trace, review in zip(traces, review_results):
                if review.grade >= self.config.min_grade_threshold:
                    steps = []
                    for msg in trace.messages:
                        role = msg.get("role", "user")
                        content = msg.get("content", "")
                        if role == "system":
                            continue
                        steps.append(TraceStep(
                            role="human" if role == "user" else role,
                            content=content,
                        ))
                    dataset_traces.append(DatasetTrace(
                        messages=steps,
                        grade=review.grade,
                        tool_definitions=agent_tool_defs,
                    ))

            gen_config = DatasetConfig(
                output_dir=output_dir,
                min_grade_threshold=self.config.min_grade_threshold,
            )
            generator = DatasetGenerator(gen_config)
            stats = await generator.generate(dataset_traces)

            progress.update(dataset_task, completed=True)

        self._print_stats(stats)
        console.print("[bold green]Dataset generation complete![/bold green]")
        return 0

    async def run_review(self) -> int:
        console.print(Panel.fit(
            "[bold cyan]Geiger Review[/bold cyan]",
            subtitle="Review existing traces",
        ))

        traces_dir = Path(self.args.traces_dir)
        if not traces_dir.exists():
            console.print(f"[red]Traces directory not found: {traces_dir}[/red]")
            return 1

        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        trace_files = list(traces_dir.glob("*.json"))
        if not trace_files:
            console.print(f"[yellow]No trace files found in {traces_dir}[/yellow]")
            return 0

        console.print(f"[cyan]Found {len(trace_files)} trace files[/cyan]")

        reviewer = Reviewer(
                base_url=self.args.api_base or "https://api.openai.com/v1",
                api_key=self.args.api_key or "",
                min_score=self.args.min_grade_threshold,
                max_workers=self.args.max_reviewer_workers,
            )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"[cyan]Reviewing traces...",
                total=len(trace_files),
            )

            semaphore = asyncio.Semaphore(self.args.max_reviewer_workers)

            async def review_with_semaphore(trace_file):
                async with semaphore:
                    try:
                        with open(trace_file) as f:
                            trace_data = json.load(f)
                        result = await reviewer.review_result({
                            "success": True,
                            "result": trace_data,
                            "error": trace_data.get("error"),
                        })
                        return {
                            "trace_id": trace_file.stem,
                            "grade": result.grade,
                            "status": result.status.value,
                            "reasoning": result.reasoning,
                        }
                    except Exception as e:
                        self.logger.error(f"Failed to review {trace_file}: {e}")
                        return {
                            "trace_id": trace_file.stem,
                            "grade": 0.0,
                            "status": ReviewStatus.REJECTED.value,
                            "reasoning": f"Error: {e}",
                        }

            results: list[dict[str, Any]] = await asyncio.gather(*[review_with_semaphore(tf) for tf in trace_files])
            progress.update(task, completed=len(trace_files))

        approved = sum(1 for r in results if r["status"] == ReviewStatus.APPROVED.value)
        needs_revision = sum(1 for r in results if r["status"] == ReviewStatus.NEEDS_REVISION.value)
        rejected = sum(1 for r in results if r["status"] == ReviewStatus.REJECTED.value)

        table = Table(title="Review Results")
        table.add_column("Status", style="bold")
        table.add_column("Count")
        table.add_row("[green]Approved", str(approved))
        table.add_row("[yellow]Needs Revision", str(needs_revision))
        table.add_row("[red]Rejected", str(rejected))
        console.print(table)

        output_file = output_dir / "review_results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        console.print(f"[dim]Results saved to {output_file}[/dim]")

        return 0

    async def run_stats(self) -> int:
        console.print(Panel.fit(
            "[bold cyan]Geiger Stats[/bold cyan]",
            subtitle="Dataset statistics",
        ))

        output_dir = Path(self.args.output_dir)
        if not output_dir.exists():
            console.print(f"[red]Output directory not found: {output_dir}[/red]")
            return 1

        manifest_file = output_dir / "manifest.json"
        if manifest_file.exists():
            with open(manifest_file) as f:
                stats_data = json.load(f)
            self._print_stats_dict(stats_data)
            return 0

        data_files = list(output_dir.glob("data_*.json"))
        if not data_files:
            console.print(f"[yellow]No dataset files found in {output_dir}[/yellow]")
            return 0

        total_traces = len(data_files)
        grades: list[float] = []
        total_tools = 0

        for data_file in data_files:
            try:
                with open(data_file) as f:
                    data = json.load(f)
                if "conversations" in data:
                    total_tools = len(data.get("system", "").split("tools")) - 1
            except Exception:
                continue

        stats = DatasetStats(
            total_traces=total_traces,
            filtered_traces=0,
            min_grade=min(grades) if grades else 0.0,
            max_grade=max(grades) if grades else 0.0,
            avg_grade=sum(grades) / len(grades) if grades else 0.0,
            tool_count=total_tools,
        )

        self._print_stats(stats)
        return 0

    def _load_prompts(self) -> list[str]:
        input_file = getattr(self.args, "input_prompts", None)
        if input_file:
            input_path = Path(input_file)
            if input_path.exists():
                with open(input_path) as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return [item["prompt"] if isinstance(item, dict) else str(item) for item in data]
                    elif isinstance(data, dict) and "prompts" in data:
                        return data["prompts"]
        return [
            "Search for all Python files in the current directory.",
            "List the contents of the home directory.",
            "Find all text files modified in the last 7 days.",
        ]

    def _build_tool_executor(self, tools_bin_path: Path) -> dict[str, Any]:
        return {
            "file_search": self._exec_file_search,
            "image_process": self._exec_image_process,
            "data_query": self._exec_data_query,
        }

    async def _exec_file_search(self, **kwargs) -> str:
        import glob
        import os
        from pathlib import Path

        pattern = kwargs.get("pattern", "*")
        directory = kwargs.get("directory", ".")
        recursive = kwargs.get("recursive", True)
        max_results = kwargs.get("max_results", 100)

        root = Path(directory)
        if recursive:
            files = list(root.rglob(pattern))
        else:
            files = list(root.glob(pattern))

        files = files[:max_results]
        result = {
            "files": [
                {
                    "path": str(f),
                    "name": f.name,
                    "size": f.stat().st_size if f.is_file() else 0,
                    "modified": f.stat().st_mtime if f.is_file() else 0,
                }
                for f in files if f.exists()
            ],
            "total_count": len(files),
        }
        return json.dumps(result)

    async def _exec_image_process(self, **kwargs) -> str:
        return json.dumps({
            "success": True,
            "output_path": kwargs.get("output_path", ""),
            "dimensions": {"width": 100, "height": 100},
            "file_size": 0,
        })

    async def _exec_data_query(self, **kwargs) -> str:
        return json.dumps({
            "rows": [],
            "row_count": 0,
            "execution_time_ms": 0,
            "columns": [],
        })

    def _print_stats(self, stats: DatasetStats) -> None:
        table = Table(title="Dataset Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Traces", str(stats.total_traces))
        table.add_row("Filtered Traces", str(stats.filtered_traces))
        table.add_row("Min Grade", f"{stats.min_grade:.4f}")
        table.add_row("Max Grade", f"{stats.max_grade:.4f}")
        table.add_row("Avg Grade", f"{stats.avg_grade:.4f}")
        table.add_row("Tool Count", str(stats.tool_count))

        console.print(table)

    def _print_stats_dict(self, stats: dict[str, Any]) -> None:
        table = Table(title="Dataset Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        for key, value in stats.items():
            formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
            table.add_row(key.replace("_", " ").title(), formatted_value)

        console.print(table)


async def async_main(args: argparse.Namespace) -> int:
    setup_logging(args.verbose)

    cli = GeigerCLI(args)

    if args.command == "generate":
        return await cli.run_generate()
    elif args.command == "review":
        return await cli.run_review()
    elif args.command == "stats":
        return await cli.run_stats()
    else:
        console.print(f"[red]Unknown command: {args.command}[/red]")
        return 1


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()

    try:
        return asyncio.run(async_main(args))
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        return 130
    except Exception as e:
        console.print(f"[red]Fatal error: {e}[/red]")
        if getattr(args, "verbose", False):
            import traceback
            console.print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
