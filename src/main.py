#!/usr/bin/env python3
"""
Cantonese Lyrics Agent – Main Entry Point
==========================================
Run the multi-agent Cantonese lyric adaptation pipeline from the command line
or drop into an interactive session.

Usage examples
--------------
# Adapt a song into Cantonese from a MIDI file and source lyric/theme
python src/main.py --midi song.mid --text "青山依舊在，幾度夕陽紅"

# Adapt from a source lyric file without terminal-encoding issues
python src/main.py --midi song.mid --text-file test/lyrics/ドラえもんのうた.clean.txt

# Interactive mode (prompts for input each run)
python src/main.py --interactive

# Override the local model
python src/main.py --midi song.mid --text "..." --model qwen3.5:4b

# Verbose logging
python src/main.py --midi song.mid --text "..." -v

# Print the pipeline result as JSON
python src/main.py --midi song.mid --text "..." --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Ensure the project src directory is on the path when running directly
# ---------------------------------------------------------------------------

_SRC_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SRC_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))


# ---------------------------------------------------------------------------
# Logging setup (called before any agent imports so the root logger is
# configured before LangChain / httpx attach their own handlers)
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    # Quieten noisy third-party loggers unless in verbose mode
    if not verbose:
        for noisy in (
            "httpx",
            "httpcore",
            "openai",
            "langchain",
            "langsmith",
            "urllib3",
        ):
            logging.getLogger(noisy).setLevel(logging.WARNING)


logger = logging.getLogger("main")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cantonese-lyrics-agent",
        description=(
            "Cantonese Lyrics Agent – recreates Cantonese singable lyrics "
            "from a MIDI melody and a source lyric or theme."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Input ---
    input_group = parser.add_argument_group("Input")
    input_group.add_argument(
        "--midi", "-m",
        metavar="FILE",
        help="Path to the input MIDI file (.mid / .midi).",
    )
    input_group.add_argument(
        "--text", "-t",
        metavar="TEXT",
        help="Source lyric, translated lyric, or theme text used for Cantonese recreation.",
    )
    input_group.add_argument(
        "--text-file",
        metavar="FILE",
        help="Read the source lyric / theme text from a file instead of passing it on the command line.",
    )
    input_group.add_argument(
        "--text-encoding",
        metavar="NAME",
        help=(
            "Force the text-file encoding. Default: auto-detect from common encodings "
            "(utf-8, utf-8-sig, cp932, shift_jis, euc_jp, gbk)."
        ),
    )

    # --- Mode ---
    mode_group = parser.add_argument_group("Mode")
    mode_group.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode: prompt for MIDI path and text each session.",
    )

    # --- LLM ---
    llm_group = parser.add_argument_group("LLM")
    llm_group.add_argument(
        "--model",
        metavar="NAME",
        help=(
            "Local model name to use (overrides OLLAMA_MODEL / LMSTUDIO_MODEL env var). "
            "Defaults: LM Studio=qwen3.5-4b@q4_k_m, Ollama=qwen3.5:4b"
        ),
    )
    llm_group.add_argument(
        "--base-url",
        metavar="URL",
        help=(
            "Provider base URL (overrides OLLAMA_BASE_URL / LMSTUDIO_BASE_URL env var). "
            "Defaults: LM Studio=http://localhost:1234/v1, Ollama=http://localhost:11434"
        ),
    )
    llm_group.add_argument(
        "--temperature",
        type=float,
        metavar="FLOAT",
        help="LLM sampling temperature 0.0–1.0 (default: 0.7).",
    )

    # --- Workflow ---
    workflow_group = parser.add_argument_group("Workflow")
    workflow_group.add_argument(
        "--max-revisions",
        type=int,
        metavar="N",
        help="Maximum composer→validator revision loops (default: 3).",
    )
    workflow_group.add_argument(
        "--min-score",
        type=float,
        metavar="FLOAT",
        help="Minimum validator quality score to accept output, 0.0–1.0 (default: 0.75).",
    )
    workflow_group.add_argument(
        "--session-id",
        metavar="ID",
        help="Custom session ID for this run (used in memory logs).",
    )

    # --- Output ---
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--json",
        action="store_true",
        help="Print the full pipeline result as JSON instead of plain text.",
    )
    output_group.add_argument(
        "--output", "-o",
        metavar="FILE",
        help="Write the lyrics (or JSON result) to a file instead of stdout.",
    )

    # --- Misc ---
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="cantonese-lyrics-agent 0.1.0",
    )

    return parser


# ---------------------------------------------------------------------------
# Environment variable overrides from CLI flags
# ---------------------------------------------------------------------------

def _apply_cli_overrides(args: argparse.Namespace) -> None:
    """Push CLI flag values into environment variables so config.py picks them up."""
    if args.model:
        os.environ["OLLAMA_MODEL"] = args.model
    if args.base_url:
        os.environ["OLLAMA_BASE_URL"] = args.base_url
    if args.temperature is not None:
        os.environ["LLM_TEMPERATURE"] = str(args.temperature)
    if args.max_revisions is not None:
        os.environ["MAX_REVISION_LOOPS"] = str(args.max_revisions)
    if args.min_score is not None:
        os.environ["MIN_QUALITY_SCORE"] = str(args.min_score)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

_SEPARATOR = "─" * 60


def _print_banner() -> None:
    banner = r"""
  ██████╗ ████████╗ ██████╗  ███╗   ██╗███████╗███████╗███████╗
 ██╔════╝ ╚══██╔══╝██╔═══██╗ ████╗  ██║██╔════╝██╔════╝██╔════╝
 ██║         ██║   ██║   ██║ ██╔██╗ ██║█████╗  ███████╗█████╗
 ██║         ██║   ██║   ██║ ██║╚██╗██║██╔══╝  ╚════██║██╔══╝
 ╚██████╗    ██║   ╚██████╔╝ ██║ ╚████║███████╗███████║███████╗
  ╚═════╝    ╚═╝    ╚═════╝  ╚═╝  ╚═══╝╚══════╝╚══════╝╚══════╝
               Cantonese Lyrics Agent v0.1.0
    """
    print(banner, file=sys.stderr)


def _print_result(result: "PipelineResult", as_json: bool, output_file: str | None) -> None:  # type: ignore[name-defined]
    """Print or write the pipeline result."""
    if as_json:
        output_text = json.dumps(
            {
                "accepted":        result.accepted,
                "lyrics":          result.lyrics,
                "revision_count":  result.revision_count,
                "elapsed_seconds": round(result.elapsed_seconds, 2),
                "session_id":      result.session_id,
                "validator_scores": result.validator_scores,
                "midi_analysis":   result.midi_analysis,
                "jyutping_map":    result.jyutping_map,
                "draft_history":   result.draft_history,
                "error":           result.error,
            },
            ensure_ascii=False,
            indent=2,
        )
    else:
        status = "✓ Accepted" if result.accepted else "⚠  Best-effort (not fully accepted)"
        score_str = (
            f"{result.validator_scores[-1]:.3f}"
            if result.validator_scores
            else "N/A"
        )
        output_text = "\n".join([
            _SEPARATOR,
            f"  {status}",
            f"  Quality score  : {score_str}",
            f"  Revisions      : {result.revision_count}",
            f"  Elapsed        : {result.elapsed_seconds:.1f}s",
            f"  Session        : {result.session_id}",
            _SEPARATOR,
            "",
            result.lyrics or "(no lyrics generated)",
            "",
            _SEPARATOR,
        ])

    if output_file:
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_text, encoding="utf-8")
        print(f"Result written to: {out_path}", file=sys.stderr)
    else:
        print(output_text)


def _read_text_file(path: str, encoding: str | None = None) -> str:
    """
    Read a source lyric / theme file with optional explicit encoding.

    If encoding is not provided, try a small list of encodings commonly seen in
    the repo's foreign lyric fixtures and local Windows environments.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Text file not found: {file_path}")

    if encoding:
        return file_path.read_text(encoding=encoding)

    candidates = ("utf-8", "utf-8-sig", "cp932", "shift_jis", "euc_jp", "gbk")
    errors: list[str] = []
    for candidate in candidates:
        try:
            return file_path.read_text(encoding=candidate)
        except UnicodeDecodeError as exc:
            errors.append(f"{candidate}: {exc}")

    raise UnicodeDecodeError(
        "unknown",
        b"",
        0,
        0,
        "Could not decode text file. Tried: " + "; ".join(errors),
    )


# ---------------------------------------------------------------------------
# Single-run pipeline
# ---------------------------------------------------------------------------

async def run_pipeline(
    midi_path: str,
    reference_text: str,
    session_id: str | None = None,
    as_json: bool = False,
    output_file: str | None = None,
) -> int:
    """
    Instantiate the orchestrator, run one pipeline pass, and print the result.

    Returns an exit code: 0 = success (lyrics accepted or best-effort),
                          1 = error (pipeline crashed).
    """
    # Late import so env-var overrides are already applied
    from agent.orchestrator import AgentOrchestrator

    logger.info("Starting pipeline | midi=%s | session=%s", midi_path, session_id)

    try:
        async with AgentOrchestrator(session_id=session_id) as orch:
            result = await orch.run(
                midi_path=midi_path,
                reference_text=reference_text,
            )
    except Exception as exc:
        logger.exception("Pipeline crashed: %s", exc)
        print(f"\n[ERROR] Pipeline crashed: {exc}", file=sys.stderr)
        return 1

    _print_result(result, as_json=as_json, output_file=output_file)

    if result.error:
        logger.error("Pipeline finished with error: %s", result.error)
        return 1

    return 0


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

async def run_interactive(
    as_json: bool = False,
    output_file: str | None = None,
) -> int:
    """
    Run the agent in an interactive REPL-style loop.

    The user is prompted for a MIDI file path and source lyric/theme each
    iteration.  Type 'exit' or Ctrl-C to quit.
    """
    from agent.orchestrator import AgentOrchestrator

    _print_banner()
    print("Interactive mode – type 'exit' to quit.\n", file=sys.stderr)

    session_counter = 0

    while True:
        print(_SEPARATOR, file=sys.stderr)

        # --- MIDI path input ---
        try:
            midi_raw = input("MIDI file path (or 'exit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!", file=sys.stderr)
            return 0

        if midi_raw.lower() in {"exit", "quit", "q"}:
            print("Bye!", file=sys.stderr)
            return 0

        midi_path = Path(midi_raw)
        if not midi_path.exists():
            print(
                f"[WARNING] File not found: {midi_path} – proceeding anyway "
                "(the MIDI analyser will handle the error).",
                file=sys.stderr,
            )

        # --- Reference text input ---
        try:
            reference_text = input("Source lyric / theme text: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!", file=sys.stderr)
            return 0

        if not reference_text:
            print("[WARNING] No source lyric or theme provided – the composer will switch to original theme-based writing.",
                  file=sys.stderr)

        # --- Run pipeline ---
        session_counter += 1
        session_id = f"interactive-{session_counter}"

        print(f"\nRunning pipeline (session={session_id})…\n", file=sys.stderr)

        try:
            async with AgentOrchestrator(session_id=session_id) as orch:
                result = await orch.run(
                    midi_path=str(midi_path),
                    reference_text=reference_text,
                )
        except Exception as exc:
            logger.exception("Pipeline error: %s", exc)
            print(f"\n[ERROR] {exc}\n", file=sys.stderr)
            continue

        _print_result(result, as_json=as_json, output_file=output_file)

        # Ask if the user wants to continue
        try:
            again = input("\nRun again? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!", file=sys.stderr)
            return 0

        if again in {"n", "no"}:
            print("Bye!", file=sys.stderr)
            return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    # Configure logging first
    _setup_logging(verbose=args.verbose)

    # Apply CLI overrides to environment variables
    _apply_cli_overrides(args)

    # --- Validate arguments ---
    if not args.interactive:
        if not args.midi:
            parser.error(
                "Argument --midi is required in non-interactive mode. "
                "Use --interactive for a prompt-based session."
            )
        if args.text and args.text_file:
            parser.error("Use either --text or --text-file, not both.")
        if not args.text and not args.text_file:
            parser.error(
                "Argument --text or --text-file is required in non-interactive mode. "
                "Use --interactive for a prompt-based session."
            )

    reference_text = args.text
    if args.text_file:
        try:
            reference_text = _read_text_file(args.text_file, args.text_encoding).strip()
        except Exception as exc:  # noqa: BLE001
            parser.error(f"Failed to read --text-file: {exc}")

    # --- Dispatch ---
    try:
        if args.interactive:
            exit_code = asyncio.run(
                run_interactive(
                    as_json=args.json,
                    output_file=args.output,
                )
            )
        else:
            exit_code = asyncio.run(
                run_pipeline(
                    midi_path=args.midi,
                    reference_text=reference_text,
                    session_id=args.session_id,
                    as_json=args.json,
                    output_file=args.output,
                )
            )
    except KeyboardInterrupt:
        print("\n[Interrupted]", file=sys.stderr)
        exit_code = 130  # standard shell interrupt exit code

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
