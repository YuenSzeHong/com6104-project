#!/usr/bin/env python3
"""
Cantonese Lyrics Agent – Main Entry Point
==========================================
Start the Gradio web UI for the Cantonese lyric adaptation workflow.

Usage examples
--------------
# Launch the web GUI
python src/main.py --gui

# GUI on a custom port
python src/main.py --gui --port 8080

# Override the local model before launching GUI
python src/main.py --gui --model qwen3.5:4b

# Verbose logging
python src/main.py --gui -v
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.orchestrator import PipelineResult


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


def _create_demo():
    """Build and return the Gradio Blocks demo."""
    from gui.app import create_ui

    return create_ui()


# Expose a top-level demo object so the Gradio CLI can discover it.
demo = _create_demo()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cantonese-lyrics-agent",
        description="Cantonese Lyrics Agent – Gradio launcher.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Mode ---
    mode_group = parser.add_argument_group("Mode")
    mode_group.add_argument(
        "--gui",
        action="store_true",
        help="Launch the Gradio web interface (default behavior).",
    )
    mode_group.add_argument(
        "--port",
        type=int,
        default=7860,
        metavar="PORT",
        help="Port for the Gradio web interface (default: 7860).",
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


def _print_result(
    result: "PipelineResult", as_json: bool, output_file: str | None
) -> None:
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
                reference_text_kind="theme",
            )
    except Exception as exc:
        logger.exception("Pipeline crashed: %s", exc)
        print(f"\n[ERROR] Pipeline crashed: {exc}", file=sys.stderr)
        return 1

    _print_result(result, as_json=as_json, output_file=output_file)

    # 写入歌词文件并输出
    if result.lyrics:
        _write_lyrics_file(midi_path, result.lyrics)
        # 直接输出歌词到 stderr，确保能看到
        if not as_json:
            print("\n" + "=" * 60, file=sys.stderr)
            print(result.lyrics, file=sys.stderr)
            print("=" * 60 + "\n", file=sys.stderr)

    if result.error:
        logger.error("Pipeline finished with error: %s", result.error)
        return 1

    return 0


def _write_lyrics_file(midi_path: str, lyrics: str) -> None:
    """
    将生成的歌词写入文件。

    文件名格式：<midi 文件名>.lyrics.txt
    保存在 MIDI 文件同一目录下。
    """
    from pathlib import Path

    midi_file = Path(midi_path)
    lyrics_file = midi_file.with_suffix(".lyrics.txt")

    try:
        lyrics_file.write_text(lyrics, encoding="utf-8")
        logger.info("歌词已写入：%s", lyrics_file)
    except Exception as exc:
        logger.warning("无法写入歌词文件：%s", exc)


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
                    reference_text_kind="theme",
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

    # CLI is GUI-first. Launch the shared demo object directly.
    demo.launch(
        server_port=args.port,
        share=False,
        show_error=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
