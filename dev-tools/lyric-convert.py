from __future__ import annotations

import argparse
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag


def _extract_lyrics_nodes(html: str) -> list[Tag]:
    soup = BeautifulSoup(html, "html.parser")
    nodes = soup.select("p.lyrics-source")
    if not nodes:
        raise ValueError("No lyric nodes found. Expected <p class=\"lyrics-source ...\"> elements.")
    return nodes


def _ruby_to_kanji_yomi(node: Tag) -> str:
    parts: list[str] = []
    for child in node.children:
        if isinstance(child, NavigableString):
            text = str(child).strip()
            if text:
                parts.append(text)
            continue

        if not isinstance(child, Tag):
            continue

        if child.name == "ruby":
            rb = "".join(child.stripped_strings)
            rb_tag = child.find("rb")
            rt_tag = child.find("rt")
            if rb_tag and rt_tag:
                parts.append(f"{rb_tag.get_text(strip=True)}[{rt_tag.get_text(strip=True)}]")
            else:
                parts.append(rb)
            continue

        text = child.get_text(" ", strip=True)
        if text:
            parts.append(text)

    return "".join(parts).strip()


def _plain_text(node: Tag) -> str:
    return node.get_text(" ", strip=True)


def convert_lyrics(src: Path, output_format: str) -> list[str]:
    html = src.read_text(encoding="utf-8")
    nodes = _extract_lyrics_nodes(html)

    lines: list[str] = []
    for node in nodes:
        if output_format == "kanji-yomi":
            line = _ruby_to_kanji_yomi(node)
        elif output_format == "clean":
            line = _plain_text(node)
        else:
            raise ValueError(f"Unsupported format: {output_format}")

        if line:
            lines.append(line)

    return lines


def infer_output_path(src: Path, output_format: str) -> Path:
    suffix = ".kanji-yomi.txt" if output_format == "kanji-yomi" else ".clean.txt"
    name = src.name
    if name.endswith(".txt"):
        stem = name[:-4]
    else:
        stem = src.stem

    if src.parent.name == "raw":
        base_dir = src.parent.parent
    else:
        base_dir = src.parent

    return base_dir / f"{stem}{suffix}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert lyric page HTML into reusable text fixtures. "
            "'clean' strips page chrome and keeps only lyric lines; "
            "'kanji-yomi' keeps lyric lines plus ruby readings as kanji[yomi]."
        ),
    )
    parser.add_argument(
        "src",
        type=Path,
        help="Source HTML lyric file exported from a lyric page.",
    )
    parser.add_argument(
        "dst",
        type=Path,
        nargs="?",
        default=None,
        help=(
            "Optional destination path. Defaults to a sibling "
            "*.kanji-yomi.txt or *.clean.txt file."
        ),
    )
    parser.add_argument(
        "--format",
        choices=["kanji-yomi", "clean"],
        default="kanji-yomi",
        help=(
            "Output format. 'clean' keeps only lyric lines as plain text; "
            "'kanji-yomi' preserves ruby readings. Default: kanji-yomi."
        ),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dst = args.dst or infer_output_path(args.src, args.format)
    lines = convert_lyrics(args.src, args.format)

    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(dst)
    print(len(lines))


if __name__ == "__main__":
    main()
