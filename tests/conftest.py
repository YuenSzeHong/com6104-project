from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def doraemon_midi(repo_root: Path) -> Path:
    return repo_root / "test" / "midi" / "ドラえもんのうた.mid"


@pytest.fixture(scope="session")
def midi_analyzer_module(repo_root: Path):
    return load_module(
        "test_midi_analyzer_server",
        repo_root / "mcp-servers" / "midi-analyzer" / "server.py",
    )


@pytest.fixture(scope="session")
def melody_mapper_module(repo_root: Path):
    return load_module(
        "test_melody_mapper_server",
        repo_root / "mcp-servers" / "melody-mapper" / "server.py",
    )


@pytest.fixture(scope="session")
def jyutping_module(repo_root: Path):
    return load_module(
        "test_jyutping_server",
        repo_root / "mcp-servers" / "jyutping" / "server.py",
    )
