from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_jyutping_server_module():
    repo_root = Path(__file__).resolve().parents[1]
    server_path = repo_root / "mcp-servers" / "jyutping" / "server.py"
    spec = importlib.util.spec_from_file_location("jyutping_server", server_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_local_snapshot_loaded():
    module = _load_jyutping_server_module()
    local_map = getattr(module, "_LOCAL_POSTFIX_MAP", {})

    assert isinstance(local_map, dict)
    assert len(local_map) > 0
    assert "43" in local_map
    assert isinstance(local_map["43"], list)


def test_merge_words_keeps_order_and_dedup():
    module = _load_jyutping_server_module()
    merged = module._merge_words(["已經", "世界"], ["世界", "有興趣"])

    assert merged == ["已經", "世界", "有興趣"]
