from __future__ import annotations

import json
from pathlib import Path

import httpx

SNAPSHOT_URL = "https://www.0243.hk/api/cls_postfix/m1.all/"
TARGET_PATH = Path(__file__).resolve().parents[1] / "mcp-servers" / "jyutping" / "data" / "postfix_m1_all.json"


def main() -> int:
    TARGET_PATH.parent.mkdir(parents=True, exist_ok=True)

    with httpx.Client(timeout=20.0) as client:
        response = client.get(SNAPSHOT_URL, headers={"Accept": "application/json"})
        response.raise_for_status()
        payload = response.json()

    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected payload type: {type(payload).__name__}")

    TARGET_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved {len(payload)} tone codes to {TARGET_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
