"""Persist analysis workbench UI settings between notebook re-runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_ANALYSIS_STATE_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "analysis_workbench.json"
)


def load_analysis_state(path: Path | str = DEFAULT_ANALYSIS_STATE_PATH) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def save_analysis_state(
    state: dict[str, Any],
    path: Path | str = DEFAULT_ANALYSIS_STATE_PATH,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))
    return path
