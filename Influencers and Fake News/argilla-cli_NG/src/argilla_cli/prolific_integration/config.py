from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ProlificStudyConfig:
    """Configuration for preparing an Argilla dataset for a Prolific study.

    This config is intentionally minimal and maps directly to what you need
    to create an Argilla dataset + prepare a credentials CSV for Prolific.
    """

    dataset_csv: Path
    text_column: str
    id_column: str
    dataset_name: str
    workspace: str = "main"

    # Optional quality-of-life fields
    completion_code: Optional[str] = None
    credentials_out: Optional[Path] = None


def load_prolific_config(path: Path) -> ProlificStudyConfig:
    """Load study configuration from a YAML file."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    missing = [
        k
        for k in ("dataset_csv", "text_column", "id_column", "dataset_name")
        if k not in data or data[k] in (None, "")
    ]
    if missing:
        raise ValueError(
            "Missing required keys in Prolific config YAML: " + ", ".join(missing)
        )

    credentials_out = data.get("credentials_out")
    return ProlificStudyConfig(
        dataset_csv=Path(data["dataset_csv"]),
        text_column=str(data["text_column"]),
        id_column=str(data["id_column"]),
        dataset_name=str(data["dataset_name"]),
        workspace=str(data.get("workspace", "main")),
        completion_code=data.get("completion_code"),
        credentials_out=Path(credentials_out) if credentials_out else None,
    )