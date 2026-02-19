from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING, Iterable, cast
import json
import os
import subprocess
import csv
import warnings
import random

import typer
import yaml
import argilla as rg
import heapq

from argilla_cli.clients.argilla_client import get_client
from argilla_cli.settings import load_settings
from argilla_cli.errors import (
    exit_with_error,
    NotFoundError,
    ValidationError,
)
from argilla_cli.globals import state
from argilla_cli.io_utils import emit_json, print_error, print_ok, print_table

app = typer.Typer(help="Manage Argilla datasets")

if TYPE_CHECKING:  # only for static type checkers; no runtime dependency
    from argilla import Dataset as ArgillaDataset


# ---------------------------------------------------------------------
# Dataset resolution + export helpers (download)
# ---------------------------------------------------------------------

def _resolve_dataset(
    client: Any, dataset_name: str, workspace: Optional[str]
) -> tuple["ArgillaDataset", str]:
    """Find a dataset by name, optionally constrained to a workspace.

    Returns (dataset, workspace_name) or raises NotFoundError/ValidationError.
    """
    workspaces = list(client.workspaces)  # type: ignore[attr-defined]

    if workspace:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=(r"Dataset with name .* not found in workspace .*"),
                    category=UserWarning,
                )
                ds = client.datasets(dataset_name, workspace=workspace)
        except Exception as e:
            msg = f"dataset {dataset_name!r} not found in workspace {workspace!r}"
            raise NotFoundError(msg) from e
        if ds is None:
            raise NotFoundError(
                f"dataset {dataset_name!r} not found in workspace {workspace!r}"
            )
        return ds, workspace

    matches: list[tuple[Any, str]] = []
    for ws in workspaces:
        ws_name = getattr(ws, "name", "")
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=(r"Dataset with name .* not found in workspace .*"),
                    category=UserWarning,
                )
                ds = client.datasets(dataset_name, workspace=ws_name)
        except Exception:
            ds = None
        if ds is not None:
            matches.append((ds, ws_name))

    if not matches:
        raise NotFoundError(f"dataset {dataset_name!r} not found")
    if len(matches) > 1:
        ws_list = ", ".join(ws_name for _, ws_name in matches)
        raise ValidationError(
            f"dataset {dataset_name!r} found in multiple workspaces: {ws_list}. "
            "Use --workspace to disambiguate."
        )
    return matches[0]


def _load_mapping_file(path: Path) -> Dict[str, str]:
    """Load JSON mapping: output_field -> JMESPath expression."""
    if not path.exists():
        raise ValidationError(f"mapping file not found: {path}")
    if path.suffix.lower() != ".json":
        raise ValidationError(
            f"unsupported mapping format for {path}. only .json is accepted"
        )

    text = path.read_text(encoding="utf-8")
    try:
        data: Any = json.loads(text)
    except Exception as e:
        raise ValidationError(f"failed to parse mapping file: {e}") from e

    if not isinstance(data, dict):
        raise ValidationError("mapping file must be a JSON object")
    for k, v in data.items():
        if not isinstance(v, str):
            raise ValidationError(
                f"mapping for key {k!r} must be a string JMESPath expression"
            )
    return data  # type: ignore[return-value]


def _compile_mapping(mapping: Dict[str, str]) -> Dict[str, Any]:
    try:
        import jmespath  # type: ignore
    except Exception as e:
        raise ValidationError(
            "jmespath is required for --map. Install with: pip install jmespath"
        ) from e

    compiled: Dict[str, Any] = {}
    for key, expr in mapping.items():
        try:
            compiled[key] = jmespath.compile(expr)
        except Exception as e:
            raise ValidationError(f"invalid JMESPath for {key!r}: {expr} ({e})")
    return compiled


def _iter_record_dicts(dataset: Any) -> Iterable[Dict[str, Any]]:
    """Yield records as plain dicts using Argilla's public API."""
    records_obj = getattr(dataset, "records", None)
    if records_obj is None:
        raise ValidationError("dataset does not expose records")

    to_list = getattr(records_obj, "to_list", None)
    if not callable(to_list):
        raise ValidationError("records.to_list(...) not available; upgrade argilla")

    try:
        rows = to_list(flatten=False)  # type: ignore[call-arg]
    except Exception as e:
        raise ValidationError(f"failed to obtain records via to_list: {e}") from e

    if not isinstance(rows, list):
        rows = list(cast(Iterable[Any], rows))

    for idx, rec in enumerate(rows):
        if isinstance(rec, dict):
            yield rec
            continue

        md = getattr(rec, "model_dump", None)
        if callable(md):
            try:
                yield md()  # type: ignore[misc]
                continue
            except Exception as e:
                raise ValidationError(
                    f"failed to convert record {idx} via model_dump: {e}"
                ) from e

        dct = getattr(rec, "dict", None)
        if callable(dct):
            try:
                yield dct()  # type: ignore[misc]
                continue
            except Exception as e:
                raise ValidationError(
                    f"failed to convert record {idx} via dict(): {e}"
                ) from e

        raise ValidationError(
            f"unexpected record type at index {idx}: {type(rec).__name__}; "
            "expected dict or pydantic model"
        )


def _apply_completed_filter(
    records: Iterable[Dict[str, Any]], completed_only: bool
) -> Iterable[Dict[str, Any]]:
    if not completed_only:
        return records
    return (r for r in records if r.get("status") == "completed")


def _transform_record(
    record: Dict[str, Any],
    compiled_mapping: Dict[str, Any],
    list_policy: str,
    list_sep: str,
) -> Dict[str, Any]:
    """Apply compiled mapping to a single record and produce a flat dict."""
    out: Dict[str, Any] = {}
    for key, expr in compiled_mapping.items():
        val = expr.search(record)
        if isinstance(val, list):
            if list_policy == "join":
                out[key] = list_sep.join("" if v is None else str(v) for v in val)
            elif list_policy == "first":
                out[key] = val[0] if val else None
            else:
                raise ValidationError(
                    f"mapping for {key!r} produced a list; use --list-policy join|first"
                )
        elif isinstance(val, dict):
            out[key] = json.dumps(val, ensure_ascii=False)
        else:
            out[key] = val
    return out


# ---------------------------------------------------------------------
# File selection + preview helpers (import-upload)
# ---------------------------------------------------------------------

def _choose_file_with_macos_dialog() -> Optional[Path]:
    """Open a native macOS file chooser and return the selected path."""
    script = 'POSIX path of (choose file with prompt "Select your dataset file")'
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            check=True,
        )
        path_str = result.stdout.strip()
        if not path_str:
            return None
        return Path(path_str)
    except Exception:
        return None


def _choose_file_from_current_directory() -> Path:
    """Scan the current directory for candidate files and let the user pick one."""
    candidates = [
        Path(f)
        for f in os.listdir(".")
        if f.endswith((".jsonl", ".json", ".csv", ".xlsx", ".xls"))
    ]
    if not candidates:
        raise ValidationError(
            "No candidate files (.jsonl, .json, .csv, .xlsx, .xls) found in the current directory."
        )

    typer.echo("Files found in current directory:\n")
    for i, path in enumerate(candidates, start=1):
        typer.echo(f"[{i}] {path}")

    choice = typer.prompt("Select file", default="1")
    try:
        idx = int(choice) - 1
        return candidates[idx]
    except (ValueError, IndexError):
        raise ValidationError("Invalid choice; aborting.")


def _preview_file(path: Path, max_lines: int = 3) -> None:
    """Preview the given file depending on its type."""
    suffix = path.suffix.lower()
    typer.echo("\nPreview:")

    if suffix in {".csv", ".json", ".jsonl", ".txt"}:
        typer.echo(f"(first {max_lines} lines of text)")
        try:
            with path.open("r", encoding="utf-8") as f:
                for _ in range(max_lines):
                    line = f.readline()
                    if not line:
                        break
                    typer.echo(line.rstrip())
        except UnicodeDecodeError:
            typer.echo("[File is not valid UTF-8 text; skipping preview]")
        return

    if suffix in {".xlsx", ".xls"}:
        typer.echo(f"(first {max_lines} rows of Excel sheet)")
        try:
            import pandas as pd  # type: ignore
        except Exception:
            typer.echo(
                "[Previewing Excel files requires pandas + openpyxl. Install with: pip install pandas openpyxl]"
            )
            return

        try:
            df_pd = pd.read_excel(path, nrows=max_lines)
        except Exception as e:
            typer.echo(f"[Failed to read Excel file for preview: {e}]")
            return

        typer.echo(df_pd.to_string(index=False))
        return

    typer.echo(f"(unknown extension {suffix!r}; trying text preview)")
    try:
        with path.open("r", encoding="utf-8") as f:
            for _ in range(max_lines):
                line = f.readline()
                if not line:
                    break
                typer.echo(line.rstrip())
    except UnicodeDecodeError:
        typer.echo("[Binary or non-text file; skipping preview]")


# ---------------------------------------------------------------------
# Load local files into list[dict]
# ---------------------------------------------------------------------

def _load_csv_file(path: Path) -> List[Dict[str, Any]]:
    try:
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(dict(row))
        return rows
    except Exception as e:
        raise ValidationError(f"Failed to read CSV file {path}: {e}") from e


def _load_json_file(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(data, list):
        return [dict(x) if isinstance(x, dict) else {"value": x} for x in data]

    if isinstance(data, dict):
        if "records" in data and isinstance(data["records"], list):
            return [
                dict(x) if isinstance(x, dict) else {"value": x}
                for x in data["records"]
            ]
        return [data]

    return [{"value": data}]


def _load_jsonl_file(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append(obj if isinstance(obj, dict) else {"value": obj})
    return rows


def _load_excel_file(path: Path) -> List[Dict[str, Any]]:
    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        raise ValidationError(
            "Excel import requires pandas + openpyxl. Install with: pip install pandas openpyxl"
        ) from e

    try:
        df_pd = pd.read_excel(path)
    except Exception as e:
        raise ValidationError(f"Failed to read Excel file {path}: {e}") from e

    return df_pd.to_dict(orient="records")


def _load_records(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _load_csv_file(path)
    if suffix == ".json":
        return _load_json_file(path)
    if suffix == ".jsonl":
        return _load_jsonl_file(path)
    if suffix in {".xlsx", ".xls"}:
        return _load_excel_file(path)
    raise ValidationError(f"Unsupported file type: {suffix}")


# ---------------------------------------------------------------------
# Build Argilla v2 Settings from YAML or fallback
# ---------------------------------------------------------------------

def _load_dataset_schema_from_yaml(path: Path) -> Tuple[List[Any], List[Any], str]:
    if yaml is None:
        raise ValidationError("YAML support requires pyyaml.")

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValidationError("Invalid YAML schema file")

    guidelines = str(data.get("guidelines", ""))

    fields_cfg = data.get("fields", [])
    if not isinstance(fields_cfg, list):
        raise ValidationError("YAML: 'fields' must be a list")

    fields: List[Any] = []
    for f in fields_cfg:
        if not isinstance(f, dict) or "type" not in f or "name" not in f:
            raise ValidationError("YAML: each field must be an object with 'type' and 'name'")
        if f["type"] == "text":
            fields.append(
                rg.TextField(
                    name=f["name"],
                    title=f.get("title", f["name"]),
                    use_markdown=bool(f.get("use_markdown", False)),
                )
            )
        else:
            raise ValidationError(f"Unsupported field type: {f['type']}")

    questions_cfg = data.get("questions", [])
    if not isinstance(questions_cfg, list):
        raise ValidationError("YAML: 'questions' must be a list")

    questions: List[Any] = []
    for q in questions_cfg:
        if not isinstance(q, dict) or "type" not in q or "name" not in q:
            raise ValidationError("YAML: each question must be an object with 'type' and 'name'")
        q_type = q["type"]
        if q_type == "text":
            questions.append(
                rg.TextQuestion(
                    name=q["name"],
                    title=q.get("title", q["name"]),
                    description=q.get("description"),
                    required=bool(q.get("required", False)),
                    use_markdown=bool(q.get("use_markdown", False)),
                )
            )
        elif q_type == "label":
            labels = q.get("labels")
            if not labels:
                raise ValidationError(f"Label question {q['name']} needs labels.")
            questions.append(
                rg.LabelQuestion(
                    name=q["name"],
                    title=q.get("title", q["name"]),
                    description=q.get("description"),
                    required=bool(q.get("required", False)),
                    labels=list(labels),
                )
            )
        elif q_type == "rating":
            values = q.get("values")
            if not values:
                raise ValidationError(f"Rating question {q['name']} needs values.")
            questions.append(
                rg.RatingQuestion(
                    name=q["name"],
                    title=q.get("title", q["name"]),
                    description=q.get("description"),
                    required=bool(q.get("required", False)),
                    values=list(values),
                )
            )
        else:
            raise ValidationError(f"Unsupported question type: {q_type}")

    return fields, questions, guidelines


def _default_dataset_schema(records: List[Dict[str, Any]]):
    if not records:
        raise ValidationError("No records loaded; cannot infer schema.")
    first_row = records[0]

    fields = [
        rg.TextField(
            name=k,
            title=k.replace("_", " ").title(),
            use_markdown=False,
        )
        for k in first_row.keys()
    ]

    question = rg.LabelQuestion(
        name="label",
        title="Label",
        description="Select the label",
        labels=["Entailment", "Neutral", "Contradiction"],
        required=True,
    )

    guidelines = "Please annotate the record."
    return fields, [question], guidelines




def _pick_record_id(rec: Dict[str, Any], fallback_idx: int) -> str:
    """Try to get a stable id from an Argilla record dict."""
    for key in ("id", "_id", "record_id", "uuid"):
        v = rec.get(key)
        if v is not None and str(v).strip():
            return str(v)
    return f"idx_{fallback_idx:06d}"


def _split_records_greedy(
    records: List[Dict[str, Any]],
    items_per_user: int,
    overlap: int,
    seed: Optional[int] = None,
) -> tuple[List[List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """
    Split records into k subsets with:
      - each record assigned 'overlap' times to distinct subsets
      - each subset capacity ~ items_per_user (can be slightly exceeded if necessary)

    Returns (subsets, plan_rows), where plan_rows includes:
      subset_index, subset_name, record_id
    """
    if items_per_user < 1:
        raise ValidationError("--items-per-user must be >= 1")
    if overlap < 1:
        raise ValidationError("--overlap must be >= 1")

    n = len(records)
    if n == 0:
        raise ValidationError("Main dataset has 0 records; nothing to split.")

    # number of subsets/users needed
    k = int(math.ceil((n * overlap) / float(items_per_user)))
    k = max(k, 1)
    if overlap > k:
        raise ValidationError(
            f"overlap={overlap} is impossible with only k={k} subsets. "
            "Increase items_per_user or reduce overlap."
        )

    rng = random.Random(seed)
    idxs = list(range(n))
    rng.shuffle(idxs)

    subsets: List[List[Dict[str, Any]]] = [[] for _ in range(k)]
    # min-heap of (current_size, subset_index)
    heap: List[tuple[int, int]] = [(0, i) for i in range(k)]
    heapq.heapify(heap)

    plan_rows: List[Dict[str, Any]] = []

    for shuffled_pos, rec_idx in enumerate(idxs):
        rec = records[rec_idx]
        record_id = _pick_record_id(rec, fallback_idx=rec_idx)

        # pick 'overlap' distinct subsets with smallest current fill
        chosen: List[int] = []
        popped: List[tuple[int, int]] = []

        for _ in range(overlap):
            # pop until we find a subset not yet chosen
            while True:
                if not heap:
                    raise ValidationError("Internal error: heap empty during assignment.")
                size, si = heapq.heappop(heap)
                popped.append((size, si))
                if si not in chosen:
                    chosen.append(si)
                    break

        # push back all popped (we'll update chosen ones next)
        for size, si in popped:
            heapq.heappush(heap, (size, si))

        # now actually assign record to chosen subsets and update heap sizes
        # simplest: rebuild heap after updates (k is usually not huge)
        for si in chosen:
            subsets[si].append(rec)
            plan_rows.append(
                {
                    "subset_index": si,
                    "record_id": record_id,
                }
            )

        heap = [(len(subsets[i]), i) for i in range(k)]
        heapq.heapify(heap)

    return subsets, plan_rows



# ---------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------

@app.command("list")
def list_datasets(
    json_output: bool = typer.Option(
        False,
        "--json/--no-json",
        help="Output JSON",
    ),
) -> None:
    """List all datasets across all workspaces."""
    try:
        client = get_client(load_settings().settings)
        rows: List[dict[str, Any]] = []
        for ws in client.workspaces:  # type: ignore[attr-defined]
            ws_name = getattr(ws, "name", "")
            for ds in ws.datasets:  # type: ignore[attr-defined]
                rows.append(
                    {
                        "name": getattr(ds, "name", ""),
                        "id": getattr(ds, "id", ""),
                        "workspace": ws_name,
                        "created_at": getattr(ds, "created_at", ""),
                        "description": getattr(ds, "description", ""),
                    }
                )
    except Exception as e:
        exit_with_error(e, verbose=state.verbose)
        return

    if state.json_output or json_output:
        emit_json(rows)
    else:
        print_table(rows)


@app.command("download")
def download_dataset(
    dataset_name: str = typer.Argument(..., help="Dataset name"),
    workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace name"),
    map_file: Optional[Path] = typer.Option(
        None,
        "--map",
        help="Path to JSON mapping file (output_field -> JMESPath expression).",
    ),
    fmt: str = typer.Option("jsonl", "--fmt", help="Output format: jsonl|csv|parquet"),
    output: Optional[Path] = typer.Option(None, "--output", help="Output file path"),
    force: bool = typer.Option(False, "--force", help="Overwrite existing output"),
    json_output: bool = typer.Option(False, "--json/--no-json", help="Output JSON"),
    completed_only: bool = typer.Option(
        False,
        "--completed-only/--no-completed-only",
        help="Only include records with status=completed",
    ),
) -> None:
    """Download/export a dataset from Argilla."""
    client = get_client(load_settings().settings)
    try:
        dataset, _ws_name = _resolve_dataset(client, dataset_name, workspace)
    except Exception as e:
        exit_with_error(e, verbose=state.verbose)
        return

    records_iter = _iter_record_dicts(dataset)
    records_iter = _apply_completed_filter(records_iter, completed_only)

    if output is None:
        output = Path.cwd() / f"{dataset_name}.{fmt}"

    target_path = output
    if target_path.suffix == "":
        target_path = target_path.with_suffix(f".{fmt}")

    if target_path.exists() and not force:
        print_error("Output path exists; use --force to overwrite.")
        raise typer.Exit(code=13)

    target_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if fmt == "jsonl":
            if map_file is not None:
                mapping = _load_mapping_file(map_file)
                compiled = _compile_mapping(mapping)
                with target_path.open("w", encoding="utf-8") as f:
                    for row in records_iter:
                        out_row = _transform_record(row, compiled, "join", ", ")
                        f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            else:
                with target_path.open("w", encoding="utf-8") as f:
                    for row in records_iter:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")

        elif fmt in {"csv", "parquet"}:
            try:
                import pandas as pd  # type: ignore
            except Exception as e:
                raise ValidationError("CSV/Parquet export requires pandas") from e

            if map_file is not None:
                mapping = _load_mapping_file(map_file)
                compiled = _compile_mapping(mapping)
                rows: List[Dict[str, Any]] = [
                    _transform_record(r, compiled, "join", ", ") for r in records_iter
                ]
                df = pd.DataFrame(rows)
            else:
                records_list = dataset.records.to_list(flatten=True)
                df = pd.DataFrame(records_list)
                if completed_only and "status" in df.columns:
                    df = df[df["status"] == "completed"]

            if fmt == "csv":
                df.to_csv(target_path, index=False)
            else:
                df.to_parquet(target_path, index=False)

        else:
            raise ValidationError("Unsupported format; choose from jsonl,csv,parquet")

    except Exception as e:
        exit_with_error(e, verbose=state.verbose)
        return

    if state.json_output or json_output:
        emit_json([str(target_path)])
    else:
        print_ok(f"Saved: {target_path}")


@app.command(
    "import-upload",
    help="Import a local dataset file with preview and optionally upload it to Argilla.",
)
def import_upload(
    source: Optional[Path] = typer.Option(
        None,
        "--file",
        "-f",
        help="Path to the dataset file (.jsonl, .json, .csv, .xlsx, .xls). If omitted, you'll be prompted.",
    ),
    use_dialog: bool = typer.Option(
        False,
        "--dialog",
        "-d",
        help="Use a macOS file dialog to select the file.",
    ),
    current_directory: bool = typer.Option(
        False,
        "--current-directory",
        "-cd",
        help="List candidate files in the current directory to choose from.",
    ),
    workspace: str = typer.Option(
        ...,
        "--workspace",
        "-ws",
        help="Target Argilla workspace for the dataset.",
    ),
    name: str = typer.Option(
        ...,
        "--name",
        "-n",
        help="Name of the dataset to create in Argilla.",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Optional YAML schema defining fields/questions.",
    ),
) -> None:
    """Read a local file, preview it, and optionally upload it as an Argilla v2 dataset."""

    # 1) Pick file
    try:
        if use_dialog:
            path = _choose_file_with_macos_dialog()
            if path is None:
                typer.echo("No file selected. Aborting.")
                raise typer.Exit(code=1)
        elif current_directory:
            path = _choose_file_from_current_directory()
        elif source is not None:
            path = source
        else:
            raise ValidationError("You must provide either --file, --dialog, or --current-directory.")
    except Exception as e:
        exit_with_error(e, verbose=state.verbose)
        return

    if not path.exists():
        exit_with_error(ValidationError(f"File not found: {path}"), verbose=state.verbose)
        return

    typer.echo(f"\nSelected file: {path}")
    _preview_file(path)

    if not typer.confirm("\nDoes this look like the correct file?"):
        typer.echo("Import cancelled by user.")
        raise typer.Exit(code=0)

    # 2) Load rows
    try:
        raw_records = _load_records(path)
    except Exception as e:
        exit_with_error(e, verbose=state.verbose)
        return

    typer.echo(f"\nSuccessfully read {len(raw_records)} records from {path.name}.")

    if raw_records and typer.confirm("Do you want to see a sample of the loaded records?", default=False):
        sample_size = min(3, len(raw_records))
        typer.echo(f"\nSample of first {sample_size} records:")
        for i in range(sample_size):
            typer.echo(f"\nRecord #{i + 1}:")
            typer.echo(json.dumps(raw_records[i], ensure_ascii=False, indent=2))

    if not typer.confirm(
        f"\nDo you want to upload this dataset now to workspace '{workspace}' as '{name}'?",
        default=True,
    ):
        typer.echo("Upload cancelled by user.")
        raise typer.Exit(code=0)

    # 3) Build settings + upload via Argilla v2 API
    try:
        if config:
            fields, questions, guidelines = _load_dataset_schema_from_yaml(config)
        else:
            fields, questions, guidelines = _default_dataset_schema(raw_records)

        settings = rg.Settings(
            guidelines=guidelines,
            fields=fields,
            questions=questions,
        )

        # Note: get_client(...) is still used for config/auth checks in your CLI,
        # but the upload itself uses the official Argilla v2 SDK objects.
        _ = get_client(load_settings().settings)

        dataset = rg.Dataset(name=name, settings=settings, workspace=workspace)
        dataset.create()
        dataset.records.log(records=raw_records)

        print_ok(f"Dataset '{name}' uploaded to workspace '{workspace}' ({len(raw_records)} records)")

    except Exception as e:
        exit_with_error(e, verbose=state.verbose)
        return


@app.command("split-push")
def split_push(
    dataset_name: str = typer.Option(..., "--name", "-n", help="Main dataset name in Argilla"),
    workspace: str = typer.Option(..., "--workspace", "-ws", help="Workspace of the main dataset"),
    # splitting rules
    items_per_user: int = typer.Option(..., "--items-per-user", help="How many items one participant annotates"),
    overlap: int = typer.Option(1, "--overlap", help="How many different participants see the same item"),
    seed: Optional[int] = typer.Option(42, "--seed", help="Random seed for reproducible splits"),
    # schema for the NEW sub-datasets
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="YAML schema file (fields/questions/guidelines)"),
    # output + naming
    mode: str = typer.Option(
        "same-workspace",
        "--mode",
        help="same-workspace|per-user-workspace",
    ),
    workspace_prefix: str = typer.Option(
        "prolific",
        "--workspace-prefix",
        help="Used only for mode=per-user-workspace: workspace name prefix",
    ),
    dataset_prefix: str = typer.Option(
        "subset",
        "--dataset-prefix",
        help="Name prefix for created sub-datasets",
    ),
    plan_out: Path = typer.Option(
        Path("./split_plan.csv"),
        "--plan-out",
        help="Where to write the split plan CSV",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run/--no-dry-run",
        help="Only compute and write split plan; do not push datasets",
    ),
) -> None:
    """
    Split an existing Argilla dataset into enough sub-datasets according to rules
    and push them back to Argilla.

    Typical Prolific flow (Option B):
      - mode=per-user-workspace
      - one workspace + one dataset per participant
      - later: one credential per workspace/user
    """
    if mode not in {"same-workspace", "per-user-workspace"}:
        raise typer.BadParameter("--mode must be same-workspace or per-user-workspace")

    try:
        client = get_client(load_settings().settings)
        main_ds, _ = _resolve_dataset(client, dataset_name, workspace)

        # export records as dicts
        records = list(_iter_record_dicts(main_ds))

        subsets, plan_rows = _split_records_greedy(
            records=records,
            items_per_user=items_per_user,
            overlap=overlap,
            seed=seed,
        )

        # write plan CSV
        plan_out.parent.mkdir(parents=True, exist_ok=True)
        with plan_out.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["subset_index", "record_id", "workspace", "dataset"],
            )
            w.writeheader()
            for r in plan_rows:
                si = int(r["subset_index"])
                ws_name = workspace
                if mode == "per-user-workspace":
                    ws_name = f"{workspace_prefix}_{(si + 1):03d}"
                ds_name = f"{dataset_prefix}_{(si + 1):03d}"
                w.writerow(
                    {
                        "subset_index": si,
                        "record_id": r["record_id"],
                        "workspace": ws_name,
                        "dataset": ds_name,
                    }
                )

        print_ok(
            f"Split plan written: {plan_out} (records={len(records)}, "
            f"items_per_user={items_per_user}, overlap={overlap}, subsets={len(subsets)})"
        )

        if dry_run:
            print_ok("Dry-run: not pushing any sub-datasets.")
            return

        # load schema for the new datasets
        if config:
            fields, questions, guidelines = _load_dataset_schema_from_yaml(config)
        else:
            fields, questions, guidelines = _default_dataset_schema(records)

        # Build Settings once and reuse for all subsets
        settings = rg.Settings(
            guidelines=guidelines,
            fields=fields,
            questions=questions,
        )

        # helper: ensure workspace exists (best-effort)
        def _ensure_workspace(ws_name: str) -> None:
            """Ensure a workspace exists (best-effort).

            Argilla SDKs differ a bit across versions. We try multiple public-ish
            entry points. If we cannot create the workspace, we raise a
            ValidationError telling the user to create it manually.
            """
            # 1) Does it already exist?
            try:
                for ws in client.workspaces:  # type: ignore[attr-defined]
                    if getattr(ws, "name", None) == ws_name:
                        return
            except Exception:
                # If listing workspaces fails, we still try to create below.
                pass

            last_err: Optional[Exception] = None

            # 2) Try common manager methods: client.workspaces.create(...)
            try:
                mgr = getattr(client, "workspaces", None)
                for meth in ("create", "add", "new"):
                    fn = getattr(mgr, meth, None)
                    if callable(fn):
                        try:
                            fn(name=ws_name)
                            return
                        except TypeError:
                            # Some SDKs use positional arg
                            fn(ws_name)
                            return
                        except Exception as e:
                            last_err = e
            except Exception as e:
                last_err = e

            # 3) Try client.create_workspace(...) style
            try:
                fn = getattr(client, "create_workspace", None)
                if callable(fn):
                    try:
                        fn(name=ws_name)
                    except TypeError:
                        fn(ws_name)
                    return
            except Exception as e:
                last_err = e

            # 4) Try rg.Workspace(...).create(...)
            try:
                WorkspaceCls = getattr(rg, "Workspace", None)
                if WorkspaceCls is not None:
                    ws_obj = WorkspaceCls(name=ws_name)
                    create_fn = getattr(ws_obj, "create", None)
                    if callable(create_fn):
                        try:
                            create_fn(client=client)
                        except TypeError:
                            create_fn()
                        return
            except Exception as e:
                last_err = e

            # If we got here, we could not create it.
            hint = (
                f"Workspace {ws_name!r} does not exist and could not be created via the SDK. "
                "Please create it in the Argilla UI (or ensure your API key has permissions), "
                "then re-run split-push."
            )
            if last_err is not None:
                raise ValidationError(hint + f" Last error: {last_err}")
            raise ValidationError(hint)

        # push each subset using the public Argilla v2 Dataset API
        for i, subset_records in enumerate(subsets, start=1):
            subset_ws = workspace
            if mode == "per-user-workspace":
                subset_ws = f"{workspace_prefix}_{i:03d}"
                _ensure_workspace(subset_ws)

            subset_name = f"{dataset_prefix}_{i:03d}"

            # Build clean records dicts that match the schema field names
            cleaned: List[Dict[str, Any]] = []
            schema_field_names = [f.name for f in fields]
            for row in subset_records:
                out: Dict[str, Any] = {}
                for fn in schema_field_names:
                    v = row.get(fn)
                    # TextField expects text-like values; normalise to string
                    out[fn] = "" if v is None else str(v)
                cleaned.append(out)

            ds = rg.Dataset(name=subset_name, settings=settings, workspace=subset_ws)
            try:
                ds.create()
            except Exception as e:
                # Most common cause here: workspace missing or no permission.
                raise ValidationError(
                    f"Failed to create dataset {subset_name!r} in workspace {subset_ws!r}: {e}"
                ) from e
            ds.records.log(records=cleaned)

            print_ok(
                f"Pushed: workspace={subset_ws} dataset={subset_name} records={len(cleaned)}"
            )

    except Exception as e:
        exit_with_error(e, verbose=state.verbose)
        return