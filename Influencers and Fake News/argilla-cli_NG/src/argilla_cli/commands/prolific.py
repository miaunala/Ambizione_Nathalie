from __future__ import annotations

import csv
import math
import os
import secrets
import string
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import typer

from argilla_cli.clients.argilla_client import get_client
from argilla_cli.errors import ValidationError, exit_with_error
from argilla_cli.globals import state
from argilla_cli.io_utils import emit_json, print_error, print_ok
from argilla_cli.settings import load_settings

app = typer.Typer(help="Prolific / Study integration commands")


# -----------------------------
# Config
# -----------------------------

@dataclass
class ProlificStudyConfig:
    """Minimal config for printing instructions.

    You can keep this YAML tiny and still get useful output.

    Example YAML:

    dataset_name: test
    workspace: deepfake_ncii
    completion_code: ABC123
    credentials_csv: ./prolific_credentials.csv
    """

    dataset_name: str
    workspace: str = "main"
    completion_code: Optional[str] = None
    credentials_csv: Optional[Path] = None


def load_prolific_config(path: Path) -> ProlificStudyConfig:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise ValidationError(
            "PyYAML is required for --config. Install with: pip install pyyaml"
        ) from e

    if not path.exists():
        raise ValidationError(
            "Config file not found: "
            f"{path}\n\n"
            "Example prolific.yaml:\n\n"
            f"{_prolific_yaml_template()}"
        )

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValidationError("Prolific config YAML must be a mapping/object")

    dataset_name = data.get("dataset_name")
    if not dataset_name or not isinstance(dataset_name, str):
        raise ValidationError("Config must include dataset_name (string)")

    workspace = data.get("workspace", "main")
    if not isinstance(workspace, str) or not workspace:
        raise ValidationError("workspace must be a non-empty string")

    completion_code = data.get("completion_code")
    if completion_code is not None and not isinstance(completion_code, str):
        raise ValidationError("completion_code must be a string if provided")

    cred = data.get("credentials_csv")
    credentials_csv = Path(cred) if isinstance(cred, str) and cred else None

    return ProlificStudyConfig(
        dataset_name=dataset_name,
        workspace=workspace,
        completion_code=completion_code,
        credentials_csv=credentials_csv,
    )


# -----------------------------
# Helpers
# -----------------------------

def _random_password(length: int = 14) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def _write_credentials_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["username", "password"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _count_datasets_in_workspace(client: Any, workspace: str) -> int:
    # Works with Argilla 2.x: client.workspaces yields workspace objects; each has .name and .datasets
    for ws in client.workspaces:  # type: ignore[attr-defined]
        if getattr(ws, "name", None) == workspace:
            datasets = getattr(ws, "datasets", None)
            if datasets is None:
                return 0
            try:
                return len(list(datasets))
            except Exception:
                # some SDKs expose datasets as already-list-like
                try:
                    return len(datasets)
                except Exception:
                    return 0
    raise ValidationError(f"Workspace not found: {workspace!r}")


def _default_ui_base_from_api(api_url: str) -> str:
    """Best-effort: many deployments use the same base for API and UI.

    If your API is e.g. https://argilla.example.com, UI is usually the same.
    If your API includes a path, we strip it.
    """
    api_url = api_url.rstrip("/")
    # If someone set api_url to .../api, strip that
    if api_url.endswith("/api"):
        api_url = api_url[: -len("/api")]
    return api_url


def _choose_file_with_macos_dialog(prompt: str) -> Optional[Path]:
    """Open a native macOS file chooser and return the selected path.

    Requires macOS with `osascript` available. Returns None if the user cancels.
    """
    script = f'POSIX path of (choose file with prompt "{prompt}")'
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


def _prolific_yaml_template() -> str:
    return (
        "# Prolific → Argilla integration config\n"
        "# Save this as e.g. prolific.yaml\n\n"
        "dataset_name: test\n"
        "workspace: deepfake_ncii\n"
        "# Optional: shown to participants to paste back into Prolific\n"
        "completion_code: ABC123\n"
        "# Optional: where you stored the generated credentials CSV\n"
        "credentials_csv: ./prolific_credentials.csv\n"
    )


# -----------------------------
# Prolific API helpers
# -----------------------------

PROLIFIC_API_BASE_DEFAULT = "https://api.prolific.com/api/v1"


def _get_prolific_token() -> str:
    """Read the Prolific API token from env.

    Prolific uses header: Authorization: Token <token>
    """
    token = os.environ.get("PROLIFIC_API_TOKEN") or os.environ.get("PROLIFIC_TOKEN")
    if not token:
        raise ValidationError(
            "Missing Prolific API token. Set PROLIFIC_API_TOKEN in your environment.\n\n"
            "Example:\n"
            "  export PROLIFIC_API_TOKEN='your_token_here'"
        )
    return token


def _get_prolific_base() -> str:
    return (os.environ.get("PROLIFIC_API_BASE") or PROLIFIC_API_BASE_DEFAULT).rstrip("/")


def _prolific_headers() -> Dict[str, str]:
    return {
        "Authorization": f"Token {_get_prolific_token()}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def _prolific_request(method: str, path: str, *, json_body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """Small wrapper around Prolific API calls.

    Raises ValidationError with a readable message on non-2xx.
    """
    base = _get_prolific_base()
    url = f"{base}/{path.lstrip('/')}"
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.request(method.upper(), url, headers=_prolific_headers(), json=json_body, params=params)
    except Exception as e:
        raise ValidationError(f"Failed to call Prolific API: {e}") from e

    if resp.status_code < 200 or resp.status_code >= 300:
        # best-effort error extraction
        try:
            payload = resp.json()
        except Exception:
            payload = {"text": resp.text}
        raise ValidationError(
            f"Prolific API error {resp.status_code} for {method.upper()} {url}: {payload}"
        )

    try:
        return resp.json()
    except Exception:
        return {"text": resp.text}


def _read_credentials_count(credentials_csv: Path) -> int:
    """Count rows in a username/password CSV produced by generate-credentials."""
    if not credentials_csv.exists():
        raise ValidationError(f"Credentials file not found: {credentials_csv}")
    with credentials_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "username" not in reader.fieldnames or "password" not in reader.fieldnames:
            raise ValidationError("CSV must contain columns: username,password")
        return sum(1 for _ in reader)


def _default_external_url(argilla_api_url: str) -> str:
    """Default external URL for Prolific external studies.

    We point to the Argilla UI base. Prolific will append participant identifiers via query params if you include them.
    """
    ui_base = _default_ui_base_from_api(argilla_api_url).rstrip("/")
    # Include Prolific placeholders as documented by Prolific.
    return (
        f"{ui_base}/?participant={{%PROLIFIC_PID%}}&study={{%STUDY_ID%}}&session={{%SESSION_ID%}}"
    )


# -----------------------------
# Commands
# -----------------------------

@app.command("ping")
def ping() -> None:
    """Small test command to check wiring."""
    typer.echo("Prolific integration CLI is alive.")


@app.command("generate-credentials")
def generate_credentials(
    workspace: str = typer.Option(
        ..., "--workspace", "-ws", help="Workspace to target"
    ),
    base: str = typer.Option(
        "workspace",
        "--base",
        help="How many credentials to generate: workspace|datasets|fixed",
    ),
    n_fixed: Optional[int] = typer.Option(
        None,
        "--n",
        help="When --base fixed: number of credentials to generate",
    ),
    margin: float = typer.Option(
        0.2,
        "--margin",
        help="Extra margin on top of dataset count (only used for --base datasets)",
    ),
    prefix: str = typer.Option(
        "prolific",
        "--prefix",
        help="Username prefix (result: <prefix>_001, <prefix>_002, ...) ",
    ),
    out: Path = typer.Option(
        Path("./prolific_credentials.csv"), "--out", help="Output CSV path"
    ),
    password_length: int = typer.Option(14, "--password-length", help="Password length"),
    json_output: bool = typer.Option(False, "--json/--no-json", help="Output JSON"),
) -> None:
    """
    Generate participant credentials.

    Defaults to --base workspace => exactly 1 credential for the workspace
    (unless you override with --base fixed --n K).
    """
    if margin < 0:
        raise typer.BadParameter("--margin must be >= 0")
    if password_length < 8:
        raise typer.BadParameter("--password-length should be >= 8")

    try:
        client = get_client(load_settings().settings)
        ds_count = _count_datasets_in_workspace(client, workspace)

        if base == "workspace":
            n = 1
        elif base == "datasets":
            n = max(1, int(math.ceil(ds_count * (1.0 + margin))))
        elif base == "fixed":
            if n_fixed is None or n_fixed < 1:
                raise ValidationError("When --base fixed you must pass --n >= 1")
            n = int(n_fixed)
        else:
            raise ValidationError("--base must be one of: workspace, datasets, fixed")

        rows: List[Dict[str, str]] = []
        for i in range(1, n + 1):
            rows.append(
                {
                    "username": f"{prefix}_{i:03d}",
                    "password": _random_password(password_length),
                }
            )

        _write_credentials_csv(out, rows)

    except Exception as e:
        exit_with_error(e, verbose=state.verbose)
        return

    payload = {
        "workspace": workspace,
        "dataset_count": ds_count,
        "margin": margin,
        "n_credentials": n,
        "out": str(out),
        "rows": rows if (state.json_output or json_output) else None,
        "base": base,
        "n_fixed": n_fixed,
    }

    if state.json_output or json_output:
        emit_json(payload)
    else:
        print_ok(
            f"Saved credentials: {out} (workspace={workspace}, base={base}, n={n})"
        )


@app.command("print-instructions")
def print_instructions(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to Prolific YAML config"
    ),
    use_dialog: bool = typer.Option(
        False,
        "--dialog",
        "-d",
        help="Pick the Prolific YAML config via a macOS file dialog",
    ),
    base_url: Optional[str] = typer.Option(
        None,
        "--base-url",
        help="Argilla UI base URL (overrides env). If omitted, derived from ARGILLA_API_URL.",
    ),
    json_output: bool = typer.Option(False, "--json/--no-json", help="Output JSON"),
) -> None:
    """Print copy-pastable instructions you can paste into Prolific."""
    try:
        if use_dialog:
            picked = _choose_file_with_macos_dialog("Select prolific.yaml")
            if picked is None:
                raise ValidationError("No file selected.")
            config_path = picked
        else:
            if config is None:
                raise ValidationError("You must pass --config or use --dialog")
            config_path = config

        cfg = load_prolific_config(config_path)
        settings = load_settings().settings
        ui_base = (base_url or _default_ui_base_from_api(str(settings.argilla_api_url))).rstrip("/")

        # Best-effort link. If you want a direct dataset link, you can extend this later
        landing = f"{ui_base}/"

        instructions = (
            "Participant instructions (copy/paste)\n"
            "--------------------------------\n"
            "1) Open the task link:\n"
            f"   {landing}\n"
            "2) Log in with the username/password provided in this Prolific study.\n"
            f"3) Go to workspace '{cfg.workspace}' and open dataset '{cfg.dataset_name}'.\n"
            "4) Complete the task and submit your answers.\n"
        )

        if cfg.completion_code:
            instructions += (
                "5) Return to Prolific and enter this completion code:\n"
                f"   {cfg.completion_code}\n"
            )

        if cfg.credentials_csv:
            instructions += (
                "\nResearcher note:\n"
                f"Credentials file: {cfg.credentials_csv}\n"
            )

    except Exception as e:
        exit_with_error(e, verbose=state.verbose)
        return

    if state.json_output or json_output:
        emit_json({"instructions": instructions, "config": config_path.as_posix()})
    else:
        typer.echo(instructions)


@app.command("sanity")
def sanity(
    workspace: str = typer.Option(..., "--workspace", "-ws", help="Workspace name"),
) -> None:
    """Quick check: can we reach Argilla and list dataset count for a workspace?"""
    try:
        client = get_client(load_settings().settings)
        ds_count = _count_datasets_in_workspace(client, workspace)
        print_ok(f"OK: workspace={workspace}, datasets={ds_count}")
    except Exception as e:
        exit_with_error(e, verbose=state.verbose)


@app.command("list-workspaces")
def list_workspaces(
    json_output: bool = typer.Option(False, "--json/--no-json", help="Output JSON"),
) -> None:
    """List Prolific workspaces (requires PROLIFIC_API_TOKEN)."""
    try:
        data = _prolific_request("GET", "/workspaces/")
    except Exception as e:
        exit_with_error(e, verbose=state.verbose)
        return

    # Prolific usually returns a list; be defensive
    if state.json_output or json_output:
        emit_json(data)
        return

    rows: List[Dict[str, Any]] = []
    if isinstance(data, list):
        for ws in data:
            if isinstance(ws, dict):
                rows.append(
                    {
                        "id": ws.get("id"),
                        "name": ws.get("name"),
                        "balance": ws.get("balance"),
                    }
                )
    elif isinstance(data, dict) and isinstance(data.get("results"), list):
        for ws in data["results"]:
            if isinstance(ws, dict):
                rows.append(
                    {
                        "id": ws.get("id"),
                        "name": ws.get("name"),
                        "balance": ws.get("balance"),
                    }
                )
    else:
        rows = [{"note": "Unexpected response shape", "data": str(data)[:200]}]

    # Local helper already exists in other commands; here keep it simple:
    for r in rows:
        typer.echo(str(r))


@app.command("create-study")
def create_study(
    prolific_workspace_id: str = typer.Option(
        ..., "--prolific-workspace-id", help="Prolific workspace ID (NOT the Argilla workspace name)"
    ),
    title: str = typer.Option(..., "--title", help="Study title shown to participants"),
    internal_name: Optional[str] = typer.Option(None, "--internal-name", help="Internal study name"),
    external_url: Optional[str] = typer.Option(
        None,
        "--external-url",
        help=(
            "External study URL (participants are redirected here). If omitted, we derive it from ARGILLA_API_URL and add Prolific placeholders."
        ),
    ),
    completion_code: str = typer.Option(
        ..., "--completion-code", help="Completion code participants submit on ProLific"
    ),
    credentials_csv: Path = typer.Option(
        ..., "--credentials", help="CSV with username,password (used to set total places)"
    ),
    credential_pool_id: Optional[str] = typer.Option(
        None,
        "--credential-pool-id",
        help=(
            "Optional: Prolific credential pool ID. If provided, Prolific will distribute unique credentials to participants automatically."
        ),
    ),
    argilla_workspace: Optional[str] = typer.Option(
        None, "--argilla-workspace", help="Argilla workspace name (for instructions only)"
    ),
    argilla_dataset: Optional[str] = typer.Option(
        None, "--argilla-dataset", help="Argilla dataset name (for instructions only)"
    ),
    json_output: bool = typer.Option(False, "--json/--no-json", help="Output JSON"),
) -> None:
    """Create a Prolific external study.

    Minimal, pragmatic integration:
    - We set `total_available_places` from the number of credentials you generated.
    - If you pass `--credential-pool-id`, Prolific will hand out those credentials per participant.
      (This is the *actual* bridge between Prolific and Argilla accounts.)

    Notes:
    - You still configure eligibility, reward, etc. in Prolific (or extend this command later).
    - Publishing/launching can be done in the Prolific UI if you prefer.
    """
    try:
        settings = load_settings().settings
        n_places = _read_credentials_count(credentials_csv)
        if n_places < 1:
            raise ValidationError("credentials CSV contains no rows")

        if external_url is None:
            external_url = _default_external_url(str(settings.argilla_api_url))

        payload: Dict[str, Any] = {
            "title": title,
            "internal_name": internal_name or title,
            "workspace_id": prolific_workspace_id,
            "external_study_url": external_url,
            "completion_codes": [completion_code],
            "total_available_places": n_places,
        }

        if credential_pool_id:
            payload["credential_pool_id"] = credential_pool_id

        # Create draft study
        data = _prolific_request("POST", "/studies/", json_body=payload)

        # Print a short next-steps block
        study_id = data.get("id") if isinstance(data, dict) else None
        study_url = data.get("url") if isinstance(data, dict) else None

    except Exception as e:
        exit_with_error(e, verbose=state.verbose)
        return

    if state.json_output or json_output:
        emit_json({"study": data})
        return

    print_ok(f"Created Prolific study draft: id={study_id}")
    if study_url:
        typer.echo(f"Study URL (UI): {study_url}")

    typer.echo("\nWhat to do next:")
    typer.echo("- In Prolific, review the draft, set reward/eligibility, then publish.")
    if credential_pool_id:
        typer.echo("- Credential pool is attached: Prolific will distribute Argilla usernames/passwords automatically.")
    else:
        typer.echo(
            "- No credential pool attached. Participants will NOT receive unique Argilla credentials automatically.\n"
            "  Either create a credential pool in Prolific and re-run with --credential-pool-id,\n"
            "  or distribute credentials another way (not recommended)."
        )

    if argilla_workspace and argilla_dataset:
        typer.echo("\nArgilla target (for your instructions):")
        typer.echo(f"- workspace: {argilla_workspace}")
        typer.echo(f"- dataset:   {argilla_dataset}")


@app.command("create-users")
def create_users(
    credentials: Path = typer.Option(
        ..., "--credentials", "-c", help="CSV with username,password"
    ),
    role: str = typer.Option(
        "annotator",
        "--role",
        help="Argilla role to assign (annotator or admin)",
    ),
    skip_existing: bool = typer.Option(
        True,
        "--skip-existing/--no-skip-existing",
        help="Skip users that already exist",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print what would be done without creating users",
    ),
    use_dialog: bool = typer.Option(
        False,
        "--dialog",
        "-d",
        help="Pick the credentials CSV via a macOS file dialog",
    ),
) -> None:
    """
    Create Argilla users from a credentials CSV (username,password).

    This is REQUIRED for Prolific integration:
    credentials alone are not enough.
    """
    if use_dialog:
        picked = _choose_file_with_macos_dialog("Select prolific_credentials.csv")
        if picked is None:
            raise ValidationError("No credentials file selected.")
        credentials = picked

    if role not in {"annotator", "admin"}:
        raise typer.BadParameter("role must be 'annotator' or 'admin'")

    if not credentials.exists():
        raise ValidationError(f"Credentials file not found: {credentials}")

    try:
        client = get_client(load_settings().settings)

        created = []
        skipped = []

        with credentials.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "username" not in reader.fieldnames or "password" not in reader.fieldnames:
                raise ValidationError(
                    "CSV must contain columns: username,password"
                )

            for row in reader:
                username = row["username"].strip()
                password = row["password"].strip()

                if not username or not password:
                    continue

                try:
                    if dry_run:
                        typer.echo(f"[dry-run] would create user: {username} (role={role})")
                        created.append(username)
                    else:
                        client.users.create(
                            username=username,
                            password=password,
                            role=role,
                        )
                        created.append(username)
                except Exception as e:
                    # User probably already exists
                    if skip_existing:
                        skipped.append(username)
                        continue
                    raise e

    except Exception as e:
        exit_with_error(e, verbose=state.verbose)
        return

    if dry_run:
        print_ok(f"Dry-run complete. Users to create: {len(created)}")
    else:
        print_ok(f"Users created: {len(created)}")
    if skipped:
        print_ok(f"Users skipped (already exist): {len(skipped)}")