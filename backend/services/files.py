"""File storage helpers used by the API layer."""
from __future__ import annotations

from pathlib import Path

from fastapi import UploadFile

PROJECT_DIR = Path(__file__).resolve().parents[2]
HDF5_DATA_DIR = PROJECT_DIR / "data" / "hdf5_file"
WELL_DATA_DIR = PROJECT_DIR / "data" / "well_data"
STORAGE_DIR = PROJECT_DIR / "backend" / "storage"
UPLOAD_DIR = STORAGE_DIR / "uploads"
UPLOAD_H5_DIR = UPLOAD_DIR / "h5"
UPLOAD_WELL_DIR = UPLOAD_DIR / "well"
OUTPUT_DIR = STORAGE_DIR / "outputs"


async def save_upload_file(upload: UploadFile, target_path: Path) -> None:
    """Save an uploaded file to disk in chunks.

    Parameters
    ----------
    upload:
        FastAPI upload object received from the frontend.
    target_path:
        Final path under the backend storage directory.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("wb") as target:
        while chunk := await upload.read(1024 * 1024):
            target.write(chunk)


def get_upload_path(file_id: str | None, suffix: str | None = None) -> Path | None:
    """Return an upload path by id, optionally creating a path with a suffix.

    When ``suffix`` is provided, the function returns the target path for a new
    upload. When ``suffix`` is omitted, it searches for an existing uploaded
    file whose stem is ``file_id``.
    """
    if not file_id:
        return None
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    if suffix:
        return UPLOAD_DIR / f"{file_id}{suffix}"

    matches = list(UPLOAD_DIR.glob(f"{file_id}.*"))
    return matches[0] if matches else None


def safe_filename(filename: str) -> str:
    """Return a safe basename for a user-provided filename."""
    name = Path(filename).name.strip()
    if not name or name in {".", ".."}:
        raise ValueError("Invalid filename.")
    return name


def make_file_id(source: str, filename: str) -> str:
    """Build a stable file id used by the frontend and API."""
    return f"{source}:{safe_filename(filename)}"


def parse_file_id(file_id: str | None) -> tuple[str, str] | None:
    """Split a file id into source and filename."""
    if not file_id or ":" not in file_id:
        return None
    source, filename = file_id.split(":", 1)
    return source, safe_filename(filename)


def get_h5_upload_path(filename: str) -> Path:
    """Return the upload path for an HDF5 file, preserving its filename."""
    UPLOAD_H5_DIR.mkdir(parents=True, exist_ok=True)
    return UPLOAD_H5_DIR / safe_filename(filename)


def get_well_upload_path(filename: str) -> Path:
    """Return the upload path for a well trajectory file, preserving its filename."""
    UPLOAD_WELL_DIR.mkdir(parents=True, exist_ok=True)
    return UPLOAD_WELL_DIR / safe_filename(filename)


def resolve_file_path(file_id: str | None, expected_kind: str) -> Path | None:
    """Resolve a frontend file id to an allowed local file path.

    ``expected_kind`` must be ``h5`` or ``well``. Supported ids are:

    - ``data_h5:<filename>``
    - ``upload_h5:<filename>``
    - ``data_well:<filename>``
    - ``upload_well:<filename>``
    """
    parsed = parse_file_id(file_id)
    if parsed is None:
        return None
    source, filename = parsed
    roots = {
        ("h5", "data_h5"): HDF5_DATA_DIR,
        ("h5", "upload_h5"): UPLOAD_H5_DIR,
        ("well", "data_well"): WELL_DATA_DIR,
        ("well", "upload_well"): UPLOAD_WELL_DIR,
    }
    root = roots.get((expected_kind, source))
    if root is None:
        return None
    path = (root / filename).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return None
    return path if path.exists() and path.is_file() else None


def list_available_h5_files() -> list[dict]:
    """List existing HDF5 files from data and upload directories."""
    return _list_files(
        [
            ("data_h5", HDF5_DATA_DIR),
            ("upload_h5", UPLOAD_H5_DIR),
        ],
        suffixes={".h5", ".hdf5"},
    )


def list_available_well_files() -> list[dict]:
    """List existing well trajectory files from data and upload directories."""
    return _list_files(
        [
            ("data_well", WELL_DATA_DIR),
            ("upload_well", UPLOAD_WELL_DIR),
        ],
        suffixes=None,
    )


def _list_files(sources: list[tuple[str, Path]], suffixes: set[str] | None) -> list[dict]:
    """List files under allowed source directories."""
    files = []
    for source, root in sources:
        if not root.exists():
            continue
        for path in sorted(root.iterdir(), key=lambda item: item.name.lower()):
            if not path.is_file():
                continue
            if suffixes is not None and path.suffix.lower() not in suffixes:
                continue
            files.append(
                {
                    "file_id": make_file_id(source, path.name),
                    "filename": path.name,
                    "source": source,
                    "size": path.stat().st_size,
                }
            )
    return files


def get_output_path() -> Path:
    """Return the directory used to store generated animation files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR
