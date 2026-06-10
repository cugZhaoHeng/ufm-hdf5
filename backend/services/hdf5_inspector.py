"""HDF5 metadata inspection functions for UFM files.

These functions are intentionally independent from FastAPI so they can be
imported later by scripts, notebooks, or a packaged Python library.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np

INVALID_LIMIT = 1.0e30
FRACTURE_GROUP_PATTERN = re.compile(r"^Fracture Simulation\s+(\d+)$")


def decode_hdf5_value(value: Any) -> Any:
    """Convert HDF5 scalar/bytes values into JSON-friendly Python values."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return decode_hdf5_value(value.item())
        if value.size == 1:
            return decode_hdf5_value(value.reshape(-1)[0])
        return [decode_hdf5_value(item) for item in value.reshape(-1).tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def list_fracture_simulation_groups(h5_file: h5py.File) -> list[str]:
    """Return all top-level ``Fracture Simulation N`` groups sorted by N."""
    groups: list[tuple[int, str]] = []
    for name in h5_file.keys():
        match = FRACTURE_GROUP_PATTERN.match(name)
        if match:
            groups.append((int(match.group(1)), name))
    return [name for _, name in sorted(groups)]


def read_group_display_name(group: h5py.Group) -> str:
    """Read the best available display name for a simulation group.

    The function first checks group attributes named ``Name`` or ``name``. If
    they do not exist, it searches direct children and descendants whose final
    path component is ``Name``. If no name is found, the group path is returned.
    """
    for key in ("Name", "name"):
        if key in group.attrs:
            return str(decode_hdf5_value(group.attrs[key]))

    for key in ("Name", "name"):
        if key in group and isinstance(group[key], h5py.Dataset):
            return str(decode_hdf5_value(group[key][()]))

    found: list[str] = []

    def visitor(name: str, obj: h5py.Dataset | h5py.Group) -> None:
        if found:
            return
        if isinstance(obj, h5py.Dataset) and name.split("/")[-1].lower() == "name":
            found.append(str(decode_hdf5_value(obj[()])))

    group.visititems(visitor)
    return found[0] if found else group.name.rsplit("/", 1)[-1]


def list_dataset_names(group: h5py.Group) -> list[str]:
    """Return dataset names directly below a group, excluding coordinate folders."""
    names: list[str] = []
    for name, obj in group.items():
        if isinstance(obj, h5py.Dataset):
            names.append(name)
    return sorted(names)


def inspect_hdf5_file(h5_file_path: str | Path) -> dict[str, Any]:
    """Inspect a UFM HDF5 file and return groups plus available properties.

    Returns
    -------
    dict
        A JSON-friendly dictionary containing simulation groups, element-level
        property names, and cell-level property names.
    """
    h5_file_path = Path(h5_file_path)
    groups_info = []
    element_properties: set[str] = set()
    cell_properties: set[str] = set()

    with h5py.File(h5_file_path, "r") as h5_file:
        group_names = list_fracture_simulation_groups(h5_file)
        for index, group_name in enumerate(group_names):
            group = h5_file[group_name]
            base_path = f"{group_name}/Results/Bulk/UFM/FractureSet"
            elements_path = f"{base_path}/Elements"
            cells_path = f"{elements_path}/Cells"

            time_steps = 0
            element_count = 0
            has_geometry = False

            if f"{elements_path}/Points/X" in h5_file:
                x_dataset = h5_file[f"{elements_path}/Points/X"]
                has_geometry = True
                if len(x_dataset.shape) >= 2:
                    time_steps = int(x_dataset.shape[0])
                    element_count = int(x_dataset.shape[1])

            if elements_path in h5_file:
                element_properties.update(list_dataset_names(h5_file[elements_path]))

            if cells_path in h5_file:
                cell_properties.update(list_dataset_names(h5_file[cells_path]))

            groups_info.append(
                {
                    "index": index,
                    "group": group_name,
                    "name": read_group_display_name(group),
                    "simulation_id": int(group_name.split()[-1]),
                    "time_steps": time_steps,
                    "element_count": element_count,
                    "has_geometry": has_geometry,
                }
            )

    return {
        "groups": groups_info,
        "element_properties": sorted(element_properties),
        "cell_properties": sorted(cell_properties),
    }


def clean_invalid(values: np.ndarray) -> np.ndarray:
    """Replace non-finite and UFM sentinel values with NaN."""
    values = np.asarray(values, dtype=float)
    return np.where(np.isfinite(values) & (np.abs(values) < INVALID_LIMIT), values, np.nan)


def safe_values(values: np.ndarray) -> np.ndarray:
    """Return finite numeric values, excluding UFM sentinel values."""
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values) & (np.abs(values) < INVALID_LIMIT)
    return values[mask]
