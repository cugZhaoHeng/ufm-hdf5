"""Final-state UFM snapshot extraction for browser-side 3D preview."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import h5py
import numpy as np

from backend.services.animation import read_well_trajectory, split_element_to_cell_quads
from backend.services.hdf5_inspector import clean_invalid, safe_values


def _value_range(values: list[float]) -> tuple[float | None, float | None]:
    """Return min/max for a list of numeric values."""
    if not values:
        return None, None
    arr = safe_values(np.asarray(values, dtype=float))
    if arr.size == 0:
        return None, None
    return float(np.nanmin(arr)), float(np.nanmax(arr))


def _axis_bounds(points: list[list], well_points: list[list], quads: list[list]) -> dict:
    """Return padded x/y/z axis bounds for 3D rendering."""
    xyz = []
    xyz.extend([point[:3] for point in points])
    xyz.extend(well_points)
    for quad in quads:
        xyz.extend(quad[:4])
    if not xyz:
        return {}

    arr = np.asarray(xyz, dtype=float)
    bounds = {}
    for axis, index in (("x", 0), ("y", 1), ("z", 2)):
        values = safe_values(arr[:, index])
        if values.size == 0:
            continue
        raw_min = float(np.nanmin(values))
        raw_max = float(np.nanmax(values))
        span = raw_max - raw_min
        pad = max(span * 0.05, 1.0)
        if np.isclose(span, 0.0):
            pad = max(abs(raw_min) * 0.001, 1.0)
        bounds[axis] = {
            "min": raw_min - pad,
            "max": raw_max + pad,
            "raw_min": raw_min,
            "raw_max": raw_max,
        }
    return bounds


def _element_final_geometry(h5_file: h5py.File, group_name: str, property_name: str) -> tuple[list[list], list[list]]:
    """Build final-time element-center points colored by an element property."""
    base_path = f"{group_name}/Results/Bulk/UFM/FractureSet/Elements"
    points_path = f"{base_path}/Points"
    required = [
        f"{points_path}/X",
        f"{points_path}/Y",
        f"{points_path}/Z",
        f"{base_path}/{property_name}",
    ]
    missing = [path for path in required if path not in h5_file]
    if missing:
        raise KeyError(f"missing required paths: {missing}")

    x_ds = h5_file[f"{points_path}/X"]
    prop_ds = h5_file[f"{base_path}/{property_name}"]
    time_step = min(x_ds.shape[0], prop_ds.shape[0] if prop_ds.ndim > 1 else x_ds.shape[0]) - 1

    x_t = clean_invalid(h5_file[f"{points_path}/X"][time_step])
    y_t = clean_invalid(h5_file[f"{points_path}/Y"][time_step])
    z_t = clean_invalid(h5_file[f"{points_path}/Z"][time_step])
    prop = clean_invalid(prop_ds[:] if prop_ds.ndim == 1 else prop_ds[time_step])

    n_element = min(x_t.shape[0], prop.shape[0])
    points = []
    quads = []
    for element_idx in range(n_element):
        coords = np.column_stack([x_t[element_idx], y_t[element_idx], z_t[element_idx]])
        value = prop[element_idx]
        if np.isnan(coords).any() or not np.isfinite(value):
            continue
        center = np.nanmean(coords, axis=0)
        points.append(
            [
                float(center[0]),
                float(center[1]),
                float(center[2]),
                float(value),
                group_name,
                int(element_idx),
            ]
        )
        quads.append(
            [
                [float(coord[0]), float(coord[1]), float(coord[2])] for coord in coords
            ]
            + [float(value), group_name, int(element_idx)]
        )
    return points, quads


def _cell_final_geometry(h5_file: h5py.File, group_name: str, property_name: str) -> tuple[list[list], list[list]]:
    """Build final-time cell-center points colored by a cell property."""
    base_path = f"{group_name}/Results/Bulk/UFM/FractureSet"
    points_path = f"{base_path}/Elements/Points"
    cells_path = f"{base_path}/Elements/Cells"
    required = [
        f"{points_path}/X",
        f"{points_path}/Y",
        f"{points_path}/Z",
        f"{cells_path}/{property_name}",
    ]
    missing = [path for path in required if path not in h5_file]
    if missing:
        raise KeyError(f"missing required paths: {missing}")

    x_ds = h5_file[f"{points_path}/X"]
    prop_ds = h5_file[f"{cells_path}/{property_name}"]
    n_time = min(x_ds.shape[0], prop_ds.shape[0])
    time_step = n_time - 1

    x_t = clean_invalid(h5_file[f"{points_path}/X"][time_step])
    y_t = clean_invalid(h5_file[f"{points_path}/Y"][time_step])
    z_t = clean_invalid(h5_file[f"{points_path}/Z"][time_step])
    prop_t = clean_invalid(prop_ds[time_step])
    cell_count = int(prop_t.shape[1])

    element_count_path = f"{base_path}/ElementCount"
    if element_count_path in h5_file:
        element_counts = np.asarray(h5_file[element_count_path][:]).reshape(-1)
        element_count = int(element_counts[min(time_step, len(element_counts) - 1)])
    else:
        element_count = x_t.shape[0]

    n_element = min(element_count, x_t.shape[0], prop_t.shape[0])
    points = []
    quads = []
    for element_idx in range(n_element):
        coords = np.column_stack([x_t[element_idx], y_t[element_idx], z_t[element_idx]])
        if np.isnan(coords).any():
            continue

        cell_values = prop_t[element_idx]
        if np.all(~np.isfinite(cell_values)):
            continue

        cell_quads = split_element_to_cell_quads(coords, cell_count)
        for cell_idx, quad in enumerate(cell_quads):
            value = cell_values[cell_idx]
            if not np.isfinite(value):
                continue
            center = np.nanmean(quad, axis=0)
            points.append(
                [
                    float(center[0]),
                    float(center[1]),
                    float(center[2]),
                    float(value),
                    group_name,
                    int(element_idx),
                    int(cell_idx),
                ]
            )
            quads.append(
                [
                    [float(coord[0]), float(coord[1]), float(coord[2])] for coord in quad
                ]
                + [float(value), group_name, int(element_idx), int(cell_idx)]
            )
    return points, quads


def build_final_state_snapshot(
    h5_file_path: str | Path,
    group_names: list[str],
    level: Literal["element", "cell"],
    property_name: str,
    well_file_path: str | Path | None = None,
) -> dict:
    """Return final-time 3D point data for all selected UFM groups.

    The returned points are suitable for browser-side ECharts GL rendering.
    Element mode returns one point per valid element. Cell mode returns one
    point per valid cell, where the number of cells comes from the HDF5 property
    array's final dimension.
    """
    if not group_names:
        raise ValueError("At least one group is required.")

    all_points = []
    all_quads = []
    skipped_groups = []
    reader = _element_final_geometry if level == "element" else _cell_final_geometry

    with h5py.File(h5_file_path, "r") as h5_file:
        for group_name in group_names:
            try:
                points, quads = reader(h5_file, group_name, property_name)
                all_points.extend(points)
                all_quads.extend(quads)
            except Exception as exc:
                skipped_groups.append({"group": group_name, "reason": str(exc)})

    values = [point[3] for point in all_points]
    value_min, value_max = _value_range(values)

    well_points = []
    if well_file_path:
        w_x, w_y, w_z = read_well_trajectory(well_file_path)
        well_points = [
            [float(x), float(y), float(z)]
            for x, y, z in zip(w_x, w_y, w_z)
            if np.isfinite(x) and np.isfinite(y) and np.isfinite(z)
        ]

    bounds = _axis_bounds(all_points, well_points, all_quads)

    return {
        "level": level,
        "property_name": property_name,
        "groups": group_names,
        "point_count": len(all_points),
        "points": all_points,
        "quad_count": len(all_quads),
        "quads": all_quads,
        "value_min": value_min,
        "value_max": value_max,
        "bounds": bounds,
        "well": well_points,
        "skipped_groups": skipped_groups,
    }
