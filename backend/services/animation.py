"""Reusable UFM animation generation functions.

The API layer calls these functions, but they do not depend on FastAPI. They
can also be imported by external Python code after the project is packaged.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Literal

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from utils.date_util import get_current_time
from utils.logger import create_logger

from backend.services.hdf5_inspector import clean_invalid, read_group_display_name, safe_values

logger = create_logger(__name__)

ProgressCallback = Callable[..., None]

STAGE_NAME_PATTERN = re.compile(r"\bStage\s+(\d+)\b", re.IGNORECASE)


def extract_stage_label(display_name: str, fallback: str) -> str:
    """Return ``Stage N`` parsed from a simulation display name.

    Parameters
    ----------
    display_name:
        Human-readable HDF5 group name, usually from
        ``Metadata/Identification/Name``.
    fallback:
        Value returned when the display name does not contain a stage label.
    """
    match = STAGE_NAME_PATTERN.search(display_name or "")
    if not match:
        return fallback
    return f"Stage {int(match.group(1))}"


def read_simulation_display_name(h5_file: h5py.File, group_name: str) -> str:
    """Read the display name used for animation titles and overlays.

    The UFM files store the business-facing stage label in
    ``Metadata/Identification/Name``. This function checks that path first so
    the animation title does not accidentally fall back to the technical group
    id such as ``Fracture Simulation 45``.
    """
    preferred_path = f"{group_name}/Metadata/Identification/Name"
    if preferred_path in h5_file:
        from backend.services.hdf5_inspector import decode_hdf5_value

        return str(decode_hdf5_value(h5_file[preferred_path][()]))
    return read_group_display_name(h5_file[group_name])


def format_animation_title(level: str, property_name: str, stage_label: str, step_str: str) -> str:
    """Build the Matplotlib title for one animation frame."""
    return f"{stage_label} | {level.title()} {property_name} | Step: {step_str}"


def read_well_trajectory(well_file_path: str | Path, z_column: str = "TVD") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a well trajectory text/CSV file.

    Parameters
    ----------
    well_file_path:
        Path to a whitespace-delimited file containing X/Y/TVD columns.
    z_column:
        Name of the depth column. Existing project files use ``TVD``.
    """
    df = pd.read_csv(well_file_path, sep=r"\s+", comment="#")
    return df["X"].values, df["Y"].values, df[z_column].values


def compute_vmin_vmax(values: np.ndarray) -> tuple[float, float]:
    """Compute a stable color scale range for a numeric property array."""
    values = safe_values(values)
    if values.size == 0:
        raise ValueError("No valid property values found.")
    raw_min = float(np.nanmin(values))
    raw_max = float(np.nanmax(values))
    if np.isclose(raw_min, raw_max):
        delta = 0.5 if np.isclose(raw_min, 0.0) else abs(raw_min) * 0.05
        return raw_min - delta, raw_max + delta
    return raw_min, raw_max


def split_element_to_cell_quads(points4: np.ndarray, split_count: int) -> list[np.ndarray]:
    """Split one quadrilateral element into cell-level quadrilaterals.

    ``split_count`` must come from the HDF5 cell property array's last
    dimension so geometry slices match the actual cell values.
    """
    p1, p2, p3, p4 = points4
    quads = []
    for cell_idx in range(split_count):
        eta0 = cell_idx / split_count
        eta1 = (cell_idx + 1) / split_count
        left0 = (1.0 - eta0) * p1 + eta0 * p4
        right0 = (1.0 - eta0) * p2 + eta0 * p3
        right1 = (1.0 - eta1) * p2 + eta1 * p3
        left1 = (1.0 - eta1) * p1 + eta1 * p4
        quads.append(np.vstack([left0, right0, right1, left1]))
    return quads


def read_element_stage_data(h5_file: h5py.File, group_name: str, property_name: str) -> dict:
    """Read element-level geometry and one element property for a stage."""
    display_name = read_simulation_display_name(h5_file, group_name)
    base_path = f"{group_name}/Results/Bulk/UFM/FractureSet/Elements"
    points_path = f"{base_path}/Points"
    required_paths = [
        f"{points_path}/X",
        f"{points_path}/Y",
        f"{points_path}/Z",
        f"{base_path}/{property_name}",
    ]
    missing = [path for path in required_paths if path not in h5_file]
    if missing:
        raise KeyError(f"missing required paths: {missing}")

    x = clean_invalid(h5_file[f"{points_path}/X"][:])
    y = clean_invalid(h5_file[f"{points_path}/Y"][:])
    z = clean_invalid(h5_file[f"{points_path}/Z"][:])
    prop = clean_invalid(h5_file[f"{base_path}/{property_name}"][:])
    if prop.ndim == 1:
        prop = np.repeat(prop.reshape(1, -1), x.shape[0], axis=0)
    if prop.ndim != 2:
        raise ValueError(f"{property_name} should be a 2D array: time x element")

    n_time = min(x.shape[0], prop.shape[0])
    n_element = min(x.shape[1], prop.shape[1])
    return {
        "id": group_name,
        "display_name": display_name,
        "stage_label": extract_stage_label(display_name, group_name),
        "X": x[:n_time, :n_element, :],
        "Y": y[:n_time, :n_element, :],
        "Z": z[:n_time, :n_element, :],
        "Property": prop[:n_time, :n_element],
        "n_time": n_time,
        "n_element_max": n_element,
    }


def apply_display_name(sim_data: dict, display_name: str | None) -> dict:
    """Apply a Name value and derived Stage label to loaded simulation data."""
    if display_name:
        sim_data["display_name"] = display_name
        sim_data["stage_label"] = extract_stage_label(display_name, sim_data["id"])
    return sim_data


def read_cell_stage_data(h5_file: h5py.File, group_name: str, property_name: str) -> dict:
    """Read element geometry and one cell-level property for a stage."""
    display_name = read_simulation_display_name(h5_file, group_name)
    base_path = f"{group_name}/Results/Bulk/UFM/FractureSet"
    points_path = f"{base_path}/Elements/Points"
    cells_path = f"{base_path}/Elements/Cells"
    required_paths = [
        f"{points_path}/X",
        f"{points_path}/Y",
        f"{points_path}/Z",
        f"{cells_path}/{property_name}",
    ]
    missing = [path for path in required_paths if path not in h5_file]
    if missing:
        raise KeyError(f"missing required paths: {missing}")

    x = clean_invalid(h5_file[f"{points_path}/X"][:])
    y = clean_invalid(h5_file[f"{points_path}/Y"][:])
    z = clean_invalid(h5_file[f"{points_path}/Z"][:])
    prop = clean_invalid(h5_file[f"{cells_path}/{property_name}"][:])
    if prop.ndim != 3:
        raise ValueError(f"{property_name} should be a 3D array: time x element x cell")

    element_count_path = f"{base_path}/ElementCount"
    if element_count_path in h5_file:
        element_count = np.asarray(h5_file[element_count_path][:]).reshape(-1).astype(int)
    else:
        element_count = np.full(x.shape[0], x.shape[1], dtype=int)

    n_time = min(x.shape[0], prop.shape[0], len(element_count))
    n_element = min(x.shape[1], prop.shape[1])
    return {
        "id": group_name,
        "display_name": display_name,
        "stage_label": extract_stage_label(display_name, group_name),
        "X": x[:n_time, :n_element, :],
        "Y": y[:n_time, :n_element, :],
        "Z": z[:n_time, :n_element, :],
        "Property": prop[:n_time, :n_element, :],
        "ElementCount": element_count[:n_time],
        "n_time": n_time,
        "n_element_max": n_element,
        "n_cells": prop.shape[2],
    }


def build_element_verts_for_stage(sim_data: dict, time_step: int, cmap, norm, alpha: float) -> tuple[list, list]:
    """Build 3D quadrilateral faces and colors for an element-level frame."""
    x_t = sim_data["X"][time_step]
    y_t = sim_data["Y"][time_step]
    z_t = sim_data["Z"][time_step]
    prop_t = sim_data["Property"][time_step]
    valid = np.where(~np.isnan(x_t).any(axis=1))[0]
    verts = []
    face_colors = []
    for element_idx in valid:
        value = prop_t[element_idx]
        if not np.isfinite(value):
            continue
        quad = np.stack([x_t[element_idx], y_t[element_idx], z_t[element_idx]], axis=-1)
        color = list(cmap(norm(value)))
        color[3] = alpha
        verts.append(quad)
        face_colors.append(color)
    return verts, face_colors


def build_cell_verts_for_stage(
    sim_data: dict,
    time_step: int,
    cmap,
    norm,
    alpha: float,
) -> tuple[list, list]:
    """Build 3D faces and colors for a cell-level frame."""
    n_active = min(int(sim_data["ElementCount"][time_step]), sim_data["n_element_max"])
    cell_count = int(sim_data["n_cells"])
    x_t = sim_data["X"][time_step, :n_active, :]
    y_t = sim_data["Y"][time_step, :n_active, :]
    z_t = sim_data["Z"][time_step, :n_active, :]
    prop_t = sim_data["Property"][time_step, :n_active, :]
    verts = []
    face_colors = []

    for element_idx in range(n_active):
        points4 = np.column_stack([x_t[element_idx], y_t[element_idx], z_t[element_idx]])
        if np.isnan(points4).any():
            continue

        cell_values = prop_t[element_idx]
        if np.all(~np.isfinite(cell_values)):
            continue

        cell_quads = split_element_to_cell_quads(points4, cell_count)
        for cell_idx, quad in enumerate(cell_quads):
            value = cell_values[cell_idx]
            if not np.isfinite(value):
                continue
            color = list(cmap(norm(value)))
            color[3] = alpha
            verts.append(quad)
            face_colors.append(color)
    return verts, face_colors


def generate_ufm_animation(
    h5_file_path: str | Path,
    group_names: list[str],
    level: Literal["element", "cell"],
    property_name: str,
    output_dir: str | Path,
    group_display_names: dict[str, str] | None = None,
    output_format: Literal["gif", "mp4"] = "gif",
    well_file_path: str | Path | None = None,
    fps: int = 20,
    interval: int = 50,
    keep_aspect: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> Path:
    """Generate a UFM fracture animation and return the output file path.

    Parameters
    ----------
    h5_file_path:
        Uploaded UFM HDF5 file path.
    group_names:
        Ordered ``Fracture Simulation`` groups selected by the user.
    level:
        ``element`` uses datasets directly under ``Elements``; ``cell`` uses
        datasets under ``Elements/Cells`` and splits each element into cells.
    property_name:
        HDF5 dataset name used to color the fracture faces.
    output_dir:
        Directory where the generated animation is written.
    group_display_names:
        Optional ``Fracture Simulation`` group to metadata ``Name`` mapping.
        When provided, the title extracts ``Stage N`` from this value.
    progress_callback:
        Optional callback used by the web API to report current group progress.
    """
    if not group_names:
        raise ValueError("At least one group must be selected.")
    if output_format not in {"gif", "mp4"}:
        raise ValueError("output_format only supports 'gif' or 'mp4'.")

    reader = read_element_stage_data if level == "element" else read_cell_stage_data
    all_sim_data = []
    global_min = np.array([np.inf, np.inf, np.inf])
    global_max = np.array([-np.inf, -np.inf, -np.inf])
    all_property_values = []

    if progress_callback:
        progress_callback(percent=1, total_groups=len(group_names), message="Reading HDF5 data.")

    with h5py.File(h5_file_path, "r") as h5_file:
        for group_index, group_name in enumerate(group_names, start=1):
            try:
                sim_data = reader(h5_file, group_name, property_name)
                sim_data = apply_display_name(sim_data, (group_display_names or {}).get(group_name))
            except Exception as exc:
                logger.warning(f"Skip {group_name}: {exc}")
                continue

            all_sim_data.append(sim_data)
            for axis_name, pos in (("X", 0), ("Y", 1), ("Z", 2)):
                values = safe_values(sim_data[axis_name])
                if values.size:
                    global_min[pos] = min(global_min[pos], float(np.nanmin(values)))
                    global_max[pos] = max(global_max[pos], float(np.nanmax(values)))
            all_property_values.append(sim_data["Property"].reshape(-1))
            if progress_callback:
                progress_callback(
                    percent=max(1, int(group_index / len(group_names) * 15)),
                    current_group=group_name,
                    current_group_index=group_index,
                    total_groups=len(group_names),
                    message=f"Loaded {group_name} as {sim_data['stage_label']}.",
                )

    if not all_sim_data:
        raise ValueError("No valid stage data found for the selected groups/property.")
    if not np.all(np.isfinite(global_min)) or not np.all(np.isfinite(global_max)):
        raise ValueError("No valid fracture coordinates found.")

    prop_min, prop_max = compute_vmin_vmax(np.concatenate(all_property_values))
    cmap = mpl.colormaps["jet"]
    norm = mpl.colors.Normalize(vmin=prop_min, vmax=prop_max, clip=True)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection="3d")
    span = global_max - global_min
    pad = np.maximum(span * 0.04, 1.0)
    global_min = global_min - pad
    global_max = global_max + pad
    ax.set_xlim(global_min[0], global_max[0])
    ax.set_ylim(global_min[1], global_max[1])
    ax.set_zlim(global_min[2], global_max[2])
    ax.invert_zaxis()
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Depth (m)")

    if well_file_path:
        w_x, w_y, w_z = read_well_trajectory(well_file_path)
        ax.plot(w_x, w_y, w_z, color="black", linewidth=2.5, label="Well Trajectory", zorder=10)
        ax.scatter(w_x[0], w_y[0], w_z[0], color="green", s=50, label="Well Head")
        ax.legend(loc="upper left")

    if keep_aspect:
        try:
            ax.set_box_aspect(
                (
                    np.ptp([global_min[0], global_max[0]]) + 1,
                    np.ptp([global_min[1], global_max[1]]) + 1,
                    np.ptp([global_min[2], global_max[2]]) + 1,
                )
            )
        except Exception as exc:
            logger.warning(f"Set box aspect failed, skipped: {exc}")

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label(property_name)

    poly_collections = []
    edge_color = "none" if level == "element" else "0.55"
    edge_width = 0 if level == "element" else 0.02
    alpha = 0.7 if level == "element" else 0.85
    for _ in all_sim_data:
        collection = Poly3DCollection([], alpha=alpha, edgecolors=edge_color, linewidths=edge_width)
        ax.add_collection3d(collection)
        poly_collections.append(collection)

    first_sim = all_sim_data[0]
    first_step = f"1/{first_sim['n_time']}"
    title = ax.set_title(format_animation_title(level, property_name, first_sim["stage_label"], first_step))
    text_info = ax.text2D(
        0,
        0,
        "",
        transform=ax.transAxes,
        fontsize=10.5,
        color="#1f6fd1",
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#eef6ff", edgecolor="#1f6fd1", alpha=0.94),
    )

    frames_map = [
        (sim_idx, time_step)
        for sim_idx, sim_data in enumerate(all_sim_data)
        for time_step in range(sim_data["n_time"])
    ]
    finalized_sims = set()
    frame_count = max(1, len(frames_map))

    def set_collection_at_time(sim_idx: int, time_step: int) -> int:
        sim_data = all_sim_data[sim_idx]
        if level == "element":
            verts, colors = build_element_verts_for_stage(sim_data, time_step, cmap, norm, alpha)
        else:
            verts, colors = build_cell_verts_for_stage(
                sim_data,
                time_step,
                cmap,
                norm,
                alpha,
            )
        poly_collections[sim_idx].set_verts(verts)
        poly_collections[sim_idx].set_facecolor(colors)
        return len(verts)

    def update(frame_idx: int):
        curr_sim_idx, curr_time_step = frames_map[frame_idx]
        sim_data = all_sim_data[curr_sim_idx]
        patch_count = set_collection_at_time(curr_sim_idx, curr_time_step)

        for prev_idx in range(curr_sim_idx):
            if prev_idx not in finalized_sims:
                set_collection_at_time(prev_idx, all_sim_data[prev_idx]["n_time"] - 1)
                finalized_sims.add(prev_idx)

        step_str = f"{curr_time_step + 1}/{sim_data['n_time']}"
        title.set_text(format_animation_title(level, property_name, sim_data["stage_label"], step_str))
        text_lines = [
            f"Current Group: {sim_data['id']}",
            f"Name: {sim_data['display_name']}",
            f"Rendered Patches: {patch_count}",
        ]
        if level == "cell":
            text_lines.append(f"Cells per Element: {sim_data['n_cells']}")
        text_info.set_text("\n".join(text_lines))

        if progress_callback:
            percent = 15 + int((frame_idx + 1) / frame_count * 84)
            progress_callback(
                percent=percent,
                current_group=sim_data["id"],
                current_group_index=curr_sim_idx + 1,
                total_groups=len(all_sim_data),
                message=f"Rendering {sim_data['stage_label']} ({sim_data['id']}) frame {step_str}.",
            )

        return poly_collections + [title, text_info]

    ani = FuncAnimation(fig, update, frames=len(frames_map), interval=interval, blit=False, repeat=False)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"ufm_{level}_{property_name}_{get_current_time()}.{output_format}"

    if progress_callback:
        progress_callback(percent=15, message="Saving animation.")
    try:
        if output_format == "mp4":
            ani.save(output_path, writer="ffmpeg", fps=fps)
        else:
            ani.save(output_path, writer="pillow", fps=fps)
    finally:
        plt.close(fig)

    logger.info(f"Animation saved: {output_path}")
    return output_path
