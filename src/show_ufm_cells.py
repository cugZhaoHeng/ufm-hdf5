# -*- coding: utf-8 -*-
from pathlib import Path
import re
import sys

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
HDF5_FILE = PROJECT_DIR / "data" / "hdf5_file"
WELL_DATA_DIR = PROJECT_DIR / "data" / "well_data"
OUT_PUT_DIR = PROJECT_DIR / "output"

project_root_str = str(PROJECT_DIR)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from utils.date_util import get_current_time
from utils.logger import create_logger

logger = create_logger(__name__)

INVALID_LIMIT = 1.0e30


def list_fracture_simulation_groups(h5_file):
    """Return Fracture Simulation groups sorted by their numeric id."""
    pattern = re.compile(r"^Fracture Simulation\s+(\d+)$")
    groups = []
    for name in h5_file.keys():
        match = pattern.match(name)
        if match:
            groups.append((int(match.group(1)), name))
    return [name for _, name in sorted(groups)]


def select_stage_groups(all_groups, show_all=True, stage_indices=None, stage_ids=None):
    """Select simulation groups by list index or by numeric simulation id."""
    if show_all:
        return all_groups

    if stage_ids is not None:
        id_set = {int(stage_id) for stage_id in stage_ids}
        selected = []
        for group_name in all_groups:
            sim_id = int(group_name.split()[-1])
            if sim_id in id_set:
                selected.append(group_name)
        return selected

    if stage_indices is None:
        stage_indices = [0]

    selected = []
    for idx in stage_indices:
        real_idx = idx if idx >= 0 else len(all_groups) + idx
        if 0 <= real_idx < len(all_groups):
            selected.append(all_groups[real_idx])
    return selected


def safe_values(values):
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values) & (np.abs(values) < INVALID_LIMIT)
    return values[mask]


def clean_invalid(values):
    values = np.asarray(values, dtype=float)
    return np.where(np.isfinite(values) & (np.abs(values) < INVALID_LIMIT), values, np.nan)


def read_well_trajectory(well_file_path, z_column="TVD", tail_count=None):
    df = pd.read_csv(well_file_path, sep=r"\s+", comment="#")
    if tail_count is not None:
        df = df.tail(tail_count)
    return df["X"].values, df["Y"].values, df[z_column].values


def split_element_to_cell_quads(points4, n_cells):
    """Split one quadrilateral element into n cell-level quadrilaterals."""
    p1, p2, p3, p4 = points4
    quads = []
    for cell_idx in range(n_cells):
        eta0 = cell_idx / n_cells
        eta1 = (cell_idx + 1) / n_cells

        left0 = (1.0 - eta0) * p1 + eta0 * p4
        right0 = (1.0 - eta0) * p2 + eta0 * p3
        right1 = (1.0 - eta1) * p2 + eta1 * p3
        left1 = (1.0 - eta1) * p1 + eta1 * p4
        quads.append(np.vstack([left0, right0, right1, left1]))
    return quads


def read_stage_cell_data(h5_file, group_name, cell_property_name):
    base_path = f"{group_name}/Results/Bulk/UFM/FractureSet"
    points_path = f"{base_path}/Elements/Points"
    cells_path = f"{base_path}/Elements/Cells"

    required_paths = [
        f"{points_path}/X",
        f"{points_path}/Y",
        f"{points_path}/Z",
        f"{cells_path}/{cell_property_name}",
    ]
    missing = [path for path in required_paths if path not in h5_file]
    if missing:
        raise KeyError(f"missing required paths: {missing}")

    x = clean_invalid(h5_file[f"{points_path}/X"][:])
    y = clean_invalid(h5_file[f"{points_path}/Y"][:])
    z = clean_invalid(h5_file[f"{points_path}/Z"][:])
    cell_property = clean_invalid(h5_file[f"{cells_path}/{cell_property_name}"][:])

    if cell_property.ndim != 3:
        raise ValueError(
            f"{cell_property_name} should be a 3D array: time x element x cell"
        )

    element_count_path = f"{base_path}/ElementCount"
    if element_count_path in h5_file:
        element_count = np.asarray(h5_file[element_count_path][:]).reshape(-1).astype(int)
    else:
        element_count = np.full(x.shape[0], x.shape[1], dtype=int)

    n_time = min(x.shape[0], cell_property.shape[0], len(element_count))
    n_element_max = min(x.shape[1], cell_property.shape[1])

    return {
        "id": group_name,
        "X": x[:n_time, :n_element_max, :],
        "Y": y[:n_time, :n_element_max, :],
        "Z": z[:n_time, :n_element_max, :],
        "CellProperty": cell_property[:n_time, :n_element_max, :],
        "ElementCount": element_count[:n_time],
        "n_time": n_time,
        "n_element_max": n_element_max,
        "n_cells": cell_property.shape[2],
    }


def compute_vmin_vmax(values):
    values = safe_values(values)
    if values.size == 0:
        raise ValueError("No valid cell property values found.")

    raw_min = float(np.nanmin(values))
    raw_max = float(np.nanmax(values))
    if np.isclose(raw_min, raw_max):
        delta = 0.5 if np.isclose(raw_min, 0.0) else abs(raw_min) * 0.05
        return raw_min - delta, raw_max + delta
    return raw_min, raw_max


def build_cell_verts_for_stage(sim_data, time_step, cmap, norm, alpha=0.85):
    n_active = min(
        int(sim_data["ElementCount"][time_step]),
        sim_data["n_element_max"],
    )
    x_t = sim_data["X"][time_step, :n_active, :]
    y_t = sim_data["Y"][time_step, :n_active, :]
    z_t = sim_data["Z"][time_step, :n_active, :]
    prop_t = sim_data["CellProperty"][time_step, :n_active, :]

    verts = []
    face_colors = []

    for element_idx in range(n_active):
        points4 = np.column_stack([x_t[element_idx], y_t[element_idx], z_t[element_idx]])
        if np.isnan(points4).any():
            continue

        cell_values = prop_t[element_idx]
        if np.all(~np.isfinite(cell_values)):
            continue

        cell_quads = split_element_to_cell_quads(points4, sim_data["n_cells"])
        for cell_idx, quad in enumerate(cell_quads):
            value = cell_values[cell_idx]
            if not np.isfinite(value):
                continue

            color = list(cmap(norm(value)))
            color[3] = alpha
            verts.append(quad)
            face_colors.append(color)

    return verts, face_colors


def animate_stage_cell_fractures(
    h5_file_path,
    well_csv_path,
    cell_property_name="WidthProfile",
    keep_aspect=True,
    save_format=None,
    show_all=True,
    stage_indices=None,
    stage_ids=None,
    well_z_column="TVD",
    cmap_name="jet",
    alpha=0.85,
    edge_color="0.55",
    edge_width=0.02,
    interval=20,
    fps=30,
):
    all_sim_data = []
    global_min = np.array([np.inf, np.inf, np.inf])
    global_max = np.array([-np.inf, -np.inf, -np.inf])
    all_cell_values = []

    w_x, w_y, w_z = read_well_trajectory(well_csv_path, z_column=well_z_column)

    logger.info("Reading HDF5 cell-level UFM data...")
    with h5py.File(h5_file_path, "r") as f:
        all_groups = list_fracture_simulation_groups(f)
        target_groups = select_stage_groups(
            all_groups,
            show_all=show_all,
            stage_indices=stage_indices,
            stage_ids=stage_ids,
        )

        logger.info(f"Target stages: {target_groups}")
        for group_name in target_groups:
            try:
                sim_data = read_stage_cell_data(f, group_name, cell_property_name)
            except Exception as exc:
                logger.warning(f"Skip {group_name}: {exc}")
                continue

            all_sim_data.append(sim_data)

            x_values = safe_values(sim_data["X"])
            y_values = safe_values(sim_data["Y"])
            z_values = safe_values(sim_data["Z"])
            if x_values.size and y_values.size and z_values.size:
                global_min = np.minimum(
                    global_min,
                    [np.nanmin(x_values), np.nanmin(y_values), np.nanmin(z_values)],
                )
                global_max = np.maximum(
                    global_max,
                    [np.nanmax(x_values), np.nanmax(y_values), np.nanmax(z_values)],
                )

            all_cell_values.append(sim_data["CellProperty"].reshape(-1))

    if not all_sim_data:
        logger.error("No valid stage cell data found.")
        return
    if not np.all(np.isfinite(global_min)) or not np.all(np.isfinite(global_max)):
        logger.error("No valid fracture coordinates found.")
        return

    prop_min, prop_max = compute_vmin_vmax(np.concatenate(all_cell_values))
    cmap = mpl.colormaps[cmap_name]
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
    cbar.set_label(cell_property_name)

    poly_collections = []
    for _ in all_sim_data:
        collection = Poly3DCollection(
            [],
            alpha=alpha,
            edgecolors=edge_color,
            linewidths=edge_width,
        )
        ax.add_collection3d(collection)
        poly_collections.append(collection)

    title = ax.set_title(f"Fracture Cell {cell_property_name} Evolution")
    text_info = ax.text2D(
        0,
        0,
        "",
        transform=ax.transAxes,
        fontsize=10.5,
        color="#1f6fd1",
        va="bottom",
        ha="left",
        bbox=dict(
            boxstyle="round,pad=0.45",
            facecolor="#eef6ff",
            edgecolor="#1f6fd1",
            alpha=0.94,
        ),
    )

    frames_map = []
    for sim_idx, sim_data in enumerate(all_sim_data):
        for time_step in range(sim_data["n_time"]):
            frames_map.append((sim_idx, time_step))

    finalized_sims = set()

    def set_collection_at_time(sim_idx, time_step):
        verts, colors = build_cell_verts_for_stage(
            all_sim_data[sim_idx],
            time_step,
            cmap=cmap,
            norm=norm,
            alpha=alpha,
        )
        poly_collections[sim_idx].set_verts(verts)
        poly_collections[sim_idx].set_facecolor(colors)
        return len(verts)

    def update(frame_idx):
        curr_sim_idx, curr_time_step = frames_map[frame_idx]
        sim_data = all_sim_data[curr_sim_idx]

        patch_count = set_collection_at_time(curr_sim_idx, curr_time_step)

        for prev_idx in range(curr_sim_idx):
            if prev_idx not in finalized_sims:
                set_collection_at_time(prev_idx, all_sim_data[prev_idx]["n_time"] - 1)
                finalized_sims.add(prev_idx)

        step_str = f"{curr_time_step + 1}/{sim_data['n_time']}"
        title.set_text(
            f"Cell {cell_property_name} Evolution | Stage: {sim_data['id']} | Step: {step_str}"
        )
        text_info.set_text(
            "\n".join(
                [
                    f"Current Stage: {sim_data['id']}",
                    f"Active Elements: {sim_data['ElementCount'][curr_time_step]}",
                    f"Cells per Element: {sim_data['n_cells']}",
                    f"Rendered Cell Patches: {patch_count}",
                ]
            )
        )

        return poly_collections + [title, text_info]

    ani = FuncAnimation(
        fig,
        update,
        frames=len(frames_map),
        interval=interval,
        blit=False,
        repeat=False,
    )

    if save_format:
        OUT_PUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = (
            OUT_PUT_DIR
            / f"fracture_cell_{cell_property_name}_evolution_{get_current_time()}.{save_format}"
        )
        logger.info(f"Saving animation to: {output_path}")
        if save_format == "mp4":
            ani.save(output_path, writer="ffmpeg", fps=fps)
        elif save_format == "gif":
            ani.save(output_path, writer="pillow", fps=fps)
        else:
            raise ValueError("save_format only supports 'gif', 'mp4', or None")
        logger.info(f"Animation saved: {output_path}")

    plt.show()


if __name__ == "__main__":
    h5_file_path = HDF5_FILE / "JY68-4HF.h5"
    well_csv_path = WELL_DATA_DIR / "68-4HF"

    animate_stage_cell_fractures(
        h5_file_path=h5_file_path,
        well_csv_path=well_csv_path,
        cell_property_name="WidthProfile",
        keep_aspect=False,
        save_format="gif",
        show_all=False,
        stage_indices=[0],
    )
