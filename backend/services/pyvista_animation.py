"""PyVista/VTK based UFM animation generation.

This module is an optional rendering backend. It is intentionally imported only
when requested because PyVista/VTK are heavy optional dependencies and may need
server-side OpenGL/EGL setup.
"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Callable, Literal

import h5py
import numpy as np

from backend.services.animation import (
    apply_display_name,
    build_sampled_time_steps,
    compute_vmin_vmax,
    extract_stage_label,
    format_animation_title,
    read_cell_stage_data,
    read_element_stage_data,
    read_well_trajectory,
    split_element_to_cell_quads,
)
from backend.services.hdf5_inspector import safe_values
from utils.date_util import get_current_time
from utils.logger import create_logger

logger = create_logger(__name__)
ProgressCallback = Callable[..., None]


def _require_pyvista():
    """Import PyVista lazily and raise a helpful dependency error."""
    try:
        import pyvista as pv
    except ImportError as exc:
        raise RuntimeError(
            "PyVista/VTK renderer requires optional dependencies. "
            "Install them with: pip install pyvista vtk imageio imageio-ffmpeg"
        ) from exc
    return pv


def _quad_mesh_from_arrays(quads: list[np.ndarray], values: list[float], pv):
    """Create a PyVista PolyData mesh from quadrilateral faces and scalars."""
    if not quads:
        return pv.PolyData()

    points = np.asarray(quads, dtype=np.float64).reshape(-1, 3)
    face_indices = np.arange(points.shape[0], dtype=np.int64).reshape(-1, 4)
    faces = np.column_stack([np.full(len(face_indices), 4, dtype=np.int64), face_indices]).ravel()
    mesh = pv.PolyData(points, faces)
    mesh.cell_data["value"] = np.asarray(values, dtype=np.float32)
    return mesh


def _element_mesh_for_stage(sim_data: dict, time_step: int, pv):
    """Build one PyVista mesh for element-level faces at one time step."""
    x_t = sim_data["X"][time_step]
    y_t = sim_data["Y"][time_step]
    z_t = sim_data["Z"][time_step]
    prop_t = sim_data["Property"][time_step]
    quads: list[np.ndarray] = []
    values: list[float] = []

    valid = np.where(~np.isnan(x_t).any(axis=1))[0]
    for element_idx in valid:
        value = float(prop_t[element_idx])
        if not np.isfinite(value):
            continue
        quads.append(np.stack([x_t[element_idx], y_t[element_idx], z_t[element_idx]], axis=-1))
        values.append(value)
    return _quad_mesh_from_arrays(quads, values, pv)


def _cell_mesh_for_stage(sim_data: dict, time_step: int, pv):
    """Build one PyVista mesh for cell-level faces at one time step."""
    n_active = min(int(sim_data["ElementCount"][time_step]), sim_data["n_element_max"])
    cell_count = int(sim_data["n_cells"])
    x_t = sim_data["X"][time_step, :n_active, :]
    y_t = sim_data["Y"][time_step, :n_active, :]
    z_t = sim_data["Z"][time_step, :n_active, :]
    prop_t = sim_data["Property"][time_step, :n_active, :]
    quads: list[np.ndarray] = []
    values: list[float] = []

    for element_idx in range(n_active):
        points4 = np.column_stack([x_t[element_idx], y_t[element_idx], z_t[element_idx]])
        if np.isnan(points4).any():
            continue
        cell_values = prop_t[element_idx]
        if np.all(~np.isfinite(cell_values)):
            continue
        for cell_idx, quad in enumerate(split_element_to_cell_quads(points4, cell_count)):
            value = float(cell_values[cell_idx])
            if not np.isfinite(value):
                continue
            quads.append(quad)
            values.append(value)
    return _quad_mesh_from_arrays(quads, values, pv)


def _grid_axis_bounds(raw_min: float, raw_max: float) -> tuple[float, float, int]:
    """Return clean grid bounds and label count for a readable engineering axis."""
    if not np.isfinite(raw_min) or not np.isfinite(raw_max):
        return 0.0, 1.0, 2
    if np.isclose(raw_min, raw_max):
        raw_min -= 0.5
        raw_max += 0.5

    # Prefer 100 m style grid lines and keep every label on a 100-m multiple.
    span = float(raw_max - raw_min)
    candidate_steps = []
    for power in range(-1, 7):
        base = 100.0 * (10**power)
        for multiplier in (1.0, 2.0, 5.0):
            step = float(base * multiplier)
            if step >= 100.0:
                candidate_steps.append(step)
    candidate_steps = sorted(set(candidate_steps), key=lambda item: (abs(item - span / 4.0), item))

    best_bounds = None
    best_score = float("inf")
    for step in candidate_steps:
        lower = float(np.floor(raw_min / step) * step)
        upper = float(np.ceil(raw_max / step) * step)
        intervals = int(round((upper - lower) / step))
        while intervals + 1 < 5:
            lower_pad = abs(raw_min - lower)
            upper_pad = abs(upper - raw_max)
            if lower_pad <= upper_pad:
                lower -= step
            else:
                upper += step
            intervals += 1
        label_count = intervals + 1
        if label_count < 2:
            continue
        density_penalty = abs(label_count - 5) * 10
        step_penalty = abs(np.log10(step / 100.0)) if step > 0 else 100
        padding_penalty = ((lower - raw_min) ** 2 + (upper - raw_max) ** 2) / max(span, 1.0) ** 2
        score = density_penalty + step_penalty + padding_penalty
        if score < best_score:
            best_score = score
            best_bounds = (lower, upper, label_count)

    if best_bounds is None:
        return float(raw_min), float(raw_max), 2
    return best_bounds


def _grid_scene_bounds(
    global_min: np.ndarray,
    global_max: np.ndarray,
) -> tuple[tuple[float, float, float, float, float, float], tuple[int, int, int]]:
    """Return clean axis bounds and label counts for PyVista cube axes."""
    x0, x1, nx = _grid_axis_bounds(float(global_min[0]), float(global_max[0]))
    y0, y1, ny = _grid_axis_bounds(float(global_min[1]), float(global_max[1]))
    z0, z1, nz = _grid_axis_bounds(float(global_min[2]), float(global_max[2]))
    return (x0, x1, y0, y1, z0, z1), (nx, ny, nz)


def _configure_camera(plotter, bounds: tuple[float, float, float, float, float, float]) -> None:
    """Set a stable camera and show depth increasing downward."""
    x0, x1, y0, y1, z0, z1 = bounds
    center = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0, (z0 + z1) / 2.0], dtype=float)
    span = np.array([x1 - x0, y1 - y0, z1 - z0], dtype=float)
    diag = float(np.linalg.norm(np.maximum(span, 1.0)))
    plotter.camera_position = (
        tuple(center + np.array([1.55, 2.15, 1.05]) * diag),
        tuple(center),
        (0.0, 0.0, -1.0),
    )
    plotter.enable_parallel_projection()
    plotter.camera.parallel_scale = max(float(np.max(span)) * 1.0, 1.0)


def _apply_axis_scale(plotter, bounds: tuple[float, float, float, float, float, float], keep_aspect: bool) -> None:
    """Apply proportional or normalized axis scale."""
    if keep_aspect:
        plotter.set_scale(xscale=1.0, yscale=1.0, zscale=1.0)
        return
    spans = np.maximum(
        np.array([bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]], dtype=float),
        1.0,
    )
    max_span = float(np.max(spans))
    plotter.set_scale(
        xscale=max_span / spans[0],
        yscale=max_span / spans[1],
        zscale=max_span / spans[2],
    )


def _read_simulation_data(
    h5_file_path: str | Path,
    group_names: list[str],
    level: Literal["element", "cell"],
    property_name: str,
    group_display_names: dict[str, str] | None,
    progress_callback: ProgressCallback | None,
) -> tuple[list[dict], np.ndarray, np.ndarray, float, float]:
    """Read all selected groups and return shared bounds/color scale."""
    reader = read_element_stage_data if level == "element" else read_cell_stage_data
    all_sim_data: list[dict] = []
    global_min = np.array([np.inf, np.inf, np.inf])
    global_max = np.array([-np.inf, -np.inf, -np.inf])
    all_property_values = []

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
                    message=f"Loaded {group_name} as {sim_data['stage_label']} for PyVista.",
                )

    if not all_sim_data:
        raise ValueError("No valid stage data found for the selected groups/property.")
    if not np.all(np.isfinite(global_min)) or not np.all(np.isfinite(global_max)):
        raise ValueError("No valid fracture coordinates found.")

    prop_min, prop_max = compute_vmin_vmax(np.concatenate(all_property_values))
    return all_sim_data, global_min, global_max, prop_min, prop_max


def generate_pyvista_ufm_animation(
    h5_file_path: str | Path,
    group_names: list[str],
    level: Literal["element", "cell"],
    property_name: str,
    output_dir: str | Path,
    group_display_names: dict[str, str] | None = None,
    output_format: Literal["gif", "mp4"] = "gif",
    well_file_path: str | Path | None = None,
    fps: int = 20,
    keep_aspect: bool = True,
    time_step_stride: int = 1,
    progress_callback: ProgressCallback | None = None,
) -> Path:
    """Generate a UFM animation using PyVista/VTK instead of Matplotlib.

    This backend is meant for GPU/OpenGL-capable servers. It keeps finalized
    stage actors in the scene and updates only the current stage actor.
    """
    if not group_names:
        raise ValueError("At least one group must be selected.")
    if output_format not in {"gif", "mp4"}:
        raise ValueError("output_format only supports 'gif' or 'mp4'.")

    render_start = perf_counter()
    logger.info(
        "PyVista animation started | file=%s level=%s property=%s groups=%s stride=%s format=%s",
        h5_file_path,
        level,
        property_name,
        len(group_names),
        time_step_stride,
        output_format,
    )

    pv = _require_pyvista()
    all_sim_data, global_min, global_max, prop_min, prop_max = _read_simulation_data(
        h5_file_path,
        group_names,
        level,
        property_name,
        group_display_names,
        progress_callback,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"ufm_pyvista_{level}_{property_name}_{get_current_time()}.{output_format}"

    well_points = None
    if well_file_path:
        w_x, w_y, w_z = read_well_trajectory(well_file_path)
        well_points = np.column_stack([w_x, w_y, w_z]).astype(np.float64)
        finite = np.isfinite(well_points).all(axis=1)
        well_points = well_points[finite]
        if well_points.size:
            global_min = np.minimum(global_min, np.nanmin(well_points, axis=0))
            global_max = np.maximum(global_max, np.nanmax(well_points, axis=0))

    scene_bounds, label_counts = _grid_scene_bounds(global_min, global_max)
    plotter = pv.Plotter(off_screen=True, window_size=(1280, 720))
    plotter.set_background("white")
    plotter.add_axes()
    _apply_axis_scale(plotter, scene_bounds, keep_aspect)
    plotter.show_bounds(
        bounds=scene_bounds,
        grid="front",
        location="closest",
        ticks="inside",
        all_edges=False,
        xtitle="X (m)",
        ytitle="Y (m)",
        ztitle="Depth (m)",
        color="black",
        font_size=11,
        fmt="%.0f",
        n_xlabels=label_counts[0],
        n_ylabels=label_counts[1],
        n_zlabels=label_counts[2],
        use_2d=False,
    )
    _configure_camera(plotter, scene_bounds)

    if well_points is not None and well_points.size:
        line = pv.lines_from_points(well_points)
        plotter.add_mesh(line, color="black", line_width=4, label="Well Trajectory")

    if output_format == "gif":
        plotter.open_gif(output_path, fps=fps)
    else:
        plotter.open_movie(output_path, framerate=fps)

    mesh_builder = _element_mesh_for_stage if level == "element" else _cell_mesh_for_stage
    frames_map = [
        (sim_idx, time_step)
        for sim_idx, sim_data in enumerate(all_sim_data)
        for time_step in build_sampled_time_steps(sim_data["n_time"], time_step_stride)
    ]
    frame_count = max(1, len(frames_map))
    finalized_sims: set[int] = set()
    finalized_actors = []
    current_actor = None

    for frame_idx, (curr_sim_idx, curr_time_step) in enumerate(frames_map):
        sim_data = all_sim_data[curr_sim_idx]

        for prev_idx in range(curr_sim_idx):
            if prev_idx in finalized_sims:
                continue
            final_mesh = mesh_builder(all_sim_data[prev_idx], all_sim_data[prev_idx]["n_time"] - 1, pv)
            actor = plotter.add_mesh(
                final_mesh,
                scalars="value",
                clim=(prop_min, prop_max),
                cmap="jet",
                show_scalar_bar=False,
                opacity=0.9,
                lighting=False,
                smooth_shading=False,
            )
            finalized_actors.append(actor)
            finalized_sims.add(prev_idx)

        if current_actor is not None:
            plotter.remove_actor(current_actor)
        current_mesh = mesh_builder(sim_data, curr_time_step, pv)
        current_actor = plotter.add_mesh(
            current_mesh,
            scalars="value",
            clim=(prop_min, prop_max),
            cmap="jet",
            show_scalar_bar=True,
            scalar_bar_args={
                "title": property_name,
                "vertical": True,
                "position_x": 0.9,
                "position_y": 0.12,
                "width": 0.06,
                "height": 0.72,
                "color": "black",
                "title_font_size": 14,
                "label_font_size": 11,
            },
            opacity=0.95,
            lighting=False,
            smooth_shading=False,
        )

        step_str = f"{curr_time_step + 1}/{sim_data['n_time']}"
        plotter.add_text(
            format_animation_title(level, property_name, sim_data["stage_label"], step_str),
            position="upper_edge",
            font_size=12,
            color="black",
            name="title",
        )
        plotter.add_text(
            (
                f"Current Group: {sim_data['id']}\n"
                f"Name: {sim_data['display_name']}\n"
                f"X: {scene_bounds[0]:.0f} - {scene_bounds[1]:.0f} m | "
                f"Y: {scene_bounds[2]:.0f} - {scene_bounds[3]:.0f} m | "
                f"Depth: {scene_bounds[4]:.0f} - {scene_bounds[5]:.0f} m"
            ),
            position="lower_left",
            font_size=8,
            color="black",
            name="info",
        )
        _configure_camera(plotter, scene_bounds)
        plotter.write_frame()

        if progress_callback:
            percent = 15 + int((frame_idx + 1) / frame_count * 84)
            progress_callback(
                percent=percent,
                current_group=sim_data["id"],
                current_group_index=curr_sim_idx + 1,
                total_groups=len(all_sim_data),
                message=f"PyVista rendering {sim_data['stage_label']} ({sim_data['id']}) frame {step_str}.",
            )

    plotter.close()
    elapsed = perf_counter() - render_start
    logger.info("PyVista animation finished | elapsed=%.2fs output=%s", elapsed, output_path)
    return output_path
