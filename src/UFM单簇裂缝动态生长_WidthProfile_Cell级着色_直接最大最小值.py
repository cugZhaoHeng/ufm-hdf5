# -*- coding: utf-8 -*-
"""
UFM 单簇裂缝动态生长展示：按 cell 显示 WidthProfile 分布

本版核心修改：
1. 使用 ElementCount 判断每个时间步有效 element 数量；
2. 读取 Elements/Cells/WidthProfile，按 cell 级别进行颜色映射；
3. 每个 element 根据 4 个角点被近似剖分为 n_cell 个小四边形 cell；
4. 每个 cell 使用对应 WidthProfile[t, element, cell] 着色；
5. 色标最大/最小值不再使用固定 0-9，也不使用 P2/P98；
6. 默认直接取“最终时间步正在显示的有效 cell”的真实最小值和最大值作为色标范围；
7. 默认将 WidthProfile 从 m 转换为 mm 展示，便于和 Petrel 软件色标对比。

说明：
H5 中 Points/X/Y/Z 每个 element 只有 4 个角点，而 WidthProfile 是 cell 级属性。
因此这里采用几何插值方式，把一个 element 沿局部高度方向近似剖分为 n_cell 个小条带，
用于把 cell 级 WidthProfile 可视化到裂缝面上。
"""

from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import re
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation


INVALID_LIMIT = 1.0e30
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
HDF5_FILE = PROJECT_DIR / "data" / "hdf5_file"

def list_fracture_simulation_ids(f):
    """列出 HDF5 根目录下全部 Fracture Simulation {sim_id}。"""
    sim_ids = []
    pattern = re.compile(r"^Fracture Simulation\s+(\d+)$")

    for name in f.keys():
        match = pattern.match(name)
        if match:
            sim_ids.append(int(match.group(1)))

    return sorted(sim_ids)


def safe_valid_values(arr):
    """去除 NaN、Inf 和 UFM 常见无效大数值。"""
    arr = np.asarray(arr, dtype=float)
    mask = np.isfinite(arr) & (np.abs(arr) < INVALID_LIMIT)
    return arr[mask]


def read_cell_property(f, base_path, prop_name="WidthProfile"):
    """
    读取 cell 级属性。

    默认路径：
    Fracture Simulation XX/Results/Bulk/UFM/FractureSet/Elements/Cells/WidthProfile

    返回：
    data, shape = (time, element, cell)
    """
    prop_path = f"{base_path}/Elements/Cells/{prop_name}"

    if prop_path not in f:
        raise KeyError(f"未找到 cell 属性路径：{prop_path}")

    data = np.asarray(f[prop_path][:], dtype=float)

    if data.ndim != 3:
        raise ValueError(
            f"{prop_name} 应为 cell 级属性，期望维度为 (time, element, cell)，"
            f"但当前维度为 {data.shape}。"
        )

    data = np.where(np.abs(data) < INVALID_LIMIT, data, np.nan)
    return data


def split_element_to_cell_quads(points4, n_cells):
    """
    将一个四边形 element 近似剖分为 n_cells 个小四边形。

    假设角点顺序为：
    P1 = corner 0, P2 = corner 1, P3 = corner 2, P4 = corner 3

    几何连接近似为：
    P1 ---- P2
    |        |
    |        |
    P4 ---- P3

    沿 P1-P4 和 P2-P3 两条边插值，生成 n_cells 个局部条带。
    """
    p1, p2, p3, p4 = points4
    quads = []

    for k in range(n_cells):
        eta0 = k / n_cells
        eta1 = (k + 1) / n_cells

        left0 = (1.0 - eta0) * p1 + eta0 * p4
        right0 = (1.0 - eta0) * p2 + eta0 * p3
        right1 = (1.0 - eta1) * p2 + eta1 * p3
        left1 = (1.0 - eta1) * p1 + eta1 * p4

        quads.append(np.vstack([left0, right0, right1, left1]))

    return quads


def get_display_values_by_scale_source(
    cell_property,
    element_count,
    n_element_max,
    n_time,
    value_scale,
    color_scale_source="final",
    frame_index_for_scale=None,
):
    """
    根据指定范围直接提取真实最小值和最大值。

    color_scale_source:
        "final"  : 只取最终时间步、最终有效 element/cell 的真实 min/max。推荐用于和 Petrel 最终显示对比。
        "global" : 取所有时间步中有效 element/cell 的真实 min/max。适合动画全过程统一色标。
        "frame"  : 每一帧单独取当前帧有效 cell 的真实 min/max。适合突出当前帧局部差异。
    """
    values = []

    if color_scale_source == "final":
        t = n_time - 1
        n_active = min(element_count[t], n_element_max, cell_property.shape[1])
        values.append(cell_property[t, :n_active, :].reshape(-1))

    elif color_scale_source == "global":
        for t in range(n_time):
            n_active = min(element_count[t], n_element_max, cell_property.shape[1])
            if n_active > 0:
                values.append(cell_property[t, :n_active, :].reshape(-1))

    elif color_scale_source == "frame":
        if frame_index_for_scale is None:
            raise ValueError("color_scale_source='frame' 时必须提供 frame_index_for_scale。")
        t = int(frame_index_for_scale)
        n_active = min(element_count[t], n_element_max, cell_property.shape[1])
        values.append(cell_property[t, :n_active, :].reshape(-1))

    else:
        raise ValueError("color_scale_source 只能为 'final'、'global' 或 'frame'。")

    if len(values) == 0:
        return np.array([])

    values = np.concatenate(values) * value_scale
    values = safe_valid_values(values)
    return values


def compute_direct_vmin_vmax(values):
    """
    直接取真实最小值和最大值。
    如果最大值等于最小值，为了 Matplotlib 正常显示，色标范围会做极小扩展；
    但 raw_min/raw_max 仍然是真实值。
    """
    values = safe_valid_values(values)

    if values.size == 0:
        raise ValueError("没有有效属性值，无法计算色标范围。")

    raw_min = float(np.nanmin(values))
    raw_max = float(np.nanmax(values))

    vmin = raw_min
    vmax = raw_max

    if np.isclose(vmin, vmax):
        delta = 0.5 if np.isclose(vmin, 0.0) else abs(vmin) * 0.05
        vmin = raw_min - delta
        vmax = raw_max + delta

    return raw_min, raw_max, vmin, vmax


def read_widthprofile_simulation_snapshot(
    f,
    sim_id,
    prop_name="WidthProfile",
    snapshot="final",
):
    """
    读取单个 Fracture Simulation 的一个结果快照。

    snapshot:
        "final"：读取该 simulation 最终有效时间步；
        整数：读取指定 time index。
    """
    base_path = f"Fracture Simulation {sim_id}/Results/Bulk/UFM/FractureSet"
    points_path = f"{base_path}/Elements/Points"
    element_count_path = f"{base_path}/ElementCount"

    required_paths = [
        f"{points_path}/X",
        f"{points_path}/Y",
        f"{points_path}/Z",
        element_count_path,
        f"{base_path}/Elements/Cells/{prop_name}",
    ]

    for p in required_paths:
        if p not in f:
            raise KeyError(f"Fracture Simulation {sim_id} 未找到必要路径：{p}")

    X = np.asarray(f[f"{points_path}/X"][:], dtype=float)
    Y = np.asarray(f[f"{points_path}/Y"][:], dtype=float)
    Z = np.asarray(f[f"{points_path}/Z"][:], dtype=float)
    ElementCount = np.asarray(f[element_count_path][:]).reshape(-1).astype(int)
    CellProperty = read_cell_property(f, base_path, prop_name=prop_name)

    n_time = min(X.shape[0], len(ElementCount), CellProperty.shape[0])
    n_element_max = min(X.shape[1], CellProperty.shape[1])

    if n_time <= 0:
        raise ValueError(f"Fracture Simulation {sim_id} 没有有效时间步。")

    if snapshot == "final":
        t = n_time - 1
    else:
        t = int(snapshot)
        if t < 0 or t >= n_time:
            raise ValueError(f"Fracture Simulation {sim_id} 的 snapshot={t} 超出范围 0-{n_time - 1}。")

    n_active = min(ElementCount[t], n_element_max)
    if n_active <= 0:
        raise ValueError(f"Fracture Simulation {sim_id} 在 time index {t} 的有效 element 数量为 0。")

    return {
        "sim_id": sim_id,
        "time_index": t,
        "n_time": n_time,
        "n_active": n_active,
        "n_element_max": n_element_max,
        "n_cells": CellProperty.shape[2],
        "X": X[t, :n_active, :],
        "Y": Y[t, :n_active, :],
        "Z": Z[t, :n_active, :],
        "CellProperty": CellProperty[t, :n_active, :],
    }


def render_all_fracture_simulations_widthprofile_cell_level(
    file_path,
    sim_ids=None,
    prop_name="WidthProfile",
    snapshot="final",
    remove_origin=True,
    invert_z_axis=True,
    cmap_name="jet",
    alpha=1.0,
    edge_color="0.50",
    edge_width=0.013,
    view_elev=24,
    view_azim=-58,
    value_scale=1000.0,
    value_unit="mm",
    skip_invalid=True,
):
    """
    一次性读取 HDF5 中全部 Fracture Simulation {sim_id}，并叠加渲染到同一个三维图。

    默认读取每个 simulation 的最终时间步；如果需要指定统一 time index，可将 snapshot 设为整数。
    """
    print("读取 H5 中全部 Fracture Simulation 数据...")

    snapshots = []
    skipped = []

    with h5py.File(file_path, "r") as f:
        if sim_ids is None:
            sim_ids = list_fracture_simulation_ids(f)

        if not sim_ids:
            raise ValueError("HDF5 根目录下未找到任何 Fracture Simulation {sim_id}。")

        print(f"发现 Fracture Simulation 数量：{len(sim_ids)}")
        print(f"sim_id 列表：{sim_ids}")

        for sim_id in sim_ids:
            try:
                snap = read_widthprofile_simulation_snapshot(
                    f,
                    sim_id=sim_id,
                    prop_name=prop_name,
                    snapshot=snapshot,
                )
                snapshots.append(snap)
                print(
                    f"  - sim {sim_id}: time {snap['time_index']}/{snap['n_time'] - 1}, "
                    f"active elements {snap['n_active']}"
                )
            except Exception as exc:
                if not skip_invalid:
                    raise
                skipped.append((sim_id, str(exc)))
                print(f"  - sim {sim_id}: 跳过，原因：{exc}")

    if not snapshots:
        raise ValueError("没有可渲染的 Fracture Simulation 数据。")

    if remove_origin:
        X0 = min(float(np.nanmin(s["X"])) for s in snapshots)
        Y0 = min(float(np.nanmin(s["Y"])) for s in snapshots)
        Z0 = min(float(np.nanmin(s["Z"])) for s in snapshots)
    else:
        X0 = Y0 = Z0 = 0.0

    all_values = []
    global_min = np.array([np.inf, np.inf, np.inf])
    global_max = np.array([-np.inf, -np.inf, -np.inf])

    for s in snapshots:
        s["X_plot"] = s["X"] - X0
        s["Y_plot"] = s["Y"] - Y0
        s["Z_plot"] = s["Z"] - Z0

        global_min = np.minimum(
            global_min,
            [
                np.nanmin(s["X_plot"]),
                np.nanmin(s["Y_plot"]),
                np.nanmin(s["Z_plot"]),
            ],
        )
        global_max = np.maximum(
            global_max,
            [
                np.nanmax(s["X_plot"]),
                np.nanmax(s["Y_plot"]),
                np.nanmax(s["Z_plot"]),
            ],
        )
        all_values.append(s["CellProperty"].reshape(-1))

    display_values = safe_valid_values(np.concatenate(all_values) * value_scale)
    raw_min, raw_max, vmin, vmax = compute_direct_vmin_vmax(display_values)

    span = global_max - global_min
    pad = np.maximum(span * 0.08, 1.0)
    global_min = global_min - pad
    global_max = global_max + pad

    print("=" * 60)
    print(f"叠加渲染 simulation 数量 = {len(snapshots)}")
    print(f"快照选择 snapshot = {snapshot}")
    print(f"{prop_name} 全部模拟真实最小值 raw_min = {raw_min:.10g} {value_unit}")
    print(f"{prop_name} 全部模拟真实最大值 raw_max = {raw_max:.10g} {value_unit}")
    if skipped:
        print(f"跳过 simulation 数量 = {len(skipped)}")
    print("=" * 60)

    cmap = mpl.colormaps[cmap_name]
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)

    verts = []
    colors = []
    rendered_elements = 0

    for s in snapshots:
        n_cells = s["n_cells"]
        values = s["CellProperty"] * value_scale

        for e in range(s["n_active"]):
            points4 = np.column_stack([s["X_plot"][e], s["Y_plot"][e], s["Z_plot"][e]])
            if np.isnan(points4).any():
                continue

            values_cells = values[e, :]
            if np.all(~np.isfinite(values_cells)):
                continue

            for c_idx, quad in enumerate(split_element_to_cell_quads(points4, n_cells=n_cells)):
                value = values_cells[c_idx]
                if not np.isfinite(value) or abs(value) >= INVALID_LIMIT:
                    continue

                verts.append(quad)
                rgba = list(cmap(norm(value)))
                rgba[3] = alpha
                colors.append(rgba)

            rendered_elements += 1

    fig = plt.figure(figsize=(14.5, 8.8))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.035, right=0.83, top=0.90, bottom=0.07)

    ax.set_xlim(global_min[0], global_max[0])
    ax.set_ylim(global_min[1], global_max[1])
    ax.set_zlim(global_min[2], global_max[2])

    if invert_z_axis:
        ax.invert_zaxis()

    ax.view_init(elev=view_elev, azim=view_azim)
    ax.set_xlabel(f"X - X0 (m), X0={X0:.3f}" if remove_origin else "X (m)", labelpad=12)
    ax.set_ylabel(f"Y - Y0 (m), Y0={Y0:.3f}" if remove_origin else "Y (m)", labelpad=12)
    ax.set_zlabel(f"Z - Z0 (m), Z0={Z0:.3f}" if remove_origin else "Depth / Z (m)", labelpad=12)

    try:
        ax.set_box_aspect((span[0], span[1], span[2]))
    except Exception:
        pass

    ax.set_facecolor("white")
    try:
        ax.xaxis.pane.set_facecolor((0.96, 0.98, 1.0, 1.0))
        ax.yaxis.pane.set_facecolor((0.96, 0.98, 1.0, 1.0))
        ax.zaxis.pane.set_facecolor((0.96, 0.98, 1.0, 1.0))
        ax.grid(True, color="0.86", linewidth=0.6)
    except Exception:
        pass

    fracture_collection = Poly3DCollection(
        verts,
        facecolors=colors,
        alpha=alpha,
        edgecolors=edge_color,
        linewidths=edge_width,
    )
    ax.add_collection3d(fracture_collection)

    fig.suptitle(
        f"UFM Fracture Results - All Simulations | Cell {prop_name} Distribution",
        fontsize=16,
        fontweight="bold",
        y=0.965,
    )

    total_active_elements = sum(s["n_active"] for s in snapshots)
    info_text = (
        f"Fracture Simulations: {len(snapshots)}\n"
        f"Snapshot: {snapshot}\n"
        f"Active elements: {total_active_elements}\n"
        f"Rendered elements: {rendered_elements}\n"
        f"Rendered cell patches: {len(verts)}\n"
        f"{prop_name} min: {raw_min:.4g} {value_unit}\n"
        f"{prop_name} max: {raw_max:.4g} {value_unit}"
    )
    if skipped:
        info_text += f"\nSkipped simulations: {len(skipped)}"

    ax.text2D(
        0.018,
        0.025,
        info_text,
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

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.90, 0.18, 0.026, 0.64])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(f"{prop_name} ({value_unit})", fontsize=11, fontweight="bold")
    cbar.ax.tick_params(labelsize=10)

    if raw_min != raw_max:
        cbar.set_ticks(np.linspace(vmin, vmax, 6))
    cbar.ax.set_title("all sim min-max", fontsize=9, pad=8)

    plt.show()


def animate_ufm_growth_widthprofile_cell_level(
    file_path,
    sim_id=24,
    prop_name="WidthProfile",
    interval=300,
    repeat=False,
    remove_origin=True,
    invert_z_axis=True,
    cmap_name="jet",
    alpha=1.0,
    edge_color="0.50",
    edge_width=0.013,
    view_elev=24,
    view_azim=-58,
    save_path=None,
    fps=5,
    value_scale=1000.0,
    value_unit="mm",
    color_scale_source="final",  # final / global / frame
):
    """
    动态展示 UFM 裂缝生长，并按 cell 级 WidthProfile 着色。

    color_scale_source:
        "final" ：直接取最终时间步有效 cell 的真实 min/max，推荐用于和 Petrel 最终图对比；
        "global"：直接取所有时间步有效 cell 的真实 min/max，适合动画全过程统一色标；
        "frame" ：每一帧直接取当前帧有效 cell 的真实 min/max，颜色对比最强但色标会变化。
    """

    print("读取 H5 数据中...")

    with h5py.File(file_path, "r") as f:
        base_path = f"Fracture Simulation {sim_id}/Results/Bulk/UFM/FractureSet"
        points_path = f"{base_path}/Elements/Points"
        element_count_path = f"{base_path}/ElementCount"

        required_paths = [
            f"{points_path}/X",
            f"{points_path}/Y",
            f"{points_path}/Z",
            element_count_path,
            f"{base_path}/Elements/Cells/{prop_name}",
        ]

        for p in required_paths:
            if p not in f:
                raise KeyError(f"未找到必要路径：{p}")

        X = np.asarray(f[f"{points_path}/X"][:], dtype=float)
        Y = np.asarray(f[f"{points_path}/Y"][:], dtype=float)
        Z = np.asarray(f[f"{points_path}/Z"][:], dtype=float)
        ElementCount = np.asarray(f[element_count_path][:]).reshape(-1).astype(int)
        CellProperty = read_cell_property(f, base_path, prop_name=prop_name)

        print(f"X shape: {X.shape}")
        print(f"Y shape: {Y.shape}")
        print(f"Z shape: {Z.shape}")
        print(f"ElementCount shape: {ElementCount.shape}")
        print(f"{prop_name} shape: {CellProperty.shape}")

    n_time = min(X.shape[0], len(ElementCount), CellProperty.shape[0])
    n_element_max = min(X.shape[1], CellProperty.shape[1])
    n_cells = CellProperty.shape[2]

    if n_time <= 0:
        raise ValueError("没有有效时间步。")

    final_t = n_time - 1
    final_n = min(ElementCount[final_t], n_element_max)

    if final_n <= 0:
        raise ValueError("最终时间步有效 element 数量为 0。")

    valid_final_x = X[final_t, :final_n, :]
    valid_final_y = Y[final_t, :final_n, :]
    valid_final_z = Z[final_t, :final_n, :]

    if remove_origin:
        X0 = np.nanmin(valid_final_x)
        Y0 = np.nanmin(valid_final_y)
        Z0 = np.nanmin(valid_final_z)
        X_plot = X - X0
        Y_plot = Y - Y0
        Z_plot = Z - Z0
    else:
        X0 = Y0 = Z0 = 0.0
        X_plot = X
        Y_plot = Y
        Z_plot = Z

    # 坐标范围：严格按 ElementCount 统计有效 element
    global_min = np.array([np.inf, np.inf, np.inf])
    global_max = np.array([-np.inf, -np.inf, -np.inf])

    for t in range(n_time):
        n_active = min(ElementCount[t], n_element_max)
        if n_active <= 0:
            continue

        x_valid = X_plot[t, :n_active, :]
        y_valid = Y_plot[t, :n_active, :]
        z_valid = Z_plot[t, :n_active, :]

        if np.all(np.isnan(x_valid)):
            continue

        global_min = np.minimum(global_min, [np.nanmin(x_valid), np.nanmin(y_valid), np.nanmin(z_valid)])
        global_max = np.maximum(global_max, [np.nanmax(x_valid), np.nanmax(y_valid), np.nanmax(z_valid)])

    span = global_max - global_min
    pad = np.maximum(span * 0.08, 1.0)
    global_min = global_min - pad
    global_max = global_max + pad

    # =============================
    # 关键修改：直接取最大值和最小值
    # =============================
    scale_values = get_display_values_by_scale_source(
        cell_property=CellProperty,
        element_count=ElementCount,
        n_element_max=n_element_max,
        n_time=n_time,
        value_scale=value_scale,
        color_scale_source=color_scale_source,
        frame_index_for_scale=None,
    )

    raw_min, raw_max, vmin, vmax = compute_direct_vmin_vmax(scale_values)

    print("=" * 60)
    print(f"色标取值方式 color_scale_source = {color_scale_source}")
    print(f"{prop_name} 真实最小值 raw_min = {raw_min:.10g} {value_unit}")
    print(f"{prop_name} 真实最大值 raw_max = {raw_max:.10g} {value_unit}")
    print(f"Matplotlib 色标 vmin = {vmin:.10g} {value_unit}")
    print(f"Matplotlib 色标 vmax = {vmax:.10g} {value_unit}")
    print("=" * 60)

    cmap = mpl.colormaps[cmap_name]
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)

    # 创建画布
    fig = plt.figure(figsize=(14.5, 8.8))
    ax = fig.add_subplot(111, projection="3d")

    # 右侧留色标，左下角放信息框
    fig.subplots_adjust(left=0.035, right=0.83, top=0.90, bottom=0.07)

    ax.set_xlim(global_min[0], global_max[0])
    ax.set_ylim(global_min[1], global_max[1])
    ax.set_zlim(global_min[2], global_max[2])

    if invert_z_axis:
        ax.invert_zaxis()

    ax.view_init(elev=view_elev, azim=view_azim)

    ax.set_xlabel(f"X - X0 (m), X0={X0:.3f}" if remove_origin else "X (m)", labelpad=12)
    ax.set_ylabel(f"Y - Y0 (m), Y0={Y0:.3f}" if remove_origin else "Y (m)", labelpad=12)
    ax.set_zlabel(f"Z - Z0 (m), Z0={Z0:.3f}" if remove_origin else "Depth / Z (m)", labelpad=12)

    try:
        ax.set_box_aspect((span[0], span[1], span[2]))
    except Exception:
        pass

    ax.set_facecolor("white")
    try:
        ax.xaxis.pane.set_facecolor((0.96, 0.98, 1.0, 1.0))
        ax.yaxis.pane.set_facecolor((0.96, 0.98, 1.0, 1.0))
        ax.zaxis.pane.set_facecolor((0.96, 0.98, 1.0, 1.0))
        ax.grid(True, color="0.86", linewidth=0.6)
    except Exception:
        pass

    fracture_collection = Poly3DCollection(
        [],
        alpha=alpha,
        edgecolors=edge_color,
        linewidths=edge_width,
    )
    ax.add_collection3d(fracture_collection)

    fig.suptitle(
        f"UFM Fracture Growth - Cell {prop_name} Distribution | Fracture Simulation {sim_id}",
        fontsize=16,
        fontweight="bold",
        y=0.965,
    )

    info_text = ax.text2D(
        0.018,
        0.025,
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

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.90, 0.18, 0.026, 0.64])
    cbar = fig.colorbar(sm, cax=cax)

    cbar.set_label(f"{prop_name} ({value_unit})", fontsize=11, fontweight="bold")
    cbar.ax.tick_params(labelsize=10)

    if raw_min != raw_max:
        ticks = np.linspace(vmin, vmax, 6)
        cbar.set_ticks(ticks)
    cbar.ax.set_title("raw min-max", fontsize=9, pad=8)

    def update_colorbar_for_frame(t):
        """如果 color_scale_source='frame'，每帧直接用当前帧的真实 min/max 更新色标。"""
        nonlocal norm

        frame_values = get_display_values_by_scale_source(
            cell_property=CellProperty,
            element_count=ElementCount,
            n_element_max=n_element_max,
            n_time=n_time,
            value_scale=value_scale,
            color_scale_source="frame",
            frame_index_for_scale=t,
        )

        frame_raw_min, frame_raw_max, frame_vmin, frame_vmax = compute_direct_vmin_vmax(frame_values)

        norm = mpl.colors.Normalize(vmin=frame_vmin, vmax=frame_vmax, clip=True)
        sm.set_norm(norm)
        cbar.update_normal(sm)

        if frame_raw_min != frame_raw_max:
            cbar.set_ticks(np.linspace(frame_vmin, frame_vmax, 6))
        cbar.ax.set_title("frame min-max", fontsize=9, pad=8)

        return frame_raw_min, frame_raw_max

    def build_cell_verts_and_colors(t):
        """根据时间步 t 构造 cell 级小面片和颜色。"""
        n_active = min(ElementCount[t], n_element_max)

        x_t = X_plot[t, :n_active, :]
        y_t = Y_plot[t, :n_active, :]
        z_t = Z_plot[t, :n_active, :]
        w_t = CellProperty[t, :n_active, :] * value_scale

        verts = []
        colors = []
        valid_element_count = 0

        for e in range(n_active):
            points4 = np.column_stack([x_t[e], y_t[e], z_t[e]])
            if np.isnan(points4).any():
                continue

            values_cells = w_t[e, :]
            if np.all(~np.isfinite(values_cells)):
                continue

            cell_quads = split_element_to_cell_quads(points4, n_cells=n_cells)

            for c_idx, quad in enumerate(cell_quads):
                value = values_cells[c_idx]
                if not np.isfinite(value) or abs(value) >= INVALID_LIMIT:
                    continue

                verts.append(quad)

                rgba = list(cmap(norm(value)))
                rgba[3] = alpha
                colors.append(rgba)

            valid_element_count += 1

        return verts, colors, n_active, valid_element_count

    def update(frame_idx):
        t = frame_idx

        if color_scale_source == "frame":
            current_min, current_max = update_colorbar_for_frame(t)
        else:
            current_min, current_max = raw_min, raw_max

        verts, colors, n_active, valid_element_count = build_cell_verts_and_colors(t)

        fracture_collection.set_verts(verts)
        fracture_collection.set_facecolors(colors)

        if t == 0:
            n_prev = 0
        else:
            n_prev = min(ElementCount[t - 1], n_element_max)

        n_new = n_active - n_prev

        now_values = safe_valid_values(CellProperty[t, :n_active, :].reshape(-1)) * value_scale
        if now_values.size > 0:
            w_min = np.min(now_values)
            w_mean = np.mean(now_values)
            w_max = np.max(now_values)
        else:
            w_min = w_mean = w_max = np.nan

        info_text.set_text(
            f"TimeIndex: {t}/{n_time - 1}\n"
            f"Active elements: {n_active}/{n_element_max}\n"
            f"Rendered cell patches: {len(verts)}\n"
            f"New elements: {n_new}\n"
            f"Frame min: {w_min:.4g} {value_unit}\n"
            f"Frame mean: {w_mean:.4g} {value_unit}\n"
            f"Frame max: {w_max:.4g} {value_unit}\n"
            f"Colorbar min: {current_min:.4g} {value_unit}\n"
            f"Colorbar max: {current_max:.4g} {value_unit}"
        )

        return [fracture_collection, info_text]

    ani = FuncAnimation(
        fig,
        update,
        frames=n_time,
        interval=interval,
        blit=False,
        repeat=repeat,
    )

    if save_path:
        if save_path.lower().endswith(".gif"):
            ani.save(save_path, writer="pillow", fps=fps)
        else:
            ani.save(save_path, writer="ffmpeg", fps=fps)
        print(f"动画已保存：{save_path}")

    plt.show()


if __name__ == "__main__":
    target = HDF5_FILE / "JY108-7HF_cluster1.h5"

    render_all_fracture_simulations_widthprofile_cell_level(
        file_path=target,
        sim_ids=["01", '02'],             # None 表示自动读取 HDF5 中全部 Fracture Simulation {sim_id}
        prop_name="WidthProfile",
        snapshot="final",         # 每个 simulation 读取最终结果，并叠加到同一个三维图
        remove_origin=True,
        invert_z_axis=True,
        cmap_name="jet",          # 更接近 Petrel 软件色卡：蓝-青-绿-黄-红
        alpha=1.0,
        edge_color="0.5",
        edge_width=0.013,
        view_elev=24,
        view_azim=-58,
        value_scale=1000.0,       # WidthProfile 通常为 m，乘以 1000 后以 mm 显示
        value_unit="mm",
    )
