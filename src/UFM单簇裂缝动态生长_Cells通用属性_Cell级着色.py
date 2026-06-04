# -*- coding: utf-8 -*-
"""
UFM 单簇裂缝动态生长展示：Cells 通用属性 cell 级着色版

功能：
1. 使用 ElementCount 判断每个时间步有效 element 数量；
2. 自动列出 FractureSet/Elements/Cells 下所有可显示参数；
3. 可通过 prop_name 切换显示 WidthProfile、FractureConductivity、ProppedWidthProfile 等 Cells 参数；
4. 每个 element 由 4 个角点近似剖分为 n_cell 个小四边形 cell；
5. 每个 cell 使用对应 Cells 属性值着色；
6. 色标默认直接取真实 min/max；
7. 可导出所有 Cells 参数统计表，方便查看每个参数范围。
"""

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation

INVALID_LIMIT = 1.0e30

# 常用 Cells 参数显示配置。可按你的单位认识继续补充。
PROPERTY_CONFIG = {
    "WidthProfile": {"scale": 1000.0, "unit": "mm", "cmap": "jet"},
    "ProppedWidthProfile": {"scale": 1000.0, "unit": "mm", "cmap": "jet"},
    "FractureConductivity": {"scale": 1.0, "unit": "raw", "cmap": "turbo"},
    "AreaBasedPropConcProfile": {"scale": 1.0, "unit": "raw", "cmap": "turbo"},
    "VolumeConcentration": {"scale": 1.0, "unit": "raw", "cmap": "turbo"},
    "ChannelDistribution3D": {"scale": 1.0, "unit": "raw", "cmap": "turbo"},
    "ZBank": {"scale": 1.0, "unit": "m", "cmap": "viridis"},
    "ZSlurry": {"scale": 1.0, "unit": "m", "cmap": "viridis"},
}


def safe_valid_values(arr):
    """去除 NaN、Inf 和 UFM 常见无效大数值。"""
    arr = np.asarray(arr, dtype=float)
    return arr[np.isfinite(arr) & (np.abs(arr) < INVALID_LIMIT)]


def list_cell_properties(f, base_path):
    """列出 Elements/Cells 下所有 Dataset，返回 {属性名: shape}。"""
    cells_path = f"{base_path}/Elements/Cells"
    if cells_path not in f:
        raise KeyError(f"未找到 Cells 路径：{cells_path}")

    props = {}
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            props[name.split("/")[-1]] = obj.shape
    f[cells_path].visititems(visitor)
    return props


def read_cell_property(f, base_path, prop_name):
    """
    读取 Elements/Cells 下任意 cell 属性，并整理为 (time, element, cell)。

    - 原始 3D: 直接使用；
    - 原始 2D: (time, element) -> (time, element, 1)；
    - 原始 >3D: 对第 4 维及之后维度取平均，压缩为 (time, element, cell)。
    """
    prop_path = f"{base_path}/Elements/Cells/{prop_name}"
    if prop_path not in f:
        available = list_cell_properties(f, base_path)
        msg = "\n".join([f"  - {k}: {v}" for k, v in available.items()])
        raise KeyError(f"未找到 cell 属性路径：{prop_path}\n可用 Cells 属性：\n{msg}")

    data = np.asarray(f[prop_path][:], dtype=float)
    data = np.where(np.abs(data) < INVALID_LIMIT, data, np.nan)

    if data.ndim == 2:
        data = data[:, :, np.newaxis]
    elif data.ndim == 3:
        pass
    elif data.ndim > 3:
        data = np.nanmean(data, axis=tuple(range(3, data.ndim)))
    else:
        raise ValueError(f"{prop_name} 维度异常，无法整理为 (time, element, cell)：{data.shape}")
    return data


def split_element_to_cell_quads(points4, n_cells):
    """
    将一个四边形 element 沿局部高度方向剖分为 n_cells 个小四边形。
    角点近似关系：P1-P2 为上边界，P4-P3 为下边界。
    """
    p1, p2, p3, p4 = points4
    quads = []
    for k in range(n_cells):
        eta0 = k / n_cells
        eta1 = (k + 1) / n_cells
        left0 = (1 - eta0) * p1 + eta0 * p4
        right0 = (1 - eta0) * p2 + eta0 * p3
        right1 = (1 - eta1) * p2 + eta1 * p3
        left1 = (1 - eta1) * p1 + eta1 * p4
        quads.append(np.vstack([left0, right0, right1, left1]))
    return quads


def get_values_for_scale(cell_property, element_count, n_element_max, n_time,
                         value_scale=1.0, color_scale_source="final", frame_index=None):
    """
    取色标范围所用数据。
    color_scale_source:
      - final: 最终时间步有效 cell 的真实 min/max；
      - global: 所有时间步有效 cell 的真实 min/max；
      - frame: 当前帧有效 cell 的真实 min/max。
    """
    values = []
    if color_scale_source == "final":
        t = n_time - 1
        n = min(element_count[t], n_element_max, cell_property.shape[1])
        values.append(cell_property[t, :n, :].reshape(-1))
    elif color_scale_source == "global":
        for t in range(n_time):
            n = min(element_count[t], n_element_max, cell_property.shape[1])
            if n > 0:
                values.append(cell_property[t, :n, :].reshape(-1))
    elif color_scale_source == "frame":
        if frame_index is None:
            raise ValueError("frame 色标模式需要 frame_index")
        t = int(frame_index)
        n = min(element_count[t], n_element_max, cell_property.shape[1])
        values.append(cell_property[t, :n, :].reshape(-1))
    else:
        raise ValueError("color_scale_source 只能是 final、global 或 frame")

    if not values:
        return np.array([])
    return safe_valid_values(np.concatenate(values) * value_scale)


def direct_minmax(values):
    """直接取真实最小值和最大值；若相等，仅为绘图做极小扩展。"""
    values = safe_valid_values(values)
    if values.size == 0:
        raise ValueError("没有有效值，无法计算色标范围")
    raw_min = float(np.min(values))
    raw_max = float(np.max(values))
    vmin, vmax = raw_min, raw_max
    if np.isclose(vmin, vmax):
        delta = 0.5 if np.isclose(vmin, 0) else abs(vmin) * 0.05
        vmin -= delta
        vmax += delta
    return raw_min, raw_max, vmin, vmax


def export_all_cells_property_stats(file_path, sim_id=24, output_csv=None):
    """导出 Elements/Cells 下所有属性的统计表。"""
    rows = []
    with h5py.File(file_path, "r") as f:
        base_path = f"Fracture Simulation {sim_id}/Results/Bulk/UFM/FractureSet"
        props = list_cell_properties(f, base_path)
        for prop_name, original_shape in props.items():
            try:
                data = read_cell_property(f, base_path, prop_name)
                raw = safe_valid_values(data.reshape(-1))
                cfg = PROPERTY_CONFIG.get(prop_name, {})
                scale = float(cfg.get("scale", 1.0))
                unit = cfg.get("unit", "raw")
                disp = raw * scale if raw.size else raw
                rows.append({
                    "Property": prop_name,
                    "OriginalShape": str(original_shape),
                    "ProcessedShape": str(data.shape),
                    "RawMin": float(np.min(raw)) if raw.size else np.nan,
                    "RawMean": float(np.mean(raw)) if raw.size else np.nan,
                    "RawMax": float(np.max(raw)) if raw.size else np.nan,
                    "DisplayScale": scale,
                    "DisplayUnit": unit,
                    "DisplayMin": float(np.min(disp)) if disp.size else np.nan,
                    "DisplayMean": float(np.mean(disp)) if disp.size else np.nan,
                    "DisplayMax": float(np.max(disp)) if disp.size else np.nan,
                })
            except Exception as e:
                rows.append({
                    "Property": prop_name,
                    "OriginalShape": str(original_shape),
                    "ProcessedShape": "ERROR",
                    "Error": str(e),
                })

    df = pd.DataFrame(rows)
    if output_csv is None:
        output_csv = f"cells_property_stats_sim{sim_id}.csv"
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Cells 属性统计表已导出：{output_csv}")
    print(df.to_string(index=False))
    return df


def animate_ufm_growth_cell_property(
    file_path,
    sim_id=24,
    prop_name="WidthProfile",
    interval=300,
    repeat=False,
    remove_origin=True,
    invert_z_axis=True,
    cmap_name=None,
    alpha=1.0,
    edge_color="0.50",
    edge_width=0.013,
    view_elev=24,
    view_azim=-58,
    save_path=None,
    fps=5,
    value_scale=None,
    value_unit=None,
    color_scale_source="final",
    export_stats=True,
):
    """动态展示 UFM 裂缝生长，并按任意 Cells 属性进行 cell 级着色。"""
    cfg = PROPERTY_CONFIG.get(prop_name, {})
    if value_scale is None:
        value_scale = float(cfg.get("scale", 1.0))
    if value_unit is None:
        value_unit = cfg.get("unit", "raw")
    if cmap_name is None:
        cmap_name = cfg.get("cmap", "turbo")

    with h5py.File(file_path, "r") as f:
        base_path = f"Fracture Simulation {sim_id}/Results/Bulk/UFM/FractureSet"
        points_path = f"{base_path}/Elements/Points"
        element_count_path = f"{base_path}/ElementCount"

        for p in [f"{points_path}/X", f"{points_path}/Y", f"{points_path}/Z", element_count_path, f"{base_path}/Elements/Cells"]:
            if p not in f:
                raise KeyError(f"未找到必要路径：{p}")

        print("\n当前 H5 中 Elements/Cells 可用属性：")
        for k, v in list_cell_properties(f, base_path).items():
            print(f"  - {k}: {v}")

        X = np.asarray(f[f"{points_path}/X"][:], dtype=float)
        Y = np.asarray(f[f"{points_path}/Y"][:], dtype=float)
        Z = np.asarray(f[f"{points_path}/Z"][:], dtype=float)
        ElementCount = np.asarray(f[element_count_path][:]).reshape(-1).astype(int)
        CellProperty = read_cell_property(f, base_path, prop_name)

    if export_stats:
        export_all_cells_property_stats(file_path, sim_id=sim_id,
                                        output_csv=f"cells_property_stats_sim{sim_id}.csv")

    n_time = min(X.shape[0], len(ElementCount), CellProperty.shape[0])
    n_element_max = min(X.shape[1], CellProperty.shape[1])
    n_cells = CellProperty.shape[2]

    final_t = n_time - 1
    final_n = min(ElementCount[final_t], n_element_max)
    if final_n <= 0:
        raise ValueError("最终时间步有效 element 数量为 0")

    if remove_origin:
        X0 = np.nanmin(X[final_t, :final_n, :])
        Y0 = np.nanmin(Y[final_t, :final_n, :])
        Z0 = np.nanmin(Z[final_t, :final_n, :])
        Xp, Yp, Zp = X - X0, Y - Y0, Z - Z0
    else:
        X0 = Y0 = Z0 = 0.0
        Xp, Yp, Zp = X, Y, Z

    # 坐标范围
    global_min = np.array([np.inf, np.inf, np.inf])
    global_max = np.array([-np.inf, -np.inf, -np.inf])
    for t in range(n_time):
        n = min(ElementCount[t], n_element_max)
        if n <= 0:
            continue
        global_min = np.minimum(global_min, [np.nanmin(Xp[t, :n, :]), np.nanmin(Yp[t, :n, :]), np.nanmin(Zp[t, :n, :])])
        global_max = np.maximum(global_max, [np.nanmax(Xp[t, :n, :]), np.nanmax(Yp[t, :n, :]), np.nanmax(Zp[t, :n, :])])
    span = global_max - global_min
    pad = np.maximum(span * 0.08, 1.0)
    global_min -= pad
    global_max += pad

    values_for_scale = get_values_for_scale(
        CellProperty, ElementCount, n_element_max, n_time,
        value_scale=value_scale, color_scale_source=color_scale_source
    )
    raw_min, raw_max, vmin, vmax = direct_minmax(values_for_scale)

    print("=" * 70)
    print(f"显示属性 prop_name = {prop_name}")
    print(f"处理后数据维度 = {CellProperty.shape}")
    print(f"显示倍率 value_scale = {value_scale}")
    print(f"显示单位 value_unit = {value_unit}")
    print(f"色标取值方式 color_scale_source = {color_scale_source}")
    print(f"真实最小值 raw_min = {raw_min:.10g} {value_unit}")
    print(f"真实最大值 raw_max = {raw_max:.10g} {value_unit}")
    print("=" * 70)

    cmap = mpl.colormaps[cmap_name]
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)

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

    fracture_collection = Poly3DCollection([], alpha=alpha, edgecolors=edge_color, linewidths=edge_width)
    ax.add_collection3d(fracture_collection)

    fig.suptitle(f"UFM Fracture Growth - Cell {prop_name} Distribution | Fracture Simulation {sim_id}",
                 fontsize=16, fontweight="bold", y=0.965)

    info_text = ax.text2D(
        0.018, 0.025, "", transform=ax.transAxes, fontsize=10.5,
        color="#1f6fd1", va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#eef6ff", edgecolor="#1f6fd1", alpha=0.94)
    )

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.90, 0.18, 0.026, 0.64])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(f"{prop_name} ({value_unit})", fontsize=11, fontweight="bold")
    if raw_min != raw_max:
        cbar.set_ticks(np.linspace(vmin, vmax, 6))
    cbar.ax.set_title("raw min-max", fontsize=9, pad=8)

    def update_frame_norm(t):
        nonlocal norm
        vals = get_values_for_scale(CellProperty, ElementCount, n_element_max, n_time,
                                    value_scale=value_scale, color_scale_source="frame", frame_index=t)
        f_raw_min, f_raw_max, f_vmin, f_vmax = direct_minmax(vals)
        norm = mpl.colors.Normalize(vmin=f_vmin, vmax=f_vmax, clip=True)
        sm.set_norm(norm)
        cbar.update_normal(sm)
        cbar.set_ticks(np.linspace(f_vmin, f_vmax, 6))
        cbar.ax.set_title("frame min-max", fontsize=9, pad=8)
        return f_raw_min, f_raw_max

    def build_cell_verts_and_colors(t):
        n = min(ElementCount[t], n_element_max)
        x_t, y_t, z_t = Xp[t, :n, :], Yp[t, :n, :], Zp[t, :n, :]
        values_t = CellProperty[t, :n, :] * value_scale

        verts, colors = [], []
        for e in range(n):
            points4 = np.column_stack([x_t[e], y_t[e], z_t[e]])
            if np.isnan(points4).any():
                continue
            quads = split_element_to_cell_quads(points4, n_cells)
            for c_idx, quad in enumerate(quads):
                val = values_t[e, c_idx]
                if not np.isfinite(val) or abs(val) >= INVALID_LIMIT:
                    continue
                verts.append(quad)
                rgba = list(cmap(norm(val)))
                rgba[3] = alpha
                colors.append(rgba)
        return verts, colors, n

    def update(frame_idx):
        t = frame_idx
        if color_scale_source == "frame":
            color_min, color_max = update_frame_norm(t)
        else:
            color_min, color_max = raw_min, raw_max

        verts, colors, n_active = build_cell_verts_and_colors(t)
        fracture_collection.set_verts(verts)
        fracture_collection.set_facecolors(colors)

        n_prev = 0 if t == 0 else min(ElementCount[t - 1], n_element_max)
        vals_now = safe_valid_values(CellProperty[t, :n_active, :].reshape(-1)) * value_scale
        p_min = np.min(vals_now) if vals_now.size else np.nan
        p_mean = np.mean(vals_now) if vals_now.size else np.nan
        p_max = np.max(vals_now) if vals_now.size else np.nan

        info_text.set_text(
            f"TimeIndex: {t}/{n_time - 1}\n"
            f"Active elements: {n_active}/{n_element_max}\n"
            f"Rendered cell patches: {len(verts)}\n"
            f"New elements: {n_active - n_prev}\n"
            f"{prop_name} min: {p_min:.4g} {value_unit}\n"
            f"{prop_name} mean: {p_mean:.4g} {value_unit}\n"
            f"{prop_name} max: {p_max:.4g} {value_unit}\n"
            f"Colorbar min: {color_min:.4g} {value_unit}\n"
            f"Colorbar max: {color_max:.4g} {value_unit}"
        )
        return [fracture_collection, info_text]

    ani = FuncAnimation(fig, update, frames=n_time, interval=interval, blit=False, repeat=repeat)

    if save_path:
        if save_path.lower().endswith(".gif"):
            ani.save(save_path, writer="pillow", fps=fps)
        else:
            ani.save(save_path, writer="ffmpeg", fps=fps)
        print(f"动画已保存：{save_path}")

    plt.show()


if __name__ == "__main__":
    target = r"D:\git\ufm-hdf5\data\hdf5_file\JY108-7HF_cluster1.h5"

    # # 示例 1：显示 WidthProfile，单位转为 mm。
    # animate_ufm_growth_cell_property(
    #     file_path=target,
    #     sim_id=24,
    #     prop_name="WidthProfile",
    #     interval=100,
    #     repeat=False,
    #     remove_origin=True,
    #     invert_z_axis=True,
    #     cmap_name="jet",
    #     alpha=1.0,
    #     edge_color="0.5",
    #     edge_width=0.013,
    #     view_elev=24,
    #     view_azim=-58,
    #     save_path=None,
    #     fps=5,
    #     value_scale=1000.0,
    #     value_unit="mm",
    #     color_scale_source="final",
    #     export_stats=True,
    # )

    # 示例 2：显示 FractureConductivity。
    # 需要显示导流能力时，注释上面的示例 1，取消下面代码注释。

    for i in range(15, 25):
        animate_ufm_growth_cell_property(
            file_path=target,
            sim_id=i,
            prop_name="AreaBasedPropConcProfile",
            interval=100,
            repeat=False,
            remove_origin=True,
            invert_z_axis=True,
            cmap_name="turbo",
            alpha=1.0,
            edge_color="0.5",
            edge_width=0.013,
            view_elev=24,
            view_azim=-58,
            save_path=None,
            fps=5,
            value_scale=1.0,
            value_unit="",
            color_scale_source="final",
            export_stats=True,
        )

