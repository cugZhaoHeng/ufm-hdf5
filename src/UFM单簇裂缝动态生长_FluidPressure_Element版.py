# -*- coding: utf-8 -*-
"""
UFM 单簇裂缝动态生长展示：按 element 显示 FluidPressure 分布

本版优化内容：
1. 使用 ElementCount 判断每个时间步有效 element 数量；
2. 以 element 为单元显示 FluidPressure / NetPressure / AvgWidth 等属性；
3. 默认使用高对比色标 turbo，并提高裂缝面片不透明度；
4. 将 TimeIndex / New elements 等信息框移动到左下角；
5. 将色标 colorbar 独立放到图像最右侧，避免和三维图重叠；
6. 支持使用百分位色阶增强颜色对比，避免少数极值导致整体颜色过浅。
"""

from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation


INVALID_LIMIT = 1.0e30
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
HDF5_FILE = PROJECT_DIR / "data" / "hdf5_file"
OUT_PUT_IDR = PROJECT_DIR / "output"


def safe_valid_values(arr):
    """去除 NaN、Inf 和 UFM 常见无效大数值。"""
    arr = np.asarray(arr, dtype=float)
    mask = np.isfinite(arr) & (np.abs(arr) < INVALID_LIMIT)
    return arr[mask]


def read_element_property(f, base_path, prop_name):
    """
    读取 element 级属性，例如 FluidPressure、NetPressure、AvgWidth 等。

    返回数组尽量整理为 shape = (time, element)。
    如果原始数据是 (time, element, 1) 或 (time, element, n)，则对最后维度取均值。
    """
    prop_path = f"{base_path}/Elements/{prop_name}"

    if prop_path not in f:
        raise KeyError(f"未找到属性路径：{prop_path}")

    data = np.asarray(f[prop_path][:], dtype=float)

    if data.ndim == 1:
        data = data.reshape(1, -1)
    elif data.ndim == 2:
        pass
    elif data.ndim >= 3:
        data = np.where(np.abs(data) < INVALID_LIMIT, data, np.nan)
        data = np.nanmean(data, axis=tuple(range(2, data.ndim)))
    else:
        raise ValueError(f"{prop_name} 维度异常：{data.shape}")

    return data


def animate_ufm_growth_with_element_property(
    file_path,
    sim_id=24,
    prop_name="NetPressure",
    interval=300,
    repeat=False,
    remove_origin=True,
    invert_z_axis=True,
    cmap_name="turbo",
    alpha=0.96,
    edge_color="0.12",
    edge_width=0.06,
    view_elev=24,
    view_azim=-58,
    save_path=None,
    fps=5,
    use_percentile_scale=True,
    percentile_low=2,
    percentile_high=98,
):
    """
    动态展示 UFM 裂缝生长，并按照 element 属性值着色。

    参数说明：
    file_path: H5 文件路径
    sim_id: Fracture Simulation 编号
    prop_name: 要显示的 element 属性，默认 FluidPressure
    interval: 动画每帧间隔，单位 ms；数值越大越慢
    repeat: 是否循环播放
    remove_origin: 是否去除局部坐标原点
    invert_z_axis: 是否反转 Z 轴
    cmap_name: 色阶名称；推荐 turbo / jet / plasma / inferno
    alpha: 裂缝面透明度；越接近 1，颜色越浓
    edge_color: element 边线颜色
    edge_width: element 边线宽度
    view_elev/view_azim: 三维视角
    save_path: 保存路径，例如 r"D:\\fluid_pressure_growth.mp4" 或 r"D:\\fluid_pressure_growth.gif"
    fps: 保存动画帧率
    use_percentile_scale: 是否使用百分位色阶增强对比
    percentile_low/percentile_high: 百分位色阶范围
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
            f"{base_path}/Elements/{prop_name}",
        ]

        for p in required_paths:
            if p not in f:
                raise KeyError(f"未找到必要路径：{p}")

        X = np.asarray(f[f"{points_path}/X"][:], dtype=float)
        Y = np.asarray(f[f"{points_path}/Y"][:], dtype=float)
        Z = np.asarray(f[f"{points_path}/Z"][:], dtype=float)

        ElementCount = np.asarray(f[element_count_path][:]).reshape(-1).astype(int)
        prop = read_element_property(f, base_path, prop_name)

        print(f"X shape: {X.shape}")
        print(f"Y shape: {Y.shape}")
        print(f"Z shape: {Z.shape}")
        print(f"ElementCount shape: {ElementCount.shape}")
        print(f"{prop_name} shape: {prop.shape}")

    n_time = min(X.shape[0], len(ElementCount), prop.shape[0])
    n_element_max = X.shape[1]

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

    # 计算坐标范围：严格按 ElementCount 统计有效 element。
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

    # 计算属性色阶范围：只统计有效时间步、有效 element。
    all_prop_values = []
    for t in range(n_time):
        n_active = min(ElementCount[t], n_element_max, prop.shape[1])
        if n_active <= 0:
            continue
        all_prop_values.append(prop[t, :n_active])

    all_prop_values = safe_valid_values(np.concatenate(all_prop_values))
    if all_prop_values.size == 0:
        raise ValueError(f"{prop_name} 没有有效数值，无法显示属性分布。")

    raw_min = float(np.nanmin(all_prop_values))
    raw_max = float(np.nanmax(all_prop_values))

    if use_percentile_scale:
        vmin = float(np.nanpercentile(all_prop_values, percentile_low))
        vmax = float(np.nanpercentile(all_prop_values, percentile_high))
        if np.isclose(vmin, vmax):
            vmin, vmax = raw_min, raw_max
    else:
        vmin, vmax = raw_min, raw_max

    print(f"{prop_name} raw min = {raw_min:.6g}, raw max = {raw_max:.6g}")
    print(f"{prop_name} color scale vmin = {vmin:.6g}, vmax = {vmax:.6g}")

    cmap = mpl.colormaps[cmap_name]
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)

    # 创建画布。
    fig = plt.figure(figsize=(14.5, 8.8))
    ax = fig.add_subplot(111, projection="3d")

    # 右侧留出独立色标区域；底部留出信息框区域。
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

    # 图像更清爽：浅色网格 + 白色背景。
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
        f"UFM Fracture Growth - {prop_name} Distribution | Fracture Simulation {sim_id}",
        fontsize=16,
        fontweight="bold",
        y=0.965,
    )

    # 信息框移动到左下角，避免和标题重叠。
    info_text = ax.text2D(
        0.018,
        0.025,
        "",
        transform=ax.transAxes,
        fontsize=10.5,
        color="#09203f",
        va="bottom",
        ha="left",
        bbox=dict(
            boxstyle="round,pad=0.45",
            facecolor="#eef6ff",
            edgecolor="#1f6fd1",
            alpha=0.94,
        ),
    )

    # 色标独立放到最右侧。
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.90, 0.18, 0.026, 0.64])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(prop_name, fontsize=11, fontweight="bold")
    cbar.ax.tick_params(labelsize=10)

    # 色标旁增加说明，明确是否使用百分位增强。
    if use_percentile_scale:
        cbar.ax.set_title(
            f"P{percentile_low}-P{percentile_high}",
            fontsize=9,
            pad=8,
        )

    def build_verts_and_colors(t):
        """根据时间步 t 构造当前有效 element 的面片和颜色。"""
        n_active = min(ElementCount[t], n_element_max, prop.shape[1])

        x_t = X_plot[t, :n_active, :]
        y_t = Y_plot[t, :n_active, :]
        z_t = Z_plot[t, :n_active, :]
        prop_t = prop[t, :n_active]

        valid = np.where(
            np.isfinite(prop_t)
            & (np.abs(prop_t) < INVALID_LIMIT)
            & ~np.isnan(x_t).any(axis=1)
            & ~np.isnan(y_t).any(axis=1)
            & ~np.isnan(z_t).any(axis=1)
        )[0]

        verts = [np.stack([x_t[e], y_t[e], z_t[e]], axis=-1) for e in valid]

        # 颜色单独设置 alpha，避免颜色过浅。
        colors = cmap(norm(prop_t[valid]))
        colors[:, 3] = alpha
        return verts, colors, valid, n_active

    def update(frame_idx):
        t = frame_idx
        verts, colors, valid, n_active = build_verts_and_colors(t)

        fracture_collection.set_verts(verts)
        fracture_collection.set_facecolors(colors)

        if t == 0:
            n_prev = 0
        else:
            n_prev = min(ElementCount[t - 1], n_element_max)
        n_new = n_active - n_prev

        prop_now = safe_valid_values(prop[t, :min(n_active, prop.shape[1])])
        if prop_now.size > 0:
            prop_min = np.min(prop_now)
            prop_max = np.max(prop_now)
            prop_mean = np.mean(prop_now)
        else:
            prop_min = prop_max = prop_mean = np.nan

        info_text.set_text(
            f"TimeIndex: {t}/{n_time - 1}\n"
            f"Active elements: {n_active}/{n_element_max}\n"
            f"New elements: {n_new}\n"
            f"{prop_name} min: {prop_min:.3g}\n"
            f"{prop_name} mean: {prop_mean:.3g}\n"
            f"{prop_name} max: {prop_max:.3g}"
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
    save_dir = OUT_PUT_IDR
    c_save_path = save_dir / "JY108-7HF_cluster1_WidthProfile.gif"

    animate_ufm_growth_with_element_property(
        file_path=target,
        sim_id=24,
        prop_name="Cells/WidthProfile",
        interval=300,
        repeat=False,
        remove_origin=True,
        invert_z_axis=True,
        cmap_name="turbo",      # 高对比色标，颜色比 coolwarm 更浓
        alpha=0.96,              # 提高不透明度，避免颜色太浅
        edge_color="0.12",
        edge_width=0.06,
        view_elev=24,
        view_azim=-58,
        save_path=str(c_save_path),
        fps=5,
        use_percentile_scale=True,
        percentile_low=2,
        percentile_high=98,
    )
