# -*- coding: utf-8 -*-
"""
UFM 多簇裂缝动态同步生长与井轨迹联合展示
"""

from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import re
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation

INVALID_LIMIT = 1.0e30
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
HDF5_FILE = PROJECT_DIR / "data" / "hdf5_file"
well_data_path = PROJECT_DIR / "data" / "well_data" / "68-4HF"
OUT_PUT_IDR = PROJECT_DIR / "output"


def list_fracture_simulation_ids(f):
    """列出 HDF5 根目录下全部 Fracture Simulation {sim_id}。"""
    sim_ids = []
    pattern = re.compile(r"^Fracture Simulation\s+(\d+)$")
    for name in f.keys():
        match = pattern.match(name)
        if match:
            sim_ids.append(name.split()[-1])
    return sorted(sim_ids)


def safe_valid_values(arr):
    """去除 NaN、Inf 和 UFM 常见无效大数值。"""
    arr = np.asarray(arr, dtype=float)
    mask = np.isfinite(arr) & (np.abs(arr) < INVALID_LIMIT)
    return arr[mask]


def read_cell_property(f, base_path, prop_name="WidthProfile"):
    """读取 cell 级属性。"""
    prop_path = f"{base_path}/Elements/Cells/{prop_name}"
    if prop_path not in f:
        raise KeyError(f"未找到 cell 属性路径：{prop_path}")
    data = np.asarray(f[prop_path][:], dtype=float)
    if data.ndim != 3:
        raise ValueError(f"{prop_name} 应为三维数组 (time, element, cell)")
    data = np.where(np.abs(data) < INVALID_LIMIT, data, np.nan)
    return data


def split_element_to_cell_quads(points4, n_cells):
    """将一个四边形 element 近似剖分为 n_cells 个小四边形。"""
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


def compute_direct_vmin_vmax(values):
    """计算色标的数值范围。"""
    values = safe_valid_values(values)
    if values.size == 0:
        raise ValueError("没有有效属性值，无法计算色标范围。")
    raw_min = float(np.nanmin(values))
    raw_max = float(np.nanmax(values))
    vmin, vmax = raw_min, raw_max
    if np.isclose(vmin, vmax):
        delta = 0.5 if np.isclose(vmin, 0.0) else abs(vmin) * 0.05
        vmin = raw_min - delta
        vmax = raw_max + delta
    return raw_min, raw_max, vmin, vmax


def load_single_simulation_data(f, sim_id, prop_name="WidthProfile"):
    """加载单个 Simulation 的完整时间序列数据。"""
    sim_key = f"Fracture Simulation {sim_id}"
    if isinstance(sim_id, int):
        sim_key = f"Fracture Simulation {sim_id:02d}"

    if sim_key not in f:
        sim_key = f"Fracture Simulation {sim_id}"
        if sim_key not in f:
            raise KeyError(f"未找到模拟数据组: {sim_key}")

    base_path = f"{sim_key}/Results/Bulk/UFM/FractureSet"
    points_path = f"{base_path}/Elements/Points"
    element_count_path = f"{base_path}/ElementCount"

    X = np.asarray(f[f"{points_path}/X"][:], dtype=float)
    Y = np.asarray(f[f"{points_path}/Y"][:], dtype=float)
    Z = np.asarray(f[f"{points_path}/Z"][:], dtype=float)
    ElementCount = np.asarray(f[element_count_path][:]).reshape(-1).astype(int)
    CellProperty = read_cell_property(f, base_path, prop_name=prop_name)

    n_time = min(X.shape[0], len(ElementCount), CellProperty.shape[0])
    n_element_max = min(X.shape[1], CellProperty.shape[1])
    n_cells = CellProperty.shape[2]

    return {
        "sim_id": sim_id,
        "n_time": n_time,
        "n_element_max": n_element_max,
        "n_cells": n_cells,
        "X": X,
        "Y": Y,
        "Z": Z,
        "ElementCount": ElementCount,
        "CellProperty": CellProperty
    }


def load_well_trajectory(file_path):
    """从文本中读取井轨迹数据。"""
    try:
        df = pd.read_csv(file_path, sep=r'\s+', comment='#')
        return {
            "X": df['X'].values,
            "Y": df['Y'].values,
            "Z": df['Z'].values,
            "TVD": df['TVD'].values,
        }
    except FileNotFoundError:
        raise FileNotFoundError(f"未找到井轨迹文件：{file_path}，请确认路径是否正确。")


def animate_fractures_and_well_simultaneously(
    h5_file_path,
    well_file_path,
    sim_ids,
    prop_name="WidthProfile",
    interval=150,
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
    fps=10,
    value_scale=1000.0,
    value_unit="mm",
):
    """
    联合展示：静态井轨迹 + 动态同步生长裂缝
    """
    # 1. 读取井轨迹
    well_data = load_well_trajectory(well_file_path)
    print(f"成功加载井轨迹数据：共 {len(well_data['X'])} 个测点")

    # 2. 读取裂缝模拟数据
    sim_data_list = []
    with h5py.File(h5_file_path, "r") as f:
        if sim_ids is None:
            sim_ids = list_fracture_simulation_ids(f)
        for s_id in sim_ids:
            try:
                data = load_single_simulation_data(f, s_id, prop_name)
                sim_data_list.append(data)
                print(f"  - 成功加载裂缝 sim {s_id}: 共 {data['n_time']} 个时间步")
            except Exception as e:
                print(f"  - 跳过裂缝 sim {s_id}，原因: {e}")

    if not sim_data_list:
        raise ValueError("没有加载到任何有效的裂缝数据。")

    # 确定最大相同步数
    total_frames = min(s_data["n_time"] for s_data in sim_data_list)
    print(f"动画同步演化步数确定为: {total_frames} 帧")

    # 3. 确定全局基准原点 (X0, Y0, Z0) —— 包含井和裂缝
    if remove_origin:
        all_x = list(well_data["X"])
        all_y = list(well_data["Y"])
        all_z = list(well_data["Z"])

        for s_data in sim_data_list:
            for t in range(total_frames):
                n_active = min(s_data["ElementCount"][t], s_data["n_element_max"])
                if n_active > 0:
                    all_x.extend(s_data["X"][t, :n_active, :].flatten())
                    all_y.extend(s_data["Y"][t, :n_active, :].flatten())
                    all_z.extend(s_data["Z"][t, :n_active, :].flatten())

        # 清除 NaN
        all_x = safe_valid_values(all_x)
        all_y = safe_valid_values(all_y)
        all_z = safe_valid_values(all_z)

        X0 = np.min(all_x) if len(all_x) > 0 else 0.0
        Y0 = np.min(all_y) if len(all_y) > 0 else 0.0
        Z0 = np.min(all_z) if len(all_z) > 0 else 0.0
    else:
        X0 = Y0 = Z0 = 0.0

    # 4. 平移数据坐标并确定 3D 边界范围
    well_x_plot = well_data["X"] - X0
    well_y_plot = well_data["Y"] - Y0
    well_z_plot = well_data["Z"] - Z0

    global_min = np.array([np.nanmin(well_x_plot), np.nanmin(well_y_plot), np.nanmin(well_z_plot)])
    global_max = np.array([np.nanmax(well_x_plot), np.nanmax(well_y_plot), np.nanmax(well_z_plot)])

    for s_data in sim_data_list:
        s_data["X_plot"] = s_data["X"] - X0
        s_data["Y_plot"] = s_data["Y"] - Y0
        s_data["Z_plot"] = s_data["Z"] - Z0
        
        t_final = total_frames - 1
        n_active = min(s_data["ElementCount"][t_final], s_data["n_element_max"])
        if n_active > 0:
            x_act = s_data["X_plot"][t_final, :n_active, :]
            y_act = s_data["Y_plot"][t_final, :n_active, :]
            z_act = s_data["Z_plot"][t_final, :n_active, :]
            global_min = np.minimum(global_min, [np.nanmin(x_act), np.nanmin(y_act), np.nanmin(z_act)])
            global_max = np.maximum(global_max, [np.nanmax(x_act), np.nanmax(y_act), np.nanmax(z_act)])

    span = global_max - global_min
    pad = np.maximum(span * 0.08, 1.0)
    global_min -= pad
    global_max += pad

    # 5. 统计全局属性极值以统一裂缝色标
    all_prop_values = []
    for s_data in sim_data_list:
        for t in range(total_frames):
            n_active = min(s_data["ElementCount"][t], s_data["n_element_max"])
            if n_active > 0:
                vals = s_data["CellProperty"][t, :n_active, :] * value_scale
                all_prop_values.append(vals.reshape(-1))
                
    flat_values = np.concatenate(all_prop_values)
    raw_min, raw_max, vmin, vmax = compute_direct_vmin_vmax(flat_values)

    # 创建绘图画布
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

    # 6. 【核心步骤】静态绘制井轨迹
    # 用醒目的粗黑实线绘制井轨迹，并标记井口
    ax.plot(well_x_plot, well_y_plot, well_z_plot, color='black', linewidth=3.0, label="Well Path", zorder=5)
    ax.scatter(well_x_plot[0], well_y_plot[0], well_z_plot[0], color='darkgreen', marker='^', s=120, label="Well Head", zorder=6)
    ax.legend(loc="upper left")

    # 用于追踪动态裂缝 Collection 对象的生命周期
    current_collection = [None]

    fig.suptitle(
        f"UFM Fractures & Wellpath Joint Visualization | Cell {prop_name}",
        fontsize=16,
        fontweight="bold",
        y=0.965,
    )

    # 信息提示框
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

    # 颜色条
    cmap = mpl.colormaps[cmap_name]
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.90, 0.18, 0.026, 0.64])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(f"Fracture {prop_name} ({value_unit})", fontsize=11, fontweight="bold")
    cbar.ax.tick_params(labelsize=10)
    if raw_min != raw_max:
        cbar.set_ticks(np.linspace(vmin, vmax, 6))
    cbar.ax.set_title("global limit", fontsize=9, pad=8)

    def build_verts_for_sim_at_t(s_data, t):
        """为特定模拟提取并剖分时间步 t 的所有小面片几何与颜色"""
        n_active = min(s_data["ElementCount"][t], s_data["n_element_max"])
        x_t = s_data["X_plot"][t, :n_active, :]
        y_t = s_data["Y_plot"][t, :n_active, :]
        z_t = s_data["Z_plot"][t, :n_active, :]
        w_t = s_data["CellProperty"][t, :n_active, :] * value_scale
        n_cells = s_data["n_cells"]

        local_verts = []
        local_colors = []

        for e in range(n_active):
            points4 = np.column_stack([x_t[e], y_t[e], z_t[e]])
            if np.isnan(points4).any():
                continue
            values_cells = w_t[e, :]
            if np.all(~np.isfinite(values_cells)):
                continue

            quads = split_element_to_cell_quads(points4, n_cells=n_cells)
            for c_idx, quad in enumerate(quads):
                val = values_cells[c_idx]
                if not np.isfinite(val) or abs(val) >= INVALID_LIMIT:
                    continue
                local_verts.append(quad)
                rgba = list(cmap(norm(val)))
                rgba[3] = alpha
                local_colors.append(rgba)

        return local_verts, local_colors

    def update(t):
        # 动态清理上一帧的裂缝 Collection（不影响静态绘制的井轨迹）
        if current_collection[0] is not None:
            current_collection[0].remove()
            current_collection[0] = None

        all_verts = []
        all_colors = []
        summary_info = []

        # 遍历所有被加载的裂缝，提取相同时间步 t 的状态并合并
        for s_data in sim_data_list:
            s_id = s_data["sim_id"]
            v, c = build_verts_for_sim_at_t(s_data, t)
            all_verts.extend(v)
            all_colors.extend(c)
            summary_info.append(f"Sim {s_id}: Active Elements={s_data['ElementCount'][t]}, Patches={len(v)}")

        # 重新创建裂缝渲染集合
        new_col = Poly3DCollection(
            all_verts,
            facecolors=all_colors,
            alpha=alpha,
            edgecolors=edge_color,
            linewidths=edge_width,
            zorder=3,
        )
        ax.add_collection3d(new_col)
        current_collection[0] = new_col

        # 更新文本
        text_content = (
            f"Time Step: {t}/{total_frames - 1}\n"
            f"Simulations Count: {len(sim_data_list)}\n"
            f"Total Rendered Patches: {len(all_verts)}\n"
            + "\n".join(summary_info)
        )
        info_text.set_text(text_content)

        return [new_col, info_text]

    print("准备就绪，开始生成联合动画...")
    
    # 动画实例
    ani = FuncAnimation(
        fig,
        update,
        frames=total_frames,
        interval=interval,
        blit=False,
        repeat=repeat,
    )
    plt.show()
    if save_path:
        save_path = str(save_path)
        print(f"正在保存联合动画到: {save_path} ... (请耐心等待渲染)")
        if save_path.lower().endswith(".gif"):
            ani.save(save_path, writer="pillow", fps=fps)
        else:
            ani.save(save_path, writer="ffmpeg", fps=fps)
        print("保存完毕！")
        # 重新回到最后一帧，保证 plt.show() 弹窗有画面
        update(total_frames - 1)

    


if __name__ == "__main__":
    # 路径配置
    target_h5 = HDF5_FILE / "JY68-4HF.h5"
    well_txt = well_data_path # 确保该文件在脚本目录下
    
    save_dir = OUT_PUT_IDR
    save_dir.mkdir(parents=True, exist_ok=True)
    combined_save_path = save_dir / "JY108-7HF_well_and_fracture.gif"

    # 要显示的裂缝组
    custom_sim_ids = ["45"]

    animate_fractures_and_well_simultaneously(
        h5_file_path=target_h5,
        well_file_path=well_txt,
        sim_ids=custom_sim_ids,
        prop_name="WidthProfile",
        interval=120,
        repeat=False,
        remove_origin=True,      # 统一平移防止渲染抖动
        invert_z_axis=True,      # 保持地学深度向下为正
        cmap_name="jet",
        alpha=0.9,               # 设置轻微透明，方便透视被遮挡的井轨迹或相邻裂缝
        edge_width=0.01,
        save_path=str(combined_save_path), # 若不需要保存，可改为 None
    )