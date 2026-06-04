# -*- coding: utf-8 -*-
"""
UFM 多簇裂缝动态同步生长展示：按输入列表 sim_ids 同时展示每个时间步的变化。
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


def animate_multiple_ufm_growth_simultaneously(
    file_path,
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
    同时展示多条裂缝在每一个时间步的同步生长状态。
    """
    print("开始读取同步展示的 H5 数据...")
    sim_data_list = []
    
    with h5py.File(file_path, "r") as f:
        if sim_ids is None:
            sim_ids = list_fracture_simulation_ids(f)
            
        for s_id in sim_ids:
            try:
                data = load_single_simulation_data(f, s_id, prop_name)
                sim_data_list.append(data)
                print(f"  - 成功加载 sim {s_id}: 共 {data['n_time']} 个时间步")
            except Exception as e:
                print(f"  - 跳过 sim {s_id}，错误原因: {e}")

    if not sim_data_list:
        raise ValueError("未找到任何有效的裂缝模拟数据。")

    # 1. 验证各裂缝时间步数并确定总动画帧数
    total_frames = min(s_data["n_time"] for s_data in sim_data_list)
    print(f"动画总时间步数确定为: {total_frames} 帧")

    # 2. 确定全局基准原点 (X0, Y0, Z0)
    if remove_origin:
        all_x_min, all_y_min, all_z_min = [], [], []
        for s_data in sim_data_list:
            for t in range(total_frames):
                n_active = min(s_data["ElementCount"][t], s_data["n_element_max"])
                if n_active > 0:
                    all_x_min.append(np.nanmin(s_data["X"][t, :n_active, :]))
                    all_y_min.append(np.nanmin(s_data["Y"][t, :n_active, :]))
                    all_z_min.append(np.nanmin(s_data["Z"][t, :n_active, :]))
        X0 = min(all_x_min) if all_x_min else 0.0
        Y0 = min(all_y_min) if all_y_min else 0.0
        Z0 = min(all_z_min) if all_z_min else 0.0
    else:
        X0 = Y0 = Z0 = 0.0

    # 3. 坐标平移并计算全局 3D 边界范围
    global_min = np.array([np.inf, np.inf, np.inf])
    global_max = np.array([-np.inf, -np.inf, -np.inf])

    for s_data in sim_data_list:
        s_data["X_plot"] = s_data["X"] - X0
        s_data["Y_plot"] = s_data["Y"] - Y0
        s_data["Z_plot"] = s_data["Z"] - Z0
        
        # 使用最后一个有效时间步确定显示视窗边界
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

    # 4. 统计全局属性极值以统一色标
    all_prop_values = []
    for s_data in sim_data_list:
        for t in range(total_frames):
            n_active = min(s_data["ElementCount"][t], s_data["n_element_max"])
            if n_active > 0:
                vals = s_data["CellProperty"][t, :n_active, :] * value_scale
                all_prop_values.append(vals.reshape(-1))
                
    flat_values = np.concatenate(all_prop_values)
    raw_min, raw_max, vmin, vmax = compute_direct_vmin_vmax(flat_values)

    print("=" * 60)
    print(f"同步展示裂缝数量: {len(sim_data_list)}")
    print(f"统一色标范围: [{vmin:.4f}, {vmax:.4f}] {value_unit}")
    print("=" * 60)

    # 设置画布
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

    # 用于动态跟踪 Collection 对象的生命周期
    current_collection = [None]

    fig.suptitle(
        f"UFM Simultaneous Fracture Growth | Cell {prop_name} Distribution",
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
    cbar.set_label(f"{prop_name} ({value_unit})", fontsize=11, fontweight="bold")
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
        # 清理上一帧的 3D Collection，防止部分环境下 3D 图层无法更新或覆盖重叠
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

        # 每一帧重建新的 Poly3DCollection 对象，彻底避开 Matplotlib 3D 渲染无法更新的 Bug
        new_col = Poly3DCollection(
            all_verts,
            facecolors=all_colors,
            alpha=alpha,
            edgecolors=edge_color,
            linewidths=edge_width,
        )
        ax.add_collection3d(new_col)
        current_collection[0] = new_col

        # 更新提示文本框
        text_content = (
            f"Time Step: {t}/{total_frames - 1}\n"
            f"Simulations Count: {len(sim_data_list)}\n"
            f"Total Rendered Patches: {len(all_verts)}\n"
            + "\n".join(summary_info)
        )
        info_text.set_text(text_content)

        # 返回需要更新的 artist 对象（用于 blit 优化，虽然 blit=False，但也需要返回以保持完整性）
        return [new_col, info_text]

    print("准备就绪，开始生成同步生长动画...")
    
    # 构建 FuncAnimation
    ani = FuncAnimation(
        fig,
        update,
        frames=total_frames,
        interval=interval,
        blit=False,
        repeat=repeat,
    )

    # 解决 "save() 消耗迭代器" 导致 "plt.show() 无法绘图" 的冲突
    if save_path:
        # 如果需要保存，先保存动画。
        # 部分后端在 save() 之后需要重新触发绘制才能展示 show()，这里重新刷新一下当前状态
        save_path = str(save_path)
        print(f"正在保存动画到: {save_path} ... (可能需要数秒至数分钟，请稍候)")
        if save_path.lower().endswith(".gif"):
            ani.save(save_path, writer="pillow", fps=fps)
        else:
            ani.save(save_path, writer="ffmpeg", fps=fps)
        print("动画保存完毕！")
        
        # 重新初始化最后一帧几何数据，使用户能在弹出的窗口中看到最终渲染结果
        update(total_frames - 1)

    # 弹出交互式渲染窗口
    plt.show()


if __name__ == "__main__":
    target_h5 = HDF5_FILE / "JY108-7HF_cluster1.h5"
    save_dir = OUT_PUT_IDR
    
    # 确保输出目录存在
    save_dir.mkdir(parents=True, exist_ok=True)
    c_save_path = save_dir / "JY108-7HF_cluster1_WidthProfile_02.gif"
    
    custom_sim_ids = ["01", "02"]

    animate_multiple_ufm_growth_simultaneously(
        file_path=target_h5,
        sim_ids=custom_sim_ids,
        prop_name="WidthProfile",
        interval=120,            # 帧率间隔 (ms)
        repeat=False,            # 是否循环播放
        remove_origin=True,      # 平移坐标至局部原点以防止抖动
        invert_z_axis=True,      # 深度轴向下为正
        cmap_name="jet",
        alpha=1.0,
        value_scale=1000.0,      # 宽度从米(m)转为毫米(mm)
        value_unit="mm",
        save_path=str(c_save_path),  # 如果只想查看画面不想写盘，可设为 None
    )