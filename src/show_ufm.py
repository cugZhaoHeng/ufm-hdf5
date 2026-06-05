from pathlib import Path
import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
HDF5_FILE = PROJECT_DIR / "data" / "hdf5_file"
WELL_DATA_DIR = PROJECT_DIR / "data" / "well_data"
OUT_PUT_IDR = PROJECT_DIR / "output"

project_root_str = str(PROJECT_DIR)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)
from utils.logger import create_logger

logger = create_logger(__name__)
from utils.date_util import get_current_time


def animate_temperature_fractures(
    h5_file_path,
    well_csv_path,
    property_name="Temperature",  # 需求3：属性名
    keep_aspect=True,  # 需求1：控制比例
    save_format=None,  # 需求2：可选 'gif', 'mp4', None
    show_all=True,          # 需求：是否展示全部
    stage_indices=[0, 1]    # 需求：若不展示全部，则指定索引列表
):
    all_sim_data = []

    # 1. 读取并绘制静态井轨迹
    df = pd.read_csv(well_csv_path, sep=r"\s+", comment="#")
    w_x, w_y, w_z = df["X"].values, df["Y"].values, df["TVD"].values  # 使用TVD

    print("读取 H5 数据中...")

    with h5py.File(h5_file_path, "r") as f:
        global_min = np.array([np.inf, np.inf, np.inf])
        global_max = np.array([-np.inf, -np.inf, -np.inf])

        temp_min = np.inf
        temp_max = -np.inf

        # -----------------------------------
        # 读取所有数据
        # -----------------------------------
        # 1. 自动发现所有存在的 Stage 路径
        # 假设 H5 结构为 "Fracture Simulation 45/...", 我们可以获取所有一级组
        all_groups = [g for g in f.keys() if "Fracture Simulation" in g]
        # 对组名进行排序（以防读取顺序混乱）
        all_groups.sort(key=lambda x: int(x.split()[-1])) 
        
        # 2. 根据参数筛选 Stage
        if show_all:
            target_groups = all_groups
        else:
            # 确保索引不越界
            target_groups = [all_groups[i] for i in stage_indices if i < len(all_groups)]
        
        print(f"准备处理的 Stage 路径: {target_groups}")
        
        for group_name in target_groups:
            base_path = f"{group_name}/Results/Bulk/UFM/FractureSet/Elements"
            points_path = f"{base_path}/Points"

            if f"{points_path}/X" not in f:
                continue

            # 几何数据
            X = f[f"{points_path}/X"][:]
            Y = f[f"{points_path}/Y"][:]
            Z = f[f"{points_path}/Z"][:]

            # 温度数据
            T = f[f"{base_path}/{property_name}"][:]

            all_sim_data.append({"id": group_name, "X": X, "Y": Y, "Z": Z, "T": T})

            # -----------------------------------
            # 更新全局坐标范围
            # -----------------------------------

            mask = ~np.isnan(X)

            if np.any(mask):
                global_min = np.minimum(
                    global_min, [np.nanmin(X), np.nanmin(Y), np.nanmin(Z)]
                )

                global_max = np.maximum(
                    global_max, [np.nanmax(X), np.nanmax(Y), np.nanmax(Z)]
                )

            # -----------------------------------
            # 更新温度范围
            # -----------------------------------

            temp_min = min(temp_min, np.nanmin(T))
            temp_max = max(temp_max, np.nanmax(T))

    if not all_sim_data:
        print("未发现有效数据")
        return

    # -----------------------------------
    # 创建颜色映射器
    # -----------------------------------

    cmap = mpl.colormaps["jet"]

    norm = mpl.colors.Normalize(vmin=temp_min, vmax=temp_max)

    # -----------------------------------
    # 创建画布
    # -----------------------------------

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(global_min[0], global_max[0])
    ax.set_ylim(global_min[1], global_max[1])
    ax.set_zlim(global_min[2], global_max[2])
    ax.invert_zaxis()
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Depth (m)")

    ax.plot(
        w_x, w_y, w_z, color="black", linewidth=2.5, label="Well Trajectory", zorder=10
    )
    ax.scatter(w_x[0], w_y[0], w_z[0], color="green", s=50, label="Well Head")
    # 保持 3D 图形比例协调
    if keep_aspect:
        try:
            ax.set_box_aspect((np.ptp(w_x) + 1, np.ptp(w_y) + 1, np.ptp(w_z) + 1))
        except Exception:
            logger.error("设置图形比例时发生错误，已跳过该步骤。")

    # -----------------------------------
    # 添加色阶
    # -----------------------------------
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label(property_name)
    # -----------------------------------
    # 创建 Poly3DCollection
    # -----------------------------------

    poly_collections = []

    for _ in all_sim_data:
        p = Poly3DCollection([], alpha=0.7, edgecolors="none")
        ax.add_collection3d(p)
        poly_collections.append(p)
    title = ax.set_title(f"Fracture {property_name} Evolution")
    # 信息提示框
    max_fractures = max([sim['X'].shape[1] for sim in all_sim_data])
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

    # -----------------------------------
    # 构建帧索引
    # -----------------------------------

    frames_map = []

    # 这里的 s_idx 是从 0 开始的索引，不是 sim_id
    for s_idx, sim in enumerate(all_sim_data):
        num_steps = sim["X"].shape[0]

        print(f"num_steps: {num_steps}")

        for t in range(num_steps):
            frames_map.append((s_idx, t))

    finalized_sims = set()

    # -----------------------------------
    # 动画更新函数
    # -----------------------------------

    def update(frame_idx):

        curr_sim_idx, curr_time_step = frames_map[frame_idx]

        sim = all_sim_data[curr_sim_idx]

        x_t = sim["X"][curr_time_step]
        y_t = sim["Y"][curr_time_step]
        z_t = sim["Z"][curr_time_step]

        temp_t = sim["T"][curr_time_step]

        # -----------------------------------
        # 有效面片
        # -----------------------------------

        valid = np.where(~np.isnan(x_t).any(axis=1))[0]

        verts = []

        face_colors = []

        # -----------------------------------
        # 构建所有面片
        # -----------------------------------

        for v in valid:
            quad = np.stack([x_t[v], y_t[v], z_t[v]], axis=-1)

            verts.append(quad)

            # 当前面片温度
            temp = temp_t[v]

            # 转换成颜色
            color = cmap(norm(temp))

            face_colors.append(color)

        # -----------------------------------
        # 更新Mesh
        # -----------------------------------

        poly_collections[curr_sim_idx].set_verts(verts)

        poly_collections[curr_sim_idx].set_facecolor(face_colors)

        # -----------------------------------
        # 固定之前完成的Stage
        # -----------------------------------
        # i 是之前的模拟索引，不是 sim_id
        for i in range(curr_sim_idx):
            if i not in finalized_sims:
                last_x = all_sim_data[i]["X"][-1]
                last_y = all_sim_data[i]["Y"][-1]
                last_z = all_sim_data[i]["Z"][-1]

                last_t = all_sim_data[i]["T"][-1]

                v_idx = np.where(~np.isnan(last_x).any(axis=1))[0]

                final_verts = []

                final_colors = []

                for v in v_idx:
                    quad = np.stack([last_x[v], last_y[v], last_z[v]], axis=-1)

                    final_verts.append(quad)

                    color = cmap(norm(last_t[v]))

                    final_colors.append(color)

                poly_collections[i].set_verts(final_verts)

                poly_collections[i].set_facecolor(final_colors)

                finalized_sims.add(i)

        # 需求4：格式化时间步 n/time
        total_steps = sim["X"].shape[0]
        step_str = f"{curr_time_step + 1}/{total_steps}"

        # 需求3：更新标题
        title.set_text(
            f"{property_name} Evolution | Stage: {sim['id']} | Step: {step_str}"
        )

        # 需求5：更新左下角信息
        text_info.set_text(f"Max Fractures: {len(x_t)}")

        return poly_collections + [title]

    # -----------------------------------
    # 创建动画
    # -----------------------------------

    ani = FuncAnimation(
        fig, update, frames=len(frames_map), interval=10, blit=False, repeat=False
    )

    # 需求2：保存逻辑
    if save_format:
        logger.info("动画创建完成，正在保存...")
        output_path = OUT_PUT_IDR / f"fracture_{property_name}_evolution_{get_current_time()}.{save_format}"
        if save_format == "mp4":
            ani.save(output_path, writer="ffmpeg", fps=30)
        elif save_format == "gif":
            ani.save(output_path, writer="pillow", fps=30)
        logger.info(f"动画已保存至: {output_path}")

    plt.show()


if __name__ == "__main__":
    h5_file_path = HDF5_FILE / "JY68-4HF.h5"
    well_csv_path = WELL_DATA_DIR / "68-4HF"

    animate_temperature_fractures(
        h5_file_path=h5_file_path,
        well_csv_path=well_csv_path,
        keep_aspect=False,
        # save_format="gif",
        property_name="NetPressure",
        show_all=False,
        stage_indices=[0, 10,-1]
    )
