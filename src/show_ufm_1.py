from pathlib import Path
import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
import os

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
HDF5_FILE = PROJECT_DIR / "data" / "hdf5_file"

project_root_str = str(PROJECT_DIR)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)
from utils.logger import create_logger
logger = create_logger(__name__)

def animate_sequential_growth_fixed(file_path):
    sim_ids = range(45, 47)
    all_sim_data = []

    logger.info("读取 H5 数据中...")
    with h5py.File(file_path, 'r') as f:
        global_min = np.array([np.inf, np.inf, np.inf])
        global_max = np.array([-np.inf, -np.inf, -np.inf])

        for sim_id in sim_ids:
            logger.info(f"sim_id: {sim_id}")
            path = f"Fracture Simulation {sim_id}/Results/Bulk/UFM/FractureSet/Elements/Points"
            if f"{path}/X" not in f:
                continue

            X, Y, Z = f[f"{path}/X"][:], f[f"{path}/Y"][:], f[f"{path}/Z"][:]
            all_sim_data.append({'id': sim_id, 'X': X, 'Y': Y, 'Z': Z})

            # 更新全局坐标范围以便固定轴比例
            mask = ~np.isnan(X)
            if np.any(mask):
                global_min = np.minimum(global_min, [np.nanmin(X), np.nanmin(Y), np.nanmin(Z)])
                global_max = np.maximum(global_max, [np.nanmax(X), np.nanmax(Y), np.nanmax(Z)])

    if not all_sim_data:
        logger.info("未发现有效数据。")
        return

    # 2. 设置画布
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(global_min[0], global_max[0])
    ax.set_ylim(global_min[1], global_max[1])
    ax.set_zlim(global_min[2], global_max[2])
    ax.invert_zaxis()
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Depth (m)')

    # 修复 get_cmap 警告
    color_map = mpl.colormaps['tab10']
    poly_collections = []
    for i in range(len(all_sim_data)):
        p = Poly3DCollection([], alpha=0.5, facecolors=color_map(i), edgecolors='none')
        ax.add_collection3d(p)
        poly_collections.append(p)

    title = ax.set_title('Sequential Multi-Stage Fracture Growth')

    # 3. 映射帧索引 (Stage_Index, Time_Step)
    frames_map = []
    for s_idx, sim in enumerate(all_sim_data):
        num_steps = sim['X'].shape[0]
        logger.info(f"num_steps: {num_steps}")
        for t in range(num_steps):
            frames_map.append((s_idx, t))
    logger.info(f"Total frames: {len(frames_map)}")
    # 用于记录哪些模拟已经完成了生长，避免重复计算
    finalized_sims = set()

    # 4. 动画更新函数
    def update(frame_idx):
        curr_sim_idx, curr_time_step = frames_map[frame_idx]

        # A. 更新当前正在生长的模拟方案
        sim = all_sim_data[curr_sim_idx]
        x_t, y_t, z_t = sim['X'][curr_time_step], sim['Y'][curr_time_step], sim['Z'][curr_time_step]

        valid = np.where(~np.isnan(x_t).any(axis=1))[0]
        verts = [np.stack([x_t[v], y_t[v], z_t[v]], axis=-1) for v in valid]
        poly_collections[curr_sim_idx].set_verts(verts)

        # B. 检查并固定之前已经长完的模拟方案
        for i in range(curr_sim_idx):
            if i not in finalized_sims:
                last_x = all_sim_data[i]['X'][-1]
                last_y = all_sim_data[i]['Y'][-1]
                last_z = all_sim_data[i]['Z'][-1]
                v_idx = np.where(~np.isnan(last_x).any(axis=1))[0]
                v_final = [np.stack([last_x[v], last_y[v], last_z[v]], axis=-1) for v in v_idx]
                poly_collections[i].set_verts(v_final)
                finalized_sims.add(i)

        title.set_text(f'Stage: {all_sim_data[curr_sim_idx]["id"]} | Time Step: {curr_time_step}')
        return poly_collections + [title]

    # 5. 执行动画
    # 注意：这里设置 blit=False 是为了保证 3D 渲染的兼容性
    # ani = FuncAnimation(fig, update, frames=len(frames_map), interval=200, blit=False)

    # 如果需要保存视频，取消下面行的注释
    # ani.save('sequential_fractures.mp4', writer='ffmpeg', fps=15)

    plt.show()


if __name__ == "__main__":
    target = HDF5_FILE / "JY68-4HF.h5"
    animate_sequential_growth_fixed(target)