from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation
import matplotlib as mpl

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
HDF5_FILE = PROJECT_DIR / "data" / "hdf5_file"

def animate_temperature_fractures(file_path):

    sim_ids = range(45, 55)

    all_sim_data = []

    print("读取 H5 数据中...")

    with h5py.File(file_path, 'r') as f:

        global_min = np.array([np.inf, np.inf, np.inf])
        global_max = np.array([-np.inf, -np.inf, -np.inf])

        temp_min = np.inf
        temp_max = -np.inf

        # -----------------------------------
        # 读取所有数据
        # -----------------------------------

        for sim_id in sim_ids:

            print(f"sim_id: {sim_id}")

            base_path = f"Fracture Simulation {sim_id}/Results/Bulk/UFM/FractureSet/Elements"

            points_path = f"{base_path}/Points"

            if f"{points_path}/X" not in f:
                continue

            # 几何数据
            X = f[f"{points_path}/X"][:]
            Y = f[f"{points_path}/Y"][:]
            Z = f[f"{points_path}/Z"][:]

            # 温度数据
            # T = f[f"{base_path}/Temperature"][:]
            T = f[f"{base_path}/NormalStress"][:]

            all_sim_data.append({
                'id': sim_id,
                'X': X,
                'Y': Y,
                'Z': Z,
                'T': T
            })

            # -----------------------------------
            # 更新全局坐标范围
            # -----------------------------------

            mask = ~np.isnan(X)

            if np.any(mask):

                global_min = np.minimum(
                    global_min,
                    [np.nanmin(X), np.nanmin(Y), np.nanmin(Z)]
                )

                global_max = np.maximum(
                    global_max,
                    [np.nanmax(X), np.nanmax(Y), np.nanmax(Z)]
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

    cmap = mpl.colormaps['jet']

    norm = mpl.colors.Normalize(
        vmin=temp_min,
        vmax=temp_max
    )

    # -----------------------------------
    # 创建画布
    # -----------------------------------

    fig = plt.figure(figsize=(6,5))

    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim(global_min[0], global_max[0])
    ax.set_ylim(global_min[1], global_max[1])
    ax.set_zlim(global_min[2], global_max[2])

    ax.invert_zaxis()

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Depth (m)')

    # -----------------------------------
    # 添加色阶
    # -----------------------------------

    sm = mpl.cm.ScalarMappable(
        cmap=cmap,
        norm=norm
    )

    sm.set_array([])

    cbar = plt.colorbar(
        sm,
        ax=ax,
        shrink=0.6,
        pad=0.1
    )

    cbar.set_label('Temperature')

    # -----------------------------------
    # 创建 Poly3DCollection
    # -----------------------------------

    poly_collections = []

    for _ in all_sim_data:

        p = Poly3DCollection(
            [],
            alpha=0.7,
            edgecolors='none'
        )

        ax.add_collection3d(p)

        poly_collections.append(p)

    title = ax.set_title('Fracture Temperature Evolution')

    # -----------------------------------
    # 构建帧索引
    # -----------------------------------

    frames_map = []

    # 这里的 s_idx 是从 0 开始的索引，不是 sim_id
    for s_idx, sim in enumerate(all_sim_data):

        num_steps = sim['X'].shape[0]

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

        x_t = sim['X'][curr_time_step]
        y_t = sim['Y'][curr_time_step]
        z_t = sim['Z'][curr_time_step]

        temp_t = sim['T'][curr_time_step]

        # -----------------------------------
        # 有效面片
        # -----------------------------------

        valid = np.where(
            ~np.isnan(x_t).any(axis=1)
        )[0]

        verts = []

        face_colors = []

        # -----------------------------------
        # 构建所有面片
        # -----------------------------------

        for v in valid:

            quad = np.stack([
                x_t[v],
                y_t[v],
                z_t[v]
            ], axis=-1)

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

                last_x = all_sim_data[i]['X'][-1]
                last_y = all_sim_data[i]['Y'][-1]
                last_z = all_sim_data[i]['Z'][-1]

                last_t = all_sim_data[i]['T'][-1]

                v_idx = np.where(
                    ~np.isnan(last_x).any(axis=1)
                )[0]

                final_verts = []

                final_colors = []

                for v in v_idx:

                    quad = np.stack([
                        last_x[v],
                        last_y[v],
                        last_z[v]
                    ], axis=-1)

                    final_verts.append(quad)

                    color = cmap(norm(last_t[v]))

                    final_colors.append(color)

                poly_collections[i].set_verts(final_verts)

                poly_collections[i].set_facecolor(final_colors)

                finalized_sims.add(i)

        title.set_text(
            f'Stage: {sim["id"]} | Time Step: {curr_time_step}'
        )

        return poly_collections + [title]

    # -----------------------------------
    # 创建动画
    # -----------------------------------

    ani = FuncAnimation(
        fig,
        update,
        frames=len(frames_map),
        interval=10,
        blit=False,
        repeat=False
    )

    plt.show()


if __name__ == "__main__":

    # target = r"JY68-4HF.h5"
    target = HDF5_FILE /  "JY68-4HF.h5"

    animate_temperature_fractures(target)