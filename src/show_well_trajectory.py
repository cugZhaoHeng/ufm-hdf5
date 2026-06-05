# 读取井轨迹的数据文件，并通过matplotlib进行3D可视化，展示井轨迹的空间分布和形态特征。

from pathlib import Path
import sys
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
WELL_DATA_DIR = PROJECT_DIR / "data" / "well_data"
OUT_PUT_IDR = PROJECT_DIR / "output"

project_root_str = str(PROJECT_DIR)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)
from utils.logger import create_logger
logger = create_logger(__name__)
from utils.date_util import get_current_time

def read_well_trajectory(file_path):
    """
    读取井轨迹数据文件，提取X、Y、Z坐标。
    假设文件格式为CSV或TXT，且包含列名"X", "Y", "Z"。
    """
    
    
    try:
        df = pd.read_csv(file_path, sep=r'\s+', comment="#")  # 根据实际文件格式调整分隔符
        n = 400
        df_last = df.tail(n)
        well_x = df_last["X"].values
        well_y = df_last["Y"].values
        well_z = df_last["TVD"].values
        return well_x, well_y, well_z
    except Exception as e:
        logger.error(f"读取井轨迹数据失败: {e}")
        raise
    
def show_well_trajectory(well_x, well_y, well_z):
    """
    使用matplotlib的3D绘图功能展示井轨迹。
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(well_x, well_y, well_z, color='blue', linewidth=2.0)
    ax.set_xlabel('X (m)',labelpad=10)
    ax.set_ylabel('Y (m)',labelpad=10)
    ax.set_zlabel('Z (m)',labelpad=10)
    ax.set_title('Well Trajectory')
    
    ax.legend()

    # 保持 3D 图形比例协调
    try:
        ax.set_box_aspect((np.ptp(well_x)+1, np.ptp(well_y)+1, np.ptp(well_z)+1))
    except Exception:
        pass
    
    plt.show()

def show1(well_x, well_y, well_z):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    
    ax.plot(well_y,  well_z, color='blue', linewidth=2.0)
    ax.set_xlabel('X (m)',labelpad=10)
    ax.set_ylabel('Z (m)',labelpad=10)
    ax.set_title('Well Trajectory')
    
    # 保持 3D 图形比例协调
    try:
        ax.set_box_aspect((np.ptp(well_y)+1, np.ptp(well_z)+1))
    except Exception:
        pass
    
    plt.show()

def show2(well_x, well_y, well_z):

    # 4. 开始绘图
    fig = plt.figure(figsize=(5, 6))

    # ---- 子图 1: 3D 井眼轨迹 ----
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.plot(well_x, well_y, well_z, color='red', marker='o', markersize=3, linewidth=2, label="Wellpath")
    ax1.scatter(well_x[0], well_y[0], well_z[0], color='green', marker='^', s=100, label="Well Head") # 标记井口

    ax1.set_title("3D Well Trajectory", fontsize=12, fontweight='bold')
    ax1.set_xlabel("X (Easting, m)", labelpad=10)
    ax1.set_ylabel("Y (Northing, m)", labelpad=10)
    ax1.set_zlabel("Elevation / Z (m)", labelpad=10)
    ax1.invert_zaxis()
    ax1.legend()

    # 保持 3D 图形比例协调
    # try:
    #     ax1.set_box_aspect((np.ptp(well_x)+1, np.ptp(well_y)+1, np.ptp(well_z)+1))
    # except Exception:
    #     pass
    plt.tight_layout()
    plt.show()

def show_dynamic(well_x, well_y, well_z):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # 先确认x,y,z三个方向的最大和最小值，提前赋值给
    global_min = [min(well_x), min(well_y), min(well_z)]
    global_max = [max(well_x), max(well_y), max(well_z)]
    ax.set_xlim(global_min[0], global_max[0])
    ax.set_ylim(global_min[1], global_max[1])
    ax.set_zlim(global_min[2], global_max[2])
    ax.set_xlabel('X (m)',labelpad=10)
    ax.set_ylabel('Y (m)',labelpad=10)
    ax.set_zlabel('Z (m)',labelpad=10)
    ax.set_title('Dynamic Well Trajectory Growth')
        
    line, = ax.plot([], [], [], color='blue', linewidth=2.0, label="Wellpath")  # 添加图例标签
    ax.set_box_aspect((np.ptp(well_x)+1, np.ptp(well_y)+1, np.ptp(well_z)+1))
    for i in range(len(well_x)):
        line.set_data(well_x[:i+1], well_y[:i+1])
        line.set_3d_properties(well_z[:i+1])
        plt.pause(0.05)  # 调整动画速度
    
    plt.show()

def show_dynamic_animation(well_x, well_y, well_z):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # 先确认x,y,z三个方向的最大和最小值，提前赋值给
    global_min = [min(well_x), min(well_y), min(well_z)]
    global_max = [max(well_x), max(well_y), max(well_z)]
    ax.set_xlim(global_min[0], global_max[0])
    ax.set_ylim(global_min[1], global_max[1])
    ax.set_zlim(global_min[2], global_max[2])
    ax.set_xlabel('X (m)',labelpad=10)
    ax.set_ylabel('Y (m)',labelpad=10)
    ax.set_zlabel('Z (m)',labelpad=10)
    ax.set_title('Dynamic Well Trajectory Growth')
        
    # 使用 FuncAnimation 创建动画
    line, = ax.plot([], [], [], color='blue', linewidth=2.0, label="Wellpath")  # 添加图例标签
    ax.set_box_aspect((np.ptp(well_x)+1, np.ptp(well_y)+1, np.ptp(well_z)+1))
    def update(frame):
        line.set_data(well_x[:frame+1], well_y[:frame+1])
        line.set_3d_properties(well_z[:frame+1])
        return line,
    ani = FuncAnimation(fig, update, frames=len(well_x), interval=50, blit=True)
    # ani.save(OUT_PUT_IDR / f'well_trajectory_animation_{get_current_time()}.gif', writer='pillow', fps=20)  # 保存为GIF动画
    ani.save(OUT_PUT_IDR / f'well_trajectory_animation_{get_current_time()}.mp4', writer='ffmpeg', fps=20) # 保存为MP4视频
    logger.info(f"动画已保存到: {OUT_PUT_IDR / f'well_trajectory_animation_{get_current_time()}.mp4'}")
    plt.show()

if __name__ == "__main__":
    well_data = read_well_trajectory(WELL_DATA_DIR / "68-4HF")
    # well_data = read_well_trajectory(WELL_DATA_DIR / "68-4HF-02.txt")
    well_x, well_y, well_z = well_data
    # well_x = well_x - well_x[0]
    # well_y = well_y - well_y[0]
    # well_z = well_z - well_z[0]
    logger.info(f"成功读取井轨迹数据: {len(well_x)} points")
    logger.info(f"x: {well_x[:5]} ...")
    logger.info(f"y: {well_y[:5]} ...")
    logger.info(f"z: {well_z[:5]} ...")
    
    # show_well_trajectory(well_x, well_y, well_z)
    # show1(well_x, well_y, well_z)
    # show2(well_x, well_y, well_z)
    # show_dynamic(well_x, well_y, well_z)
    show_dynamic_animation(well_x, well_y, well_z)