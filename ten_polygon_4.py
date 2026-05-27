import matplotlib.pyplot as plt
import numpy as np

from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# -----------------------------------
# 参数
# -----------------------------------

num_rects = 10

# 面片尺寸
width = 2.0
height = 1.0

# 两层之间的Y方向距离
y_offset = 2.0

# 波浪参数
wave_amplitude = 0.5   # 波浪振幅
wave_frequency = 0.6   # 波浪频率

# 总帧数
total_frames = num_rects * 2

# -----------------------------------
# 创建3D画布
# -----------------------------------

fig = plt.figure(figsize=(14, 6))
ax = fig.add_subplot(111, projection='3d')

# 坐标范围
ax.set_xlim(0, num_rects * width)
ax.set_ylim(0, y_offset + 1)
ax.set_zlim(-2, 3)

# 标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_title('3D Wavy Rectangle Growth')

# 保存所有面片
patches = []

# 不同层颜色
colors = ['skyblue', 'orange']

# -----------------------------------
# 动画更新函数
# -----------------------------------

def update(frame):

    # -----------------------------
    # 当前属于哪一层
    # -----------------------------

    if frame < num_rects:

        layer = 0
        idx = frame

    else:

        layer = 1
        idx = frame - num_rects

    # -----------------------------
    # 位置
    # -----------------------------

    x0 = idx * width
    y0 = layer * y_offset

    # -----------------------------
    # 波浪高度
    # -----------------------------

    z_left = wave_amplitude * np.sin(wave_frequency * x0)

    z_right = wave_amplitude * np.sin(
        wave_frequency * (x0 + width)
    )

    # -----------------------------
    # 定义波浪四边形
    # -----------------------------

    verts = [[
        [x0,         y0, z_left],
        [x0+width,   y0, z_right],
        [x0+width,   y0, z_right + height],
        [x0,         y0, z_left + height]
    ]]

    # -----------------------------
    # 创建三维面片
    # -----------------------------

    poly = Poly3DCollection(
        verts,
        alpha=0.7,
        facecolor=colors[layer],
        edgecolor='black',
        linewidth=1.5
    )

    ax.add_collection3d(poly)

    patches.append(poly)

    # -----------------------------
    # 添加编号
    # -----------------------------

    ax.text(
        x0 + width / 2,
        y0,
        (z_left + z_right) / 2 + height / 2,
        f'{idx}',
        color='black'
    )

    return patches

# -----------------------------------
# 创建动画
# -----------------------------------

ani = FuncAnimation(
    fig,
    update,
    frames=total_frames,
    interval=400,
    repeat=True,
    repeat_delay=1000
)

plt.show()