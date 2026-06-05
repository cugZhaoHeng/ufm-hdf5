import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# -----------------------------------
# 参数
# -----------------------------------

num_rects = 10

# 细长矩形尺寸（1:2）
width = 1.0
height = 2.0

# 第二层距离第一层的高度
y_offset = 1.5

# 总帧数
total_frames = num_rects * 2

# -----------------------------------
# 创建画布
# -----------------------------------

fig, ax = plt.subplots(figsize=(14, 5))

ax.set_xlim(-0.5, num_rects * width + 0.5)
ax.set_ylim(-0.5, y_offset + height + 0.5)

ax.set_aspect('equal')

ax.set_xlabel('X')
ax.set_ylabel('Y')

ax.set_title('Two-Layer Sequential Rectangles')

# 去掉网格
ax.grid(False)

# 保存矩形对象
patches = []

# -----------------------------------
# 动画更新函数
# -----------------------------------

def update(frame):

    # -----------------------------
    # 第一层
    # -----------------------------
    if frame < num_rects:

        layer = 0
        idx = frame

    # -----------------------------
    # 第二层
    # -----------------------------
    else:

        layer = 1
        idx = frame - num_rects

    # 当前矩形左下角
    x0 = idx * width

    # 当前层的y位置
    y0 = layer * y_offset

    # 四个顶点
    rect = np.array([
        [x0, y0],
        [x0 + width, y0],
        [x0 + width, y0 + height],
        [x0, y0 + height]
    ])

    x = rect[:, 0]
    y = rect[:, 1]

    # 绘制矩形
    patch = ax.fill(
        x,
        y,
        alpha=0.6,
        edgecolor='black',
        linewidth=2
    )[0]

    patches.append(patch)

    # 添加编号
    ax.text(
        x0 + width / 2,
        y0 + height / 2,
        f'{idx}',
        ha='center',
        va='center',
        fontsize=12
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
    repeat=False
)

plt.show()