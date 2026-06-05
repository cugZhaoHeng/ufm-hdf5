import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# -----------------------------------
# 参数
# -----------------------------------

num_rects = 10

# 细长矩形尺寸
width = 1.0
height = 2.0

# -----------------------------------
# 创建画布
# -----------------------------------

fig, ax = plt.subplots(figsize=(14, 3))

ax.set_xlim(-0.5, num_rects * width + 0.5)
ax.set_ylim(-0.5, height + 0.5)

ax.set_aspect('equal')

ax.set_xlabel('X')
ax.set_ylabel('Y')

ax.set_title('Sequential Thin Rectangles')

# 去掉网格线
ax.grid(False)

# 保存已经生成的矩形
patches = []

# -----------------------------------
# 动画更新函数
# -----------------------------------

def update(frame):

    # 当前矩形左下角
    x0 = frame * width

    # 四个顶点
    rect = np.array([
        [x0, 0],
        [x0 + width, 0],
        [x0 + width, height],
        [x0, height]
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
        height / 2,
        f'{frame}',
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
    frames=num_rects,
    interval=500,
    repeat=False
)

plt.show()