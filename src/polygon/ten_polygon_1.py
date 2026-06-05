import matplotlib.pyplot as plt
import numpy as np

# 创建画布
plt.figure(figsize=(12, 4))

# 四边形数量
num_quads = 10

# 每个四边形宽度和高度
width = 1.0
height = 1.0

# 用于保存所有四边形
all_quads = []

# -----------------------------------
# 生成连续共享边的四边形
# -----------------------------------

for i in range(num_quads):

    # 左下角 x 坐标
    x0 = i * width

    # 四个顶点（顺时针）
    quad = np.array([
        [x0, 0],                # 左下
        [x0 + width, 0],        # 右下
        [x0 + width, height],   # 右上
        [x0, height]            # 左上
    ])

    all_quads.append(quad)
print(all_quads)

# -----------------------------------
# 绘制
# -----------------------------------

for i, quad in enumerate(all_quads):

    x = quad[:, 0]
    y = quad[:, 1]

    plt.fill(
        x,
        y,
        alpha=0.5,
        edgecolor='black',
        linewidth=2
    )

    # 标注编号
    center_x = np.mean(x)
    center_y = np.mean(y)

    plt.text(
        center_x,
        center_y,
        f'{i}',
        ha='center',
        va='center',
        fontsize=12
    )

# -----------------------------------
# 坐标设置
# -----------------------------------

plt.axis('equal')

plt.xlim(-0.5, num_quads + 0.5)
plt.ylim(-0.5, 1.5)

plt.xlabel('X')
plt.ylabel('Y')

plt.title('Connected Quadrilaterals')

plt.grid(True)

plt.show()