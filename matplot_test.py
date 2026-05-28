import matplotlib.pyplot as plt
import numpy as np

def fill_one_polygon():
    # 绘制一个面
    x = [0, 1, 1, 0]
    y = [0, 0, 1, 1]
    plt.fill(x, y, color="blue", alpha=0.5)
    plt.xlim(-1, 2)
    plt.ylim(-1, 2)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Filled Area Example")
    plt.grid()
    plt.show()


def fill_multiple_polygons():
    # 绘制多个面
    x1 = [0, 1, 1, 0]
    y1 = [0, 0, 1, 1]
    x2 = [1, 2, 2, 1]
    y2 = [0, 0, 1, 1]
    plt.fill(x1, y1, color="blue", alpha=0.5)
    plt.fill(x2, y2, color="red", alpha=0.5)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Multiple Filled Areas Example")
    plt.grid()
    plt.show()


def fill_many_polygons():
    np.random.seed(0)

    plt.figure(figsize=(8, 8))

    for i in range(20):
        cx = np.random.uniform(0, 10)
        cy = np.random.uniform(0, 10)

        w = np.random.uniform(0.3, 1.0)
        h = np.random.uniform(0.3, 1.0)

        x = [cx - w, cx + w, cx + w * 0.8, cx - w * 0.8]

        y = [cy - h, cy - h, cy + h, cy + h]

        plt.fill(x, y, alpha=0.5)

    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    # fill_one_polygon()
    # fill_multiple_polygons()
    fill_many_polygons()
