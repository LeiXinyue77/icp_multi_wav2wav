"""
@Project ：
@File    ：plot.py
@Author  ：Lei Xinyue
@Date    ：2024/12/20 16:51
@Description: 绘图等相关方法
"""

from matplotlib import pyplot as plt


def plot_signals(sigs, chl, title=" "):
    """
    绘制多通道信号的图形
    :param signals: numpy.ndarray, 信号数组，形状为 (样本数, 通道数)。
    :param channel: list, 每个通道对应的名称列表。
    :param title: str, 图形的标题。
    """

    if sigs.shape[1] != len(chl):
        raise ValueError("signals 通道数和 channel 名称数量不匹配！")

    # 创建一个 Nx1 的子图布局
    fig, axes = plt.subplots(sigs.shape[1], 1, figsize=(8, 12))
    fig.suptitle(title, fontsize=16)

    # 遍历每个通道并绘制到对应的子图
    for n in range(sigs.shape[1]):  # 遍历通道
        ax = axes[n]  # 获取第 n 个子图
        ax.plot(sigs[:, n], label=f"{chl[n]}")
        ax.set_title(f"{chl[n]}")
        ax.set_xlabel("Sample")
        ax.set_ylabel("Amplitude")
        ax.legend()
        ax.grid(True)

    # 调整子图布局以避免重叠
    plt.tight_layout()
    # 显示图像
    plt.show()
