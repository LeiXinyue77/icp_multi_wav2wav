"""
@Project ：
@File    ：plot.py
@Author  ：Lei Xinyue
@Date    ：2024/12/20 16:51
@Description: 绘图等相关方法
"""
import numpy as np
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

        # 如果只有一个通道，axes 不是列表，需要特殊处理
    single_channel = sigs.shape[1] == 1

    # 创建一个 Nx1 的子图布局
    fig, axes = plt.subplots(sigs.shape[1], 1, figsize=(8, 6 if single_channel else 12))
    fig.suptitle(title, fontsize=16)

    # 将 axes 转换为列表形式以统一处理
    if single_channel:
        axes = [axes]

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
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    # 显示图像
    plt.show()


def plot_signal_landmark(fs, signal, peaks, troughs):
    """
    绘制PPG信号图，标注峰值和谷值

    参数:
    fs (int): 采样频率
    ppg_signal (array): PPG信号数据
    peaks (array): 峰值的索引
    troughs (array): 谷值的索引
    """
    time = np.linspace(0, len(signal)/fs, len(signal))
    plt.figure(figsize=(12, 6))
    plt.plot(time, signal, label="PPG Signal", linewidth=1)
    plt.plot(time[peaks], signal[peaks], 'ro', label="Peaks")
    plt.plot(time[troughs], signal[troughs], 'go', label="Troughs")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Signal with Detected Peaks and Troughs")
    plt.legend()
    plt.grid(True)
    plt.show()

# 调用示例
# plot_ppg_signal(time, ppg_signal, peaks, troughs)
