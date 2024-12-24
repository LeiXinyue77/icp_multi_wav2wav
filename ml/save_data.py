"""
@Project ：
@File    ：svm_miniset/save_data.py
@Author  ：Lei Xinyue
@Date    ：2024/12/20 13:36
@Description: 训练前信号预处理
"""

import os
import numpy as np
from matplotlib import pyplot as plt


def save_signals(sigs, chl, title=" ", save_path=None):
    """
    绘制多通道信号的图形并保存图像
    :param signals: numpy.ndarray, 信号数组，形状为 (样本数, 通道数)。
    :param channel: list, 每个通道对应的名称列表。
    :param title: str, 图形的标题。
    :param save_path: str, 保存图像的路径。
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

    # 保存图像到指定路径
    if save_path:
        fig_name = f"{title}.png"  # 图像文件名
        path_fig = os.path.join(save_path, "png")
        if not os.path.exists(path_fig):
            os.makedirs(path_fig)  # 创建文件夹（如果不存在）
        plt.savefig(os.path.join(path_fig, fig_name))  # 保存文件
        print(f"fig saved: {path_fig}/{fig_name}")

        # 保存信号数据
        data_name = f"{title}.npy"
        path_data = os.path.join(save_path, "npy")
        data_file = os.path.join(path_data, f"{title}.npy")  # 保存为 .npy 格式
        if not os.path.exists(path_data):
            os.makedirs(path_data)
        np.save(data_file, sigs)
        print(f"data saved: {path_data}/{data_name}")

    plt.close()  # 关闭图像，以避免显示时占用内存