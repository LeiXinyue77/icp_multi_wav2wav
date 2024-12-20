"""
@Project ：
@File    ：signal_process.py
@Author  ：Lei Xinyue
@Date    ：2024/12/19 21:45
@Description: 信号处理相关方法
"""
from scipy import signal


def bandpass_filter(data, fs, low=5, high=15):
    """
    带通滤波
    :param data: list, 输入信号数据
    :param fs: int, 采样率
    :param low: int, 截止频率1
    :param high: int, 截止频率2
    :return: list, 滤波后信号
    """
    low = 2 * low / fs
    high = 2 * high / fs
    b, a = signal.butter(3, [low, high], 'bandpass')
    filter = signal.filtfilt(b, a, data)  # data为要过滤的信号
    return filter


# 低通滤波器
def lowpass_filter(data, fs, N, high=5):
    """
    带通滤波
    :param data: list, 输入信号数据
    :param fs: int, 采样率
    :param N: int, 滤波器阶数
    :param high: int, 截止频率
    :return: list, 滤波后信号
    """
    high = 2 * high / fs
    b = signal.firwin(N, high, window="hamming", pass_zero="lowpass")
    filter = signal.filtfilt(b, 1, data)  # data为要过滤的信号
    return filter