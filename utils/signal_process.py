"""
@Project ：
@File    ：signal_process.py
@Author  ：Lei Xinyue
@Date    ：2024/12/19 21:45
@Description: 信号处理相关方法
"""
import numpy as np
import pywt
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


def splitSig(signal, sample_freq=125, segLenth=60*5, overlap=0.3):
    """
    将信号划分为带重叠的片段
     :param signal: ndarray, 输入的信号数组
     :param sample_freq: int, 采样率（Hz）
     :param  segment_length: int, 每个片段的长度（秒）
     :param  overlap : float, 重叠百分比（0 到 1）
     :return splitSeg: list of ndarray, 符合条件的信号片段
    """
    samples = sample_freq * segLenth
    overlap = int(samples * overlap)
    splitSeg = []
    for s in range(0, len(signal), samples - overlap):
        if s + samples > len(signal):
            break
        splitSeg.append(signal[s: s + samples-1])

    # 过滤掉小于5min的片段
    splitSeg = [seg for seg in splitSeg if len(seg) < samples]
    return splitSeg


def period_dwt(signal, fs, wavelet='morl', scales_range=(1, 128)):
    """
    使用小波变换来计算信号的主导周期

    :param signal: 输入的时间序列数据
    :param fs: 采样频率（Hz）
    :param wavelet: 小波基类型（默认为 'morl'）
    :param scales_range: 小波尺度范围（默认为从 1 到 128）
    :return dominant_period: 主导周期（秒）
    """

    # 生成小波尺度范围
    scales = np.arange(scales_range[0], scales_range[1])

    # 进行小波变换
    coefficients, frequencies = pywt.cwt(signal, scales, wavelet, sampling_period=1/fs)

    # 找到主频率对应的周期
    dominant_scale = np.argmax(np.abs(coefficients), axis=0).mean()  # 平均尺度
    dominant_frequency = frequencies[int(dominant_scale)]  # 主频率
    dominant_period = 1 / dominant_frequency  # 主周期

    return dominant_period


def period_autocorrelation(sig, freq):
    """
    利用自相关函数计算ICP信号的主周期

    :param icp: 输入的ICP信号（numpy.ndarray）
    :param fs: 采样频率（Hz）
    :return period: 信号的主周期（秒）
    """

    # 1. 计算自相关
    sig = sig - np.mean(sig)  # 去均值
    autocorr = np.correlate(sig, sig, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]  # 保留非负延迟部分

    # 2. 寻找主周期
    # 找到自相关的第一个局部最大值（跳过0延迟点）
    peaks = np.where((autocorr[1:-1] > autocorr[:-2]) & (autocorr[1:-1] > autocorr[2:]))[0] + 1
    if len(peaks) == 0:
        # print("未找到周期性峰值")
        return None

    # 第一个峰值对应的延迟时间
    dominant_lag = peaks[0]
    period = dominant_lag / freq  # 将延迟转换为时间

    # 可视化自相关函数
    # plt.figure(figsize=(10, 5))
    # plt.plot(autocorr, label="Autocorrelation")
    # plt.axvline(dominant_lag, color='r', linestyle='--', label=f"Dominant Lag: {dominant_lag} samples")
    # plt.title("Autocorrelation Function")
    # plt.xlabel("Lag (samples)")
    # plt.ylabel("Autocorrelation")
    # plt.legend()
    # plt.grid()
    # plt.show()

    return period
