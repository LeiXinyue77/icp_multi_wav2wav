"""
@Project ：
@File    ：preprocess/phase.py
@Author  ：Lei Xinyue
@Date    ：2024/12/26 15:18
@Description: 滤波、相位对齐后生成 dl 数据集
"""
import os
import pickle
import numpy as np
import pandas as pd
from sympy.codegen import Print
from wfdb.processing import find_local_peaks

import utils.ecg_display as dp
from ecg.delay_coordinate_mapping import delay_cor, left_padding
from utils.plot import plot_signal_landmark, plot_signals
from utils.signal_process import lowpass_filter, bandpass_filter, period_autocorrelation


def apply_filters(sig, fs):
    """
    Apply different filters to each column of the input data.

    Parameters:
    dat (numpy array): The input data (shape: n_samples x n_features).
    fs (float): The sampling frequency.

    Returns:
    numpy array: The filtered signals (same shape as input).
    """
    filicp = lowpass_filter(sig[:, 0], fs, N=65, high=5)
    filabp = lowpass_filter(sig[:, 1], fs, N=9, high=5)
    filppg = lowpass_filter(sig[:, 2], fs, N=9, high=5)
    filecg = bandpass_filter(sig[:, 3], fs, low=1, high=20)

    filsig = dat.copy()
    filsig[:, 0] = filicp
    filsig[:, 1] = filabp
    filsig[:, 2] = filppg
    filsig[:, 3] = filecg

    return filsig


def find_delays(ecg_peaks, signal_peaks):
    """
    找到信号峰值中第一个落后于ECG第一个峰值的点，并计算偏移量。

    参数:
    ecg_peaks (list): ECG信号的峰值索引。
    signal_peaks (list): 其他信号的峰值索引（如ICP, ABP, PPG）。

    返回:
    int: 信号中第一个落后于ECG第一个峰值的峰值索引。
    int: 偏移量（单位：索引）。
    """
    offsets = []
    used_peaks = set()  # 用于记录已经匹配过的信号峰值
    for ecg_peak in ecg_peaks:
        # 找到第一个大于或等于当前ECG峰值且未使用的信号峰值
        lagging_peak = next(
            (sp for sp in signal_peaks if sp >= ecg_peak and sp not in used_peaks),
            None
        )

        if lagging_peak is None:
            offsets.append(0)  # 如果没有找到，偏移量为0
        else:
            offset = lagging_peak - ecg_peak
            offsets.append(offset)
            used_peaks.add(lagging_peak)  # 将匹配的峰值加入已使用集合
    return  offsets

def cal_mean_offset(offsets):
    """
    计算偏移量的均值，排除值为0的项。

    参数:
    offsets (list): 偏移量列表。

    返回:
    float: 偏移量均值（排除0）。
    """
    non_zero_offsets = [offset for offset in offsets if offset > 0]
    if len(non_zero_offsets) == 0:
        return 0  # 如果没有非零值，返回0
    return sum(non_zero_offsets) / len(non_zero_offsets)


if __name__ == '__main__':
    # 读取 preprocess/result/ValidHypertensionTime.csv 文件
    patients = pd.read_csv("result/ValidHypertensionTime.csv")["Patient"].tolist()
    # 读取 preprocess/result/validPeriod.csv 文件
    files = pd.read_csv("result/validPeriod.csv")
    files = files[files["sub_dirname"].isin(patients)]
    # 遍历 data
    for index, row in files.iterrows():
        try:
            row_path = os.path.join("../data", row["file_path"])
            with open(row_path, "rb") as f:
                data = pickle.load(f)
            start = row['start']
            end = row['end']
            dat = data.get("sig", None)[start:end]
            fields = data.get("fields", None)
            fs = fields.get('fs', None)

            # 滤波
            filteredSig = apply_filters(dat, fs)

            # 相位对齐, 并选择 ECG 信号作为基准信号
            window_size = int(10 * fs)  # 10 seconds window
            overlap_size = int(0.3 * window_size)  # 30% overlap
            step_size = window_size - overlap_size  # Step between windows

            index = 0  # 记录片段编号
            for i in range(0, len(dat) - window_size + 1, step_size):

                # 当前片段长度小于10s则舍弃
                if len(dat[i:i + window_size]) < window_size:
                    continue

                index += 1

                ecg = filteredSig[i:i + window_size, 3]
                ppg = filteredSig[i:i + window_size, 2]
                abp = filteredSig[i:i + window_size, 1]
                icp = filteredSig[i:i + window_size, 0]

                abp_period = period_autocorrelation(abp, fs)
                radius = int(0.6 * abp_period * fs)
                ecg_peaks = find_local_peaks(ecg, radius)
                # plot_signal_landmark(fs, ecg, ecg_peaks, [])

                delay = []
                for signal in [ppg, abp, icp]:
                    signal_peaks = find_local_peaks(signal, radius)
                    offsets = find_delays(ecg_peaks, signal_peaks)
                    # print(f"Offsets: {offsets}")
                    delay.append(cal_mean_offset(offsets))

                # print(f"Delay: {delay}")

                # 根据 delay 值，对齐对应信号的相位
                shift_ppg = np.roll(ppg, int(delay[0]))[0:1024,]
                shift_abp = np.roll(abp, int(delay[1]))[0:1024,]
                shift_icp = np.roll(icp, int(delay[2]))[0:1024,]
                shift_ecg = ecg[0:1024,]

                sig = [shift_icp, shift_abp, shift_ppg, shift_ecg]
                sig = np.array(sig).T
                # plot_signals(sig, ["ICP", "ABP", "PPG", "ECG"], title="Aligned Signals")
                # print("sig shape:", np.array(sig).shape)

                # 保存数据
                save_path = f"../dl/data/{row['sub_dirname'][:3]}/{row['sub_dirname']}"
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                np.save(f"{save_path}/{row['file'][:-4]}_{index}_{start+i}_{int(delay[2])}.npy", sig)
                print(f"Saved {row['file'][:-4]}_{index}_{start+i}_{int(delay[2])}.")

        except Exception as e:
            print(f"Error loading file {row['file_path']}: {e}")


    print("======================================= phase shift and npy save finished !!! "
          "===========================================")


