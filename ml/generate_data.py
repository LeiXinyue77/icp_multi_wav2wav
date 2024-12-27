"""
@Project ：
@File    ：ml/generate_data.py
@Author  ：Lei Xinyue
@Date    ：2024/12/24 15:28
@Description: 训练前信号预处理
"""
from ecg.delay_coordinate_mapping import v_cal, delay_cor, left_padding
from ml.save_data import save_signals
from utils.plot import plot_signals
from utils.signal_process import lowpass_filter, bandpass_filter, period_autocorrelation
import utils.ecg_display as dp
import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np


# 计算心率 (HR)
def calculate_avg_hr(r_peaks, fs):
    """
    计算心率（HR）
    :param r_peaks: R波位置数组（单位：采样点）
    :param fs: 采样频率（Hz）
    :return: avg_hr: 平均心率（bpm）
    """
    rr_intervals = np.diff(r_peaks) / fs  # 计算R-R间隔（单位：秒）
    hr = 60 / rr_intervals  # 转换为bpm

    if hr.size == 0:
        return None
    if np.any(np.isnan(hr)):
        return None

    avg_hr = np.mean(hr) # 计算平均心率
    return avg_hr


def check_diff(periods, threshold):
    """
    检查任意两个周期之间的差值是否不大于某个值。

    :param periods: 周期值列表 [icp_period, abp_period, ppg_period, ecg_period]
    :param threshold: 差值阈值
    :return: 布尔值，True 表示所有差值均不大于阈值，False 表示有超出阈值的情况
    """
    n = len(periods)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(periods[i] - periods[j]) > threshold:
                # print(f"Difference too large: {periods[i]} and {periods[j]}")
                return False
    return True


def generate_ml_data():
    """
    生成机器学习数据集
    :return:
    """
    # 读取 validABP.csv 文件
    data = pd.read_csv('../preprocess/result/validABP.csv')
    num = 0  # 记录处理过的文件个数 ( 生成svm_dataSet, 每个pXX文件夹200个片段)
    # 根据 validABP 中的文件路径读取数据
    for index, row in data.iterrows():
        try:
            row_path = os.path.join("../data", row["file_path"])
            with open(row_path, "rb") as f:
                data = pickle.load(f)
            start = row['start']
            end = row['end']
            signals = data.get("sig", None)[start:end]
            fields = data.get("fields", None)
            fs = fields.get('fs', None)
            channel = fields.get('sig_name', [])

            # 1) 滤波
            filicp = lowpass_filter(signals[:, 0], fs, N=65, high=5)
            filabp = lowpass_filter(signals[:, 1], fs, N=9, high=5)
            filppg = lowpass_filter(signals[:, 2], fs, N=9, high=5)
            filecg = bandpass_filter(signals[:, 3], fs, low=1, high=20)
            filsig = signals.copy()
            filsig[:, 0] = filicp
            filsig[:, 1] = filabp
            filsig[:, 2] = filppg
            filsig[:, 3] = filecg
            file = row["file_path"]
            # plot_signals(sigs=filsig, chl=channel, title=f"filtered-{file}")
            # plot_signals(sigs=signals, chl=channel, title=f"raw-{file}")

            # 2）10s一段检测波形是否符合要求
            seg_len = 10 * fs  # 每段10秒

            for i in range(0, len(filicp), seg_len):
                # num >= 200 结束遍历
                if num >= 200:
                    break
                # normal = False
                # order += 1
                icp = filicp[i:i + seg_len]
                abp = filabp[i:i + seg_len]
                ppg = filppg[i:i + seg_len]
                ecg = filecg[i:i + seg_len]
                # plot_signals(sigs=filsig[i:i + seg_len], chl=channel, title=f"filtered-{i}-{i+seg_len}")
                # plot_signals(sigs=signals[i:i + seg_len], chl=channel, title=f"raw-{i}-{i+seg_len}")

                # 利用自相关函数求icp abp ppg主周期
                icp_period = period_autocorrelation(icp, fs)
                abp_period = period_autocorrelation(abp, fs)
                ppg_period = period_autocorrelation(ppg, fs)

                # 1) ecg_period
                v = v_cal(ecg, 8, fs)  # 单独计算V(n)

                v, llv_sum, rlv_sum, qrs, qrs_i, thrs1_arr, thrs2_arr = delay_cor(ecg, 8, fs)
                # dp.plot_peak_dot_llv_rlv(v, qrs_i, qrs, left_padding(llv_sum), left_padding(rlv_sum), thrs1_arr,
                #                          thrs2_arr)
                # dp.subplot_peaks(ecg, v, qrs_i, qrs, 'ECG', 'R peaks')

                avg_hr = calculate_avg_hr(qrs_i, fs)
                ecg_period = 60 / avg_hr if avg_hr is not None else None

                print(f"icp_period: {icp_period}, abp_period: {abp_period}, ppg_period: {ppg_period}, "
                      f"ecg_period: {ecg_period}")

                # 1) any period 返回值为none
                if (icp_period is None) or (abp_period is None) or (ppg_period is None) or (avg_hr is None):
                    print(
                        f"未找到周期性峰值 icp_period: {icp_period}, abp_period: {abp_period}, ppg_period: {ppg_period}, "
                        f"ecg_period: {avg_hr}")
                    patient = row['sub_dirname']
                    file = row['file']
                    save_path = f"data/v2/0/{patient[:3]}/{patient}"
                    save_signals(
                        sigs=filsig[i:i + seg_len], chl=channel,
                        title=f"{num}-{file.split('.')[0]}-{i}-{i + seg_len}",
                        save_path=save_path)
                    num += 1
                    continue

                # 3) 任意两两周期差异大于0.01
                diff_flag = check_diff([icp_period, abp_period, ppg_period, ecg_period], 0.01)
                # diff_flag 取反
                if not diff_flag:
                    # 保存0类
                    # 需要同时保存图片和数据，命名(XXXXX_NNNN_start_end) 保持一致（图片用于人工分类，数据用于训练
                    patient = row['sub_dirname']
                    file = row['file']
                    save_path = f"data/v2/0/{patient[:3]}/{patient}"
                    save_signals(
                        sigs=filsig[i:i + seg_len], chl=channel,
                        title=f"{num}-{file.split('.')[0]}-{i}-{i + seg_len}",
                        save_path=save_path)
                    num += 1
                    continue

                # 保存1类
                # 需要同时保存图片和数据，命名保持一致（图片用于人工分类，数据用于训练
                patient = row['sub_dirname']
                file = row['file']
                save_path = f"data/v2/1/{patient[:3]}/{patient}"
                save_signals(
                    sigs=filsig[i:i + seg_len], chl=channel,
                    title=f"{num}-{file.split('.')[0]}-{i}-{i + seg_len}",
                    save_path=save_path)
                num += 1

                # 机器学习分类 ?

        except Exception as e:
            print(f"Error loading file {row['file_path']}: {e}")


if __name__=="__main__":
    # 生成机器学习数据集（正常信号 异常信号）
    generate_ml_data()
    print("================================== generate_ml_data finished !!! ========================================")
