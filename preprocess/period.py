"""
@Project ：
@File    ：preprocess/period.py
@Author  ：Lei Xinyue
@Date    ：2024/12/24 15:31
@Description: 训练前信号预处理
"""
import csv

from sympy.logic.inference import valid

from ecg.delay_coordinate_mapping import v_cal, delay_cor, left_padding
from ml.generate_data import calculate_avg_hr, check_diff
from ml.save_data import save_signals
from utils.plot import plot_signals
from utils.signal_process import lowpass_filter, bandpass_filter, period_autocorrelation
import utils.ecg_display as dp
import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np


def period_valid_hr():
    """
        统计No NaN in mean HR 的病人总数
        逐个片段计算，符合要求的片段累计5min，则保留
    """
    # 读取 validABP.csv 文件
    data = pd.read_csv('../result/pre/validABP.csv')
    valid_hr = []
    # 根据 validABP 中的文件路径读取数据
    for index, row in data.iterrows():
        try:
            row_path = os.path.join("../data", row["file_path"])
            with open(row_path, "rb") as f:
                data = pickle.load(f)
            row_start = row['start']
            row_end = row['end']
            signals = data.get("sig", None)[row_start:row_end]
            fields = data.get("fields", None)
            fs = fields.get('fs', None)

            # 1) 滤波
            filecg = bandpass_filter(signals[:, 3], fs, low=1, high=20)
            filsig = signals.copy()
            filsig[:, 3] = filecg

            # 2）10s一段检测波形是否符合要求
            seg_len = 10 * fs  # 10s
            thre_len = 5 * 60 * fs  # 5min
            valid_len = 0
            start = None
            end = None

            for i in range(0, len(filecg), seg_len):
                # normal = False
                # order += 1eg_len]
                ecg = filecg[i:i + seg_len]
                # plot_signals(sigs=filsig[i:i + seg_len], chl=channel, title=f"filtered-{i}-{i+seg_len}")
                # plot_signals(sigs=signals[i:i + seg_len], chl=channel, title=f"raw-{i}-{i+seg_len}")

                # 1) ecg_period
                v = v_cal(ecg, 8, fs)  # 单独计算V(n)

                v, llv_sum, rlv_sum, qrs, qrs_i, thrs1_arr, thrs2_arr = delay_cor(ecg, 8, fs)
                # dp.plot_peak_dot_llv_rlv(v, qrs_i, qrs, left_padding(llv_sum), left_padding(rlv_sum), thrs1_arr,
                #                          thrs2_arr)
                # dp.subplot_peaks(ecg, v, qrs_i, qrs, 'ECG', 'R peaks')

                avg_hr = calculate_avg_hr(qrs_i, fs)
                ecg_period = 60 / avg_hr if avg_hr is not None else None

                # print(f"icp_period: {icp_period}, abp_period: {abp_period}, ppg_period: {ppg_period}, "
                #       f"ecg_period: {ecg_period}")

                lowThreshold = 0.375
                highThreshold = 1.5
                if ecg_period is not None and lowThreshold < ecg_period < highThreshold:
                    # 累计有效片段时长, 10s一段, 累计5min则为有效记录，注意累计过程中不能中断，否则重新计数
                    if valid_len == 0:
                        start = i + row_start
                    valid_len += seg_len
                    end = i + row_start + seg_len
                else:
                    if valid_len >= thre_len:
                        # 保存数据
                        print(f"valid segment: {row['file_path']}, start: {start}, end: {end}")
                        valid_hr.append(
                            [row['file_path'], row['sub_dirname'], row['file'], start, end])
                    valid_len = 0
                    start = None
                    end = None

            if valid_len >= thre_len:
                # 保存数据
                print(f"valid segment: {row['file_path']}, start: {start}, end: {end}")
                valid_hr.append(
                    [row['file_path'], row['sub_dirname'], row['file'], start, end])

        except Exception as e:
            print(f"Error loading file {row['file_path']}: {e}")

    # 保存数据
    # 使用集合 set 去重 res_noNaN 中的 sub_dirname
    validHR_patient_set = set(
        sub_dirname for _, sub_dirname, _, _, _ in valid_hr)
    print(
        f"No NaN in mean HR 的病人总数： {len(validHR_patient_set)}")

    # 写入CSV文件
    save_path = "../result/pre"
    save_file = "validHR.csv"
    csv_validABP = os.path.join(save_path, save_file)
    with open(csv_validABP, mode="a", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        if csv_file.tell() == 0:
            csv_writer.writerow(
                ["file_path", "sub_dirname", "file", "start", "end"])  # 写入表头
        csv_writer.writerows(valid_hr)

    print("=================================  save validHR finished: validHR.csv===============================")


def period_valid_icp_abp_ppg():
    # 读取 validABP.csv 文件
    data = pd.read_csv('../result/pre/validABP.csv')
    valid_hr = []
    # 根据 validABP 中的文件路径读取数据
    for index, row in data.iterrows():
        try:
            row_path = os.path.join("../data", row["file_path"])
            with open(row_path, "rb") as f:
                data = pickle.load(f)
            row_start = row['start']
            row_end = row['end']
            signals = data.get("sig", None)[row_start:row_end]
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
            seg_len = 10 * fs  # 10s
            thre_len = 5 * 60 * fs  # 5min
            valid_len = 0
            start = None
            end = None

            for i in range(0, len(filicp), seg_len):
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

                # print(f"icp_period: {icp_period}, abp_period: {abp_period}, ppg_period: {ppg_period}, "
                #       f"ecg_period: {ecg_period}")

                lowThreshold = 0.375
                highThreshold = 1.5
                if ecg_period is not None and lowThreshold < ecg_period < highThreshold:
                    # 累计有效片段时长, 10s一段, 累计5min则为有效记录，注意累计过程中不能中断，否则重新计数
                    if valid_len == 0:
                        start = i + row_start
                    valid_len += seg_len
                    end = i + row_start + seg_len
                else:
                    if valid_len >= thre_len:
                        # 保存数据
                        print(f"valid segment: {row['file_path']}, start: {start}, end: {end}")
                        valid_hr.append(
                            [row['file_path'], row['sub_dirname'], row['file'], start, end])
                    valid_len = 0
                    start = None
                    end = None

            if valid_len >= thre_len:
                # 保存数据
                print(f"valid segment: {row['file_path']}, start: {start}, end: {end}")
                valid_hr.append(
                            [row['file_path'], row['sub_dirname'], row['file'], start, end])
            else:
                print(f"invalid segment: {row['file_path']}, start: {start}, end: {end}")

                '''
                # 1) any period 返回值为none
                if (icp_period is None) or (abp_period is None) or (ppg_period is None) or (avg_hr is None):
                    print(
                        f"未找到周期性峰值 icp_period: {icp_period}, abp_period: {abp_period}, ppg_period: {ppg_period}, "
                        f"ecg_period: {avg_hr}")
                    continue

                # 3) 任意两两周期差异大于0.01
                diff_flag = check_diff([icp_period, abp_period, ppg_period, ecg_period], 0.01)
                if diff_flag:
                    # 保存0类
                    # 需要同时保存图片和数据，命名(XXXXX_NNNN_start_end) 保持一致（图片用于人工分类，数据用于训练
                    continue

                # 机器学习分类 
                '''

        except Exception as e:
            print(f"Error loading file {row['file_path']}: {e}")

    # 保存数据
    # 使用集合 set 去重 res_noNaN 中的 sub_dirname
    validHR_patient_set = set(
        sub_dirname for _, sub_dirname, _, _, _ in valid_hr)
    print(
        f"No NaN in mean HR 的病人总数： {len(validHR_patient_set)}")

    # 写入CSV文件
    save_path = "../result/pre"
    save_file = "validHR.csv"
    csv_validABP = os.path.join(save_path, save_file)
    with open(csv_validABP, mode="a", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        if csv_file.tell() == 0:
            csv_writer.writerow(
                ["file_path", "sub_dirname", "file", "start", "end"])  # 写入表头
        csv_writer.writerows(valid_hr)

        print("=================================  save validHR finished: validHR.csv===============================")




# 1.5 对信号进行分段、滤波、计算周期等操作
if __name__=="__main__":
    period_valid_hr()
    print("=================================  save validHR finished: validHR.csv===============================")
