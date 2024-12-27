"""
@Project ：
@File    ：preprocess/period.py
@Author  ：Lei Xinyue
@Date    ：2024/12/24 15:31
@Description: 训练前信号预处理
"""
import csv
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
from joblib import dump, load


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


def check_valid_period(periods, low, high):
    for period in periods:
        if period is None or period < low or period > high:
            return False
    return True


def period_valid_icp_abp_ppg():
    # 读取 validABP.csv 文件
    data = pd.read_csv('../result/pre/validHR.csv')
    xgb_model = load('../ml/xgb_model.joblib')
    valid_res = []
    # 根据 validHR 中的文件路径读取数据
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
            filicp = lowpass_filter(signals[:, 0], fs, N=65, high=5)
            filabp = lowpass_filter(signals[:, 1], fs, N=9, high=5)
            filppg = lowpass_filter(signals[:, 2], fs, N=9, high=5)
            filecg = bandpass_filter(signals[:, 3], fs, low=1, high=20)
            filsig = signals.copy()
            filsig[:, 0] = filicp
            filsig[:, 1] = filabp
            filsig[:, 2] = filppg
            filsig[:, 3] = filecg
            # file = row["file_path"]
            # plot_signals(sigs=filsig, chl=channel, title=f"filtered-{file}")
            # plot_signals(sigs=signals, chl=channel, title=f"raw-{file}")

            # 2）10s一段检测波形是否符合要求
            seg_len = 10 * fs  # 10s
            thre_len = 5 * 60 * fs  # 5min
            valid_len = 0
            start = None
            end = None

            for i in range(0, len(filicp), seg_len):

                # 判断本次循环的长度是否小于seg_len
                if i + seg_len > len(filicp):
                    break
                # normal = False
                # order += 1
                icp = filicp[i:i + seg_len]
                abp = filabp[i:i + seg_len]
                ppg = filppg[i:i + seg_len]
                ecg = filecg[i:i + seg_len]
                # plot_signals(sigs=filsig[i:i + seg_len], chl=channel, title=f"filtered-{i}-{i+seg_len}")
                # plot_signals(sigs=signals[i:i + seg_len], chl=channel, title=f"raw-{i}-{i+seg_len}")
                segment_sig = filsig[i:i + seg_len]

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

                # 利用 xgboost 判断是否为异常信号
                # filsig归一化
                min_vals = segment_sig.min(axis=0, keepdims=True)
                max_vals = segment_sig.max(axis=0, keepdims=True)
                range_vals = max_vals - min_vals
                range_vals[range_vals == 0] = 1
                standard_sig = (segment_sig - min_vals) / range_vals
                # flatten standard_sig
                standard_sig = standard_sig.reshape(1, -1)

                # 加载模型并进行预测
                y_pred = xgb_model.predict(standard_sig)

                lowThreshold = 0.375
                highThreshold = 1.5
                periods = [icp_period, abp_period, ppg_period, ecg_period]
                # 1) any period 返回值为none and any period < 0.375 and any period > 1.5 or and diff > 0.01
                if check_valid_period(periods, lowThreshold, highThreshold) and check_diff(periods, 0.01) and y_pred == 1:
                    # 累计有效片段时长, 10s一段, 累计5min则为有效记录，注意累计过程中不能中断，否则重新计数
                    if valid_len == 0:
                        start = i + row_start
                    valid_len += seg_len
                    end = i + row_start + seg_len
                else:
                    if valid_len >= thre_len:
                        # 保存数据
                        print(f"valid segment: {row['file_path']}, start: {start}, end: {end}")
                        valid_res.append(
                            [row['file_path'], row['sub_dirname'], row['file'], start, end])
                    valid_len = 0
                    start = None
                    end = None

            if valid_len >= thre_len:
                # 保存数据
                print(f"valid segment: {row['file_path']}, start: {start}, end: {end}")
                valid_res.append(
                            [row['file_path'], row['sub_dirname'], row['file'], start, end])

        except Exception as e:
            print(f"Error loading file {row['file_path']}: {e}")

    # 保存数据
    # 使用集合 set 去重 res_noNaN 中的 sub_dirname
    valid_patient_set = set(
        sub_dirname for _, sub_dirname, _, _, _ in valid_res)
    print(
        f"valid period 的病人总数： {len(valid_patient_set)}")

    # 写入CSV文件
    save_path = "../result/pre"
    save_file = "validPeriod.csv"
    csv_validABP = os.path.join(save_path, save_file)
    with open(csv_validABP, mode="a", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        if csv_file.tell() == 0:
            csv_writer.writerow(
                ["file_path", "sub_dirname", "file", "start", "end"])  # 写入表头
        csv_writer.writerows(valid_res)


def patient_time():
    """
    计算每个病人有效时间
    :return:
    """
    data = pd.read_csv('../result/pre/validPeriod.csv')
    # 初始化一个字典，用于存储每个病人的有效累计时间
    time_map = {}
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

            # 计算有效时间
            effective_time = len(signals)/fs/3600

            # 累加到对应的病人
            patient_id = row['sub_dirname']
            if patient_id in time_map:
                time_map[patient_id] += effective_time
            else:
                time_map[patient_id] = effective_time

        except Exception as e:
            print(f"Error loading file {row['file_path']}: {e}")

    # 将累计有效时间写入 CSV 文件
    output_path = '../result/pre/validTimeofPatient.csv'
    df = pd.DataFrame(list(time_map.items()), columns=['Patient', 'Effective_Time'])
    df.to_csv(output_path, index=False)
    print(f"cumulative effective time saved: {output_path}")

    # 绘制统计图
    # 绘制直方统计图
    plt.figure(figsize=(10, 6))
    times = df['Effective_Time']

    # 使用 plt.hist 绘制直方图
    bins = [0, 10, 20, 30, 40, 50, 60, 70]
    n, bins, patches = plt.hist(times, bins=bins, color='skyblue', edgecolor='black')

    # 添加区间人数标注
    for count, edge in zip(n, bins[:-1]):
        if count > 0:  # 只标注非零的柱子
            plt.text(edge + (bins[1] - bins[0]) / 2, count,  # 文本位置
                     str(int(count)),  # 标注内容
                     ha='center', va='bottom', fontsize=10)  # 样式设置

    plt.xlabel('Effective Time (h)', fontsize=12)
    plt.ylabel('Number of Patients', fontsize=12)
    plt.title('Histogram of Effective Time per Patient', fontsize=14)
    plt.tight_layout()

    # 保存统计图
    plot_path = '../result/pre/patient_time_histogram.png'
    plt.savefig(plot_path)
    plt.show()
    print(f"picture saved: {plot_path}")


# 1.5 对信号进行分段、滤波、计算周期等操作
if __name__=="__main__":
    # period_valid_hr()
    # print("=================================  save validHR finished: validHR.csv===============================")
    # period_valid_icp_abp_ppg()
    # print("=================================  save valid period finished: "
    #       "validPeriod.csv===============================")
    patient_time()
    print("======================================= patient time finished: ========================================")



