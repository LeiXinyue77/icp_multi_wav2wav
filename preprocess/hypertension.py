"""
@Project ：
@File    ：preprocess/hypertension.py
@Author  ：Lei Xinyue
@Date    ：2024/12/24 15:31
@Description: 特征提取
"""
import os
import pickle
import pandas as pd
from wfdb.processing import find_local_peaks
from utils.plot import plot_signal_landmark, plot_signals
from utils.signal_process import lowpass_filter, period_autocorrelation


def CalHypertensionTime():
    # 读取 result/pre/validPeriod.csv 文件
    data = pd.read_csv("result/validPeriod.csv")
    time_map = {}
    time_res = []
    for index, row in data.iterrows():
        try:
            total_time = 0
            row_path = os.path.join("../data", row["file_path"])
            with open(row_path, "rb") as f:
                data = pickle.load(f)
            start = row['start']
            end = row['end']
            signals = data.get("sig", None)[start:end]
            fields = data.get("fields", None)
            fs = fields.get('fs', None)
            # plot_signals(signals, ["icp", "abp", "ppg", "ecg"], title="raw Signals")

            # 1) 滤波
            filicp = lowpass_filter(signals[:, 0], fs, N=65, high=5)

            # 2）10s一段检测波形是否符合要求
            seg_len = 10 * fs  # 10s

            for i in range(0, len(filicp), seg_len):
                icp = filicp[i:i + seg_len]
                icp_period = period_autocorrelation(icp, fs)
                # 检测峰值和谷值
                radius = int( 0.6*icp_period * fs)
                peaks = find_local_peaks(icp, radius)
                troughs = find_local_peaks(-icp, radius)
                # plot_signal_landmark(fs, icp, peaks, troughs)
                # print("Peaks:", peaks, "\n")
                # print("Troughs:", troughs, "\n")

                # ave_icp =  (peak_icp - trough_icp )/3 + trough_icp
                # 确保峰值和谷值顺序正确（peak_idx < trough_idx）
                valid_peaks = []
                valid_troughs = []
                trough_ptr = 0  # 谷值的指针

                for peak_idx in peaks:
                    # 找到第一个谷值索引大于当前峰值索引的谷值
                    while trough_ptr < len(troughs) and troughs[trough_ptr] <= peak_idx:
                        trough_ptr += 1

                        # 如果找到的谷值满足条件
                    if trough_ptr < len(troughs):
                        trough_idx = troughs[trough_ptr]

                        # 检查 (troughs - peaks)
                        if trough_idx - peak_idx > 1.2 * icp_period * fs:
                            continue  # 跳过当前峰值

                        # 保存当前峰值和谷值
                        valid_peaks.append(peak_idx)
                        valid_troughs.append(trough_idx)

                # 计算 ave_icp
                beat_means = []
                for j in range(len(valid_peaks)):
                    peak_idx = valid_peaks[j]
                    trough_idx = valid_troughs[j]

                    peak_icp = icp[peak_idx]
                    trough_icp = icp[trough_idx]

                    # 按公式计算 ave_icp
                    ave_icp = (peak_icp - trough_icp) / 3 + trough_icp
                    # print(f"ave_icp[{j}]: {ave_icp}")

                    #  累积每个病人 ave_icp > 20 的时间
                    if ave_icp > 20:
                        # 累加到对应的病人
                        patient_id = row['sub_dirname']
                        if patient_id in time_map:
                            time_map[patient_id] += icp_period
                        else:
                            time_map[patient_id] = icp_period
                        print(f"Patient: {patient_id}, Hypertension Time: {time_map[patient_id]}s")

                        # 累加到对应文件路径 row
                        total_time += icp_period

            time_res.append(
                [row['file_path'], row['sub_dirname'], row['file'], row["start"], row["end"], total_time])
            # print(f"File: {row['file_path']}, Total Hypertension Time: {total_time}s")

        except Exception as e:
            print(f"Error loading file {row['file_path']}: {e}")

    # 持续颅内高压的时间写入 CSV 文件
    output_path = 'result/HypertensionTimeofPatient.csv'
    df = pd.DataFrame(list(time_map.items()), columns=['Patient', 'Hypertension_Time'])
    df.to_csv(output_path, index=False)
    print(f"cumulative hypertension time of each patient saved: {output_path}")

    # 每个文件的持续颅内高压时间写入 CSV 文件
    output_path = 'result/HypertensionTimeofFile.csv'
    df = pd.DataFrame(time_res, columns=['file_path', 'sub_dirname', 'file', 'start', 'end',
                                         'Hypertension_Time'])
    df.to_csv(output_path, index=False)
    print(f"cumulative hypertension time of each file saved: {output_path}")


if __name__=="__main__":
    CalHypertensionTime()
    print("======================================= CalHypertensionTime finished !!! ===========================================")

    # 统计 颅内高压持续时间大于1min，且有效时间大于1小时的病人
    HypertensionTime = pd.read_csv("result/HypertensionTimeofPatient.csv")  # 时间单位为秒
    ValidTime = pd.read_csv("result/validTimeofPatient.csv")   # 时间单位为小时

    # 选择颅内高压持续时间大于1min，且有效时间大于1小时的病人
    HypertensionTime = HypertensionTime[HypertensionTime['Hypertension_Time'] > 60]
    ValidTime = ValidTime[ValidTime['Effective_Time'] > 1]
    # 将两个表合并
    res = pd.merge(HypertensionTime, ValidTime, on='Patient')
    res['Effective_Time'] = res['Effective_Time'] * 3600  # 将时间单位转换为秒
    # 计算持续颅内高压时间占比
    res['Rate'] = res['Hypertension_Time'] / res['Effective_Time']
    # 计算结果写入 CSV 文件
    output_path = 'result/ValidHypertensionTime.csv'
    res.to_csv(output_path, index=False)
    print(f"Valid Hypertension Time saved: {output_path}")



