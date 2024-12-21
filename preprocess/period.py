"""
@Project ：
@File    ：preprocess/period.py
@Author  ：Lei Xinyue
@Date    ：2024/12/21 15:13
@Description: 训练前信号预处理
"""
from svm.save_data import save_signals
from utils.signal_process import lowpass_filter, bandpass_filter, period_autocorrelation
import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np


# 4 对信号进行分段、滤波、计算周期等操作
'''
  4.1 统计No NaN in mean HR 的病人总数
        逐个片段计算，符合要求的片段累计5min，则保留
'''
# 结果暂存
# res = []
# 读取 validABP.csv 文件
dataList = pd.read_csv("result/pre/validABP.csv")
num = 0  # 记录处理过的文件个数
# 根据 validABP 中的文件路径读取数据
for index, row in dataList.iterrows():
    try:
        row_path = os.path.join("data", row["file_path"])
        with open(row_path, "rb") as f:
            data = pickle.load(f)
        start = row['start']
        end = row['end']
        signals = data.get("sig", None)[start:end]
        fields = data.get("fields", None)
        fs = fields.get('fs', None)
        channel = fields.get('sig_name', [])

        # 1) low-pass 滤波
        filicp = lowpass_filter(signals[:, 0], fs, N=65, high=5)
        filabp = lowpass_filter(signals[:, 1], fs, N=9, high=5)
        filppg = lowpass_filter(signals[:, 2], fs, N=9, high=5)
        filecg = bandpass_filter(signals[:, 3], fs, low=1, high=20)
        filtered_sig = signals.copy()
        filtered_sig[:, 0] = filicp
        file = row["file_path"]
        # plot_signals(sigs=filtered_sig, chl=channel, title=f"filtered-{file}")
        # plot_signals(sigs=signals, chl=channel, title=f"raw-{file}")

        # 2）10s一段检测波形是否符合要求
        seg_len = 10 * fs  # 每段10秒
        # order = 0

        for i in range(0, len(filicp), seg_len):
            # num >= 200 结束遍历
            if num >= 200:
                break
            # normal = False
            # order += 1
            icp = filicp[i:i + seg_len]
            abp = signals[i:i + seg_len, 1]
            # plot_splotignals(sigs=filtered_sig[i:i + seg_len], chl=channel, title=f"filtered-{i}-{i+seg_len}")
            # plot_signals(sigs=signals[i:i + seg_len], chl=channel, title=f"raw-{i}-{i+seg_len}")

            # plt.plot(icp)
            # plt.title(f"Filtered ICP segment {i}-{i + seg_len}")
            # plt.show()

            # 1) 排除ICP ABP 异常值
            if (icp.min() < -10) or (icp.max() > 200) or (abp.min() < 20) or (abp.max() > 300):
                # 滤波结果写入文件
                # res_normal.append(
                #     [row['file_path'], row['sub_dirname'], row['file'], order, i, i+seg_len, normal])
                print("ICP ABP 异常值 ")
                patient = row['sub_dirname']
                file = row['file']
                save_path = f"svm_miniset/0/{patient[:3]}/{patient}"
                save_signals(
                    sigs=filtered_sig[i:i + seg_len], chl=channel,
                    title=f"{num}-{file.split('.')[0]}-{i}-{i + seg_len}",
                    save_path=save_path)
                num += 1
                continue
            # 利用自相关函数求icp abp主周期
            icp_period = period_autocorrelation(icp, fs)
            abp_period = period_autocorrelation(abp, fs)

            # 2) 其中一个返回值为none
            if (icp_period is None) or (abp_period is None):
                print(f"未找到周期性峰值 icp_period: {icp_period}, abp_period: {abp_period}")
                patient = row['sub_dirname']
                file = row['file']
                save_path = f"svm_miniset/0/{patient[:3]}/{patient}"
                save_signals(
                    sigs=filtered_sig[i:i + seg_len], chl=channel,
                    title=f"{num}-{file.split('.')[0]}-{i}-{i + seg_len}",
                    save_path=save_path)
                num += 1
                continue


            print(f"icp_period: {icp_period}, abp_period: {abp_period}, diff: {abs(icp_period-abp_period)}")

            # 3) ICP ABP 周期差异大于0.01
            if abs(icp_period - abp_period) > 0.01 or icp_period < 0.56 or icp_period > 1.04:
                # 保存0类
                # 需要同时保存图片和数据，命名(XXXXX_NNNN_start_end) 保持一致（图片用于人工分类，数据用于训练
                patient = row['sub_dirname']
                file = row['file']
                save_path = f"svm_miniset/0/{patient[:3]}/{patient}"
                save_signals(
                    sigs=filtered_sig[i:i + seg_len], chl=channel,
                    title=f"{num}-{file.split('.')[0]}-{i}-{i + seg_len}",
                    save_path=save_path)
                num += 1
                continue

            # 保存1类
            # 需要同时保存图片和数据，命名保持一致（图片用于人工分类，数据用于训练
            patient = row['sub_dirname']
            file = row['file']
            save_path = f"svm_miniset/1/{patient[:3]}/{patient}"
            save_signals(
                sigs=filtered_sig[i:i + seg_len], chl=channel,
                title=f"{num}-{file.split('.')[0]}-{i}-{i + seg_len}",
                save_path=save_path)
            num += 1

            # 机器学习分类 ?

    except Exception as e:
        print(f"Error loading file {row['file_path']}: {e}")

print("save fig and data finished !!!")
