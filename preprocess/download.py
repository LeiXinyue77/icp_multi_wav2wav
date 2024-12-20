"""
@Project ：
@File    ：preprocess/download.py
@Author  ：Lei Xinyue
@Date    ：2024/12/20 13:36
@Description: 训练前信号预处理
"""

import ast
import os
import re
import pandas as pd
import wfdb
import pickle


def download_target_signals(channel_list, save_path):
    """
    下载目标信号
    :param channel_list:
    :param save_path:
    :return:
    """
    file_signal_map = pd.read_csv(
        "result/pre/file_signal_map_icp_abp_pleth.dat", sep=' ', header=None)
    file_signal_map.columns = ['file', 'signals', 'patient']
    file_signal_map["signals"] = file_signal_map["signals"].apply(ast.literal_eval)
    # print(file_signal_map)

    for index, row in file_signal_map.iterrows():
        file = row['file']
        folder = file[:7]
        pXX = file[:3]
        subset_file = f"{pXX}/{folder}"
        root_path = 'mimic3wdb-matched/1.0/'
        subset_file_path = "".join([root_path, subset_file])
        file_record = wfdb.rdheader(file, subset_file_path)
        # display(file_record.__dict__)
        segName = file_record.seg_name
        fs = file_record.fs

        layout_pattern = r'^\d{7}_layout$'
        layout = list(filter(lambda file: re.match(layout_pattern, file), segName))
        # print(layout)

        # 根据 layout 文件名 "XXXXXXX_layout", 在segName中找到对应的信号 "XXXXXXX_NNNN"
        for layout_file in layout:
            segName_prefix = layout_file.replace("_layout", "")
            signals = [
                seg for seg in segName
                if seg.startswith(segName_prefix) and seg != f"{segName_prefix}_layout"
            ]
            for signal in signals:
                try:
                    signal_record = wfdb.rdheader(signal, subset_file_path)
                    # channel_list = ["ICP", "ABP", "PLETH","II"]
                    sig_list = signal_record.sig_name

                    # sig_list 是否包含了 channel_list 中所有的信号
                    flag = set(channel_list).issubset(sig_list)
                    print(flag)
                    if flag:
                        sig, fields = wfdb.rdsamp(
                            record_name=signal, pn_dir=subset_file_path, channel_names=channel_list)
                        path = f"{save_path}/{subset_file}"
                        if not os.path.exists(path):
                            os.makedirs(path)
                        file_path = os.path.join(path, f"{signal}.dat")
                        with open(file_path, 'wb') as f:
                            pickle.dump({'fields': fields, 'sig': sig}, f)
                        print(f"保存{file_path}")
                    else:
                        print(f"{signal}:{sig_list}")
                except Exception as e:
                    print(f"Error processing signal {signal}: {e}")


if __name__ == '__main__':
    channel_list = ["ICP", "ABP", "PLETH", "II"]
    save_path = "data/pXX"
    download_target_signals(channel_list, save_path)
    print("=================================  save finished ===============================")

