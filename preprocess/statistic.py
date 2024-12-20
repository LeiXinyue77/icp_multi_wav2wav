"""
@Project ：
@File    ：preprocess/statistics.py
@Author  ：Lei Xinyue
@Date    ：2024/12/20 21:33
@Description: 信号种类和病人数目统计
"""
import csv
import os
import pickle
import re
import pandas as pd
import wfdb


def statistic_signal():
    """
      # 统计信号总类（38）all_signal.dat
      # file_content.dat ---> file_content_list.dat
      # 统计记录总数（462）file_signal_map.dat
    """
    df_loaded = pd.read_csv("data/pXX/file_contents.dat", sep=' ', header=None, engine='python')
    # print(df_loaded)
    # df_loaded 中每一个元素是一个文件名，格式为 pXXNNNN-YYYY-MM-DD-hh-mm, 其中 XXNNNN 为病人编号
    matched_list = df_loaded.values.flatten()
    matched_list = list(filter(lambda x: isinstance(x, str) and x.strip() != '', matched_list))
    df = pd.DataFrame(matched_list, columns=["matched_file_name"])
    df.to_csv("pXX/file_content_list.dat", header=False, index=False, quoting=False)

    patient_list = list(map(lambda x: x[:7], matched_list))
    patient_count = len(set(patient_list))
    print(f"病人个数：{patient_count}")

    # 读取 matched_list 每个记录的数据，记录每个数据采集了哪些生理信号
    root_path = 'mimic3wdb-matched/1.0/'
    # 记录所有信号的集合
    all_signals = set()
    file_signal_map = []
    for matched_file in matched_list:
        try:
            folder = matched_file[:7]
            pXX = matched_file[:3]
            subset_file = f"{pXX}/{folder}"
            subset_file_path = "".join([root_path, subset_file])
            file_record = wfdb.rdheader(matched_file, subset_file_path)
            segName = file_record.seg_name

            # 匹配 layout 文件
            layout_pattern = r'^\d{7}_layout$'
            matched_layout = list(filter(lambda file: re.match(layout_pattern, file), segName))
            file_signals = set()  # 当前文件的信号集合
            for layout_file in matched_layout:
                try:
                    layout_record = wfdb.rdheader(layout_file, subset_file_path)
                    sigName = layout_record.sig_name
                    file_signals.update(sigName)  # 更新当前文件的信号集合
                    all_signals.update(sigName)  # 更新全局信号集合
                    print(f"{matched_file} -> {sigName}")
                except Exception as e:
                    print(f"无法读取 layout 文件 {layout_file}：{e}")

            # 将当前文件和其信号添加到结果列表
            file_signal_map.append({
                "matched_file": matched_file,
                "signals": ','.join(file_signals)  # 用逗号分隔信号存储为字符串
            })

        except Exception as e:
            print(f"无法读取记录文件 {matched_file}：{e}")

    df_file_signal_map = pd.DataFrame(file_signal_map)
    df_file_signal_map.to_csv("data/pXX/file_signal_map.dat", sep=' ', header=False, index=False)

    # 将 all_signals 保存为单独的 .dat 文件
    df_all_signals = pd.DataFrame({'signals': list(all_signals)})
    df_all_signals.to_csv("data/pXX/all_signals.dat", sep=' ', header=False, index=False)

    print("file_signal_map 和 all_signals 已保存为 .dat 文件")



# 根据 file_signal_map 统计包含ICP ABP PLETH 信号的记录 和 病人个数
def statistics_target(target_signals, save_file):
    """
    据 file_signal_map 统计包含目标信号的记录和病人个数
    :param target_signals: 目标信号集合 e.g. {"ICP", "ABP", "PLETH"}
    :param save_file: 保存文件名
    :return:
    """
    file_signal_map = pd.read_csv("pXX/file_signal_map.dat", sep=' ', header=None, names=["file", "signals"])
    file_signal_map["signals"] = file_signal_map["signals"].str.split(',')

    # 筛选包含 ICP, ABP, PLETH 的记录
    # target_signals = {"ICP", "ABP", "PLETH"}
    filtered_records = file_signal_map[file_signal_map["signals"].apply(lambda x: target_signals.issubset(x))]

    # 提取病人编号
    filtered_records["patient_id"] = filtered_records["file"].str[:7]
    unique_patients = filtered_records["patient_id"].nunique()

    # 统计结果
    print(f"包含{target_signals}信号的记录数：{len(filtered_records)}")
    print(f"包含{target_signals}信号的独立病人个数：{unique_patients}")

    # 保存结果
    save_path= f"data/pXX/{save_file}.dat"
    filtered_records.to_csv(save_path, sep=' ', index=False, header=False)


def statistics_Large5min(save_path, save_file, min_length):
    # 3.1 统计时间大于 5min 的信号文件
    root_path = "data/pXX"
    pattern = re.compile(r"^p0[0-9]$")
    # min_length = 125 * 60 * 5
    res_datLarge5min = []
    patient_set = set()

    for dirpath, dirnames, filenames in os.walk(root_path):
        for dirname in dirnames:
            # 继续遍历子目录
            for sub_dirpath, sub_dirnames, sub_filenames in os.walk(os.path.join(dirpath, dirname)):
                for sub_dirname in sub_dirnames:
                    # 继续遍历子目录
                    # 根据 sub_dirname 统计所有包含 icp abp pleth ii 的病人数目
                    patient_set.add(sub_dirname)
                    for path, dirs, files in os.walk(os.path.join(sub_dirpath, sub_dirname)):
                        for file in files:
                            file_path = os.path.join(path, file)
                            try:
                                with open(file_path, "rb") as f:
                                    data = pickle.load(f)
                                fields = data.get("fields", {})
                                signals = data.get("sig", None)
                                # channel = fields.get('sig_name', [])
                                # plot_signals(signals, channel)
                                # 1) len(signals) >  min_length
                                if signals is not None and len(signals) >= min_length:
                                    res_datLarge5min.append(
                                        [file_path, sub_dirname, file])
                                    print(f"{file_path} done")
                            except Exception as e:
                                print(f"Error loading file {file_path}: {e}")

    # 使用集合去重 res_datLarge5min 中的 sub_dirname
    datLarge5min_patient_set = set(
        sub_dirname for _, sub_dirname, _ in res_datLarge5min)
    print(f" 包含 ICP , ABP, PLETH, II 的病人总数: {len(patient_set)}")
    print(
        f" len(signals) >= min_length 的病人总数: {len(datLarge5min_patient_set)}")

    # 写入CSV文件
    csv_datLarge5min = os.path.join(save_path, save_file)
    with open(csv_datLarge5min, mode="a", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        if csv_file.tell() == 0:
            csv_writer.writerow(["file_path", "sub_dirname", "file"])  # 写入表头
        csv_writer.writerows(res_datLarge5min)



if __name__ == '__main__':
    '''
    # 统计信号总类（38）all_signal.dat
    # file_content.dat ---> file_content_list.dat
    # 统计记录总数（462）file_signal_map.dat
    '''

    statistic_signal()

    # 统计包含 ICP, ABP, PLETH 信号的记录 和 病人个数
    target_signals = {"ICP", "ABP", "PLETH"}
    statistics_target(target_signals, "file_signal_map_icp_abp_pleth")

    statistics_Large5min("data/pXX", "csv_datLarge5min.csv", 125 * 60 * 5)
    print(
        f" ========================= save finished: csv_datLarge5min !!! ========================")

