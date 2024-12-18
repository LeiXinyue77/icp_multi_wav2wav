from sys import prefix
from IPython.display import display
from scipy.signal import butter, filtfilt
import ast
import wfdb
import re
import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import csv
import pywt


# 数据预处理
# 1. 第一步：选出含有ICP、ABP信号，且fs=125Hz的记录
#   - 读取pXXNNNN-YYYY-MM-DD-hh-mm.hea fs ，判断fs=125Hz?
#   - 读取 seg_name 对应的xxxxxxx_layout.hea，根据 sig_name 判断是否含有ABP和ICP？
#   - 记录下所有含有ABP和ICP且，fs=125Hz为的pXXNNNN-YYYY-MM-DD-hh-mm。
''' 
# 获取根目录下所有记录列表
root_file_list = wfdb.get_record_list('mimic3wdb-matched', records="all")
root_path = 'mimic3wdb-matched/1.0/'

for subset_file_name in root_file_list:
    # 构造子集路径
    subset_file_path = "".join([root_path, subset_file_name])
    subset_file_list = wfdb.get_record_list(subset_file_path)

    # 正则表达式匹配格式为 pXXNNNN-YYYY-MM-DD-hh-mm
    file_pattern = r'^p\d{6}-\d{4}-\d{2}-\d{2}-\d{2}-\d{2}$'
    matched_files = list(filter(lambda file: re.match(file_pattern, file), subset_file_list))

    sig_flag = false
    fs_flag = false

    # 遍历所有匹配的文件
    for matched_file in matched_files:
        try:
            # 读取记录文件头
            file_record = wfdb.rdheader(matched_file, subset_file_path)
            # display(file_record.__dict__)

            # 提取 seg_name 和采样频率
            segName = file_record.seg_name
            fs = file_record.fs
            fs_flag = fs == 125

            # 匹配 layout 文件
            layout_pattern = r'^\d{7}_layout$'
            matched_layout = list(filter(lambda file: re.match(layout_pattern, file), segName))

            for layout_file in matched_layout:
                try:
                    # 读取 layout 记录
                    layout_record = wfdb.rdheader(layout_file, subset_file_path)
                    # display(layout_record.__dict__)

                    # 检查是否包含目标信号
                    sigName = layout_record.sig_name
                    keywords = ['ABP', 'ICP']
                    sig_flag = all(keyword.lower() in map(str.lower, sigName) for keyword in keywords)

                    if sig_flag:
                        print(f"匹配到关键字信号：{sigName}")
                except Exception as e:
                    print(f"无法读取 layout 文件 {layout_file}：{e}")

            # 文件写入逻辑
            match = re.match(r"p(\d{2})\d{4}-\d{4}-\d{2}-\d{2}-\d{2}", matched_file)
            if sig_flag and fs_flag and match:
                XX = match.group(1)  # 提取 XX 部分
                # folder_name = f"p{XX}"  # 创建对应的文件夹
                folder_name = "pXX"  # 创建对应的文件夹
                os.makedirs(folder_name, exist_ok=True)
                filename = f"{XX}.txt"  # 确定文件名
                file_path = os.path.join(folder_name, filename)

                with open(file_path, "a") as file:
                    file.write(matched_file + "\n")
                print(f"写入成功：{matched_file} -> {file_path}")
            else:
                print(f"文件格式错误：{matched_file}")
        except Exception as e:
            print(f"无法读取记录文件 {matched_file}：{e}")
            
 '''


# 验证第一步筛选结果
'''
# 测试上面的筛选结果, 读取pXX文件夹下的txt文件, 把文件内容存储在panda数组中
file_contents = []
for root, dirs, files in os.walk("pXX", topdown=False):
    for name in files:
        if name.endswith(".txt"):
            file_path = os.path.join(root, name)
            with open(file_path, "r") as file:
                content = file.read().splitlines()  # 读取文件内容
                file_contents.append(content)  # 存储到数组中
                # print(file.read())

# print(file_contents)

## file_contents 是一个二维数组，其中每一行是一个文件的内容
## file_contents = [
##     ["p004904-2193-06-17-03-04", "p006621-2131-05-24-07-05"], ...
##     ["p011003-2119-05-17-14-20", "p011901-2184-05-31-09-44"], ...
##       ...
##     ["p090990-2133-11-08-10-42"], ...
## ]

# 转换 file_contents 为 DataFrame
df = pd.DataFrame(file_contents)
# 保存为 .dat 文件，使用空格分隔
df.to_csv("pXX/file_contents.dat", sep=' ', header=False, index=False, quoting=False)

# 读取文件为 DataFrame，使用空格分隔
df_loaded = pd.read_csv("pXX/file_contents.dat", sep=' ', header=None, engine='python')
print(df_loaded)

# 下面是为了验证fs和sig, 随机选取文件验证
test_matched_file = file_contents[1][2]
test_folder = test_matched_file[:7]
test_pXX = test_matched_file[:3]
test_subset_file = f"{test_pXX}/{test_folder}"
root_path = 'mimic3wdb-matched/1.0/'
test_subset_file_path = "".join([root_path, test_subset_file])
test_file_record = wfdb.rdheader(test_matched_file, test_subset_file_path)
display(test_file_record.__dict__)
test_segName = test_file_record.seg_name
test_fs = test_file_record.fs
print(test_fs)

# 匹配 layout 文件
layout_pattern = r'^\d{7}_layout$'
test_matched_layout = list(filter(lambda file: re.match(layout_pattern, file), test_segName))

for layout_file in test_matched_layout:
    try:
        # 读取 layout 记录
        layout_record = wfdb.rdheader(layout_file, test_subset_file_path)
        # display(layout_record.__dict__)

        # 检查是否包含目标信号
        sigName = layout_record.sig_name
        keywords = ['ABP', 'ICP']
        sig_flag = all(keyword.lower() in map(str.lower, sigName) for keyword in keywords)

        if sig_flag:
            print(f"匹配到关键字信号：{sigName}\n fs={test_fs}")
    except Exception as e:
        print(f"无法读取 layout 文件 {layout_file}：{e}")
'''


# 统计病人个数
'''
df_loaded = pd.read_csv("pXX/file_contents.dat", sep=' ', header=None, engine='python')
# print(df_loaded)
# df_loaded 中每一个元素是一个文件名，格式为 pXXNNNN-YYYY-MM-DD-hh-mm, 其中 XXNNNN 为病人编号
matched_list = df_loaded.values.flatten()
matched_list = list(filter(lambda x: isinstance(x, str) and x.strip() != '', matched_list))
# df = pd.DataFrame(matched_list, columns=["matched_file_name"])
# df.to_csv("pXX/file_content_list.dat", header=False, index=False, quoting=False)

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
df_file_signal_map.to_csv("pXX/file_signal_map.dat", sep=' ', header=False, index=False)

# 将 all_signals 保存为单独的 .dat 文件
df_all_signals = pd.DataFrame({'signals': list(all_signals)})
df_all_signals.to_csv("pXX/all_signals.dat", sep=' ', header=False, index=False)

print("file_signal_map 和 all_signals 已保存为 .dat 文件")
'''


# 根据 file_signal_map 统计包含ICP ABP PLETH 信号的记录 和 病人个数
'''
file_signal_map = pd.read_csv("pXX/file_signal_map.dat", sep=' ', header=None, names=["file", "signals"])
file_signal_map["signals"] = file_signal_map["signals"].str.split(',')

# 筛选包含 ICP, ABP, PLETH 的记录
target_signals = {"ICP", "ABP", "PLETH"}
filtered_records = file_signal_map[file_signal_map["signals"].apply(lambda x: target_signals.issubset(x))]

# 提取病人编号
filtered_records["patient_id"] = filtered_records["file"].str[:7]
unique_patients = filtered_records["patient_id"].nunique()

# 统计结果
print(f"包含 ICP, ABP, PLETH 信号的记录数：{len(filtered_records)}")
print(f"包含 ICP, ABP, PLETH 信号的独立病人个数：{unique_patients}")

# 保存结果
filtered_records.to_csv("pXX/file_signal_map_icp_abp_pleth.dat", sep=' ', index=False, header=False)
file_signal_map.to_csv("pXX/file_signal_map.dat", sep=' ', index=False, header=False)
'''


# 根据 file_signal_map 统计包含ICP ABP PLETH I II III RESP信号的记录 和 病人个数
'''
file_signal_map = pd.read_csv("pXX/file_signal_map.dat", sep=' ', header=None, names=["file", "signals"])
file_signal_map["signals"] = file_signal_map["signals"].apply(ast.literal_eval)
target_signals = {'ICP', 'ABP', 'PLETH', 'I', 'II', 'III', 'RESP'}
filtered_records = file_signal_map[file_signal_map["signals"].apply(lambda x: target_signals.issubset(x))]

filtered_records['patient_id'] = filtered_records['file'].str[:7]
unique_patients = filtered_records["patient_id"].nunique()

print(f"包含目标信号的记录数：{len(filtered_records)}")
print(f"包含目标信号的病人数：{unique_patients}")

filtered_records.to_csv("pXX/file_signal_map_icp_abp_pleth_i_ii_iii_resp.dat", sep=' ', index=False, header=False)
'''


# 1. 第二步：下载筛选信号
'''
file_signal_map = pd.read_csv(
    "result/pre/file_signal_map_icp_abp_pleth.dat", sep=' ', header=None)
file_signal_map.columns = ['file', 'signals', 'patient']
file_signal_map["signals"] = file_signal_map["signals"].apply(ast.literal_eval)
# print(file_signal_map)

start_row = 0
end_row = 22
for index, row in file_signal_map.iloc[start_row:].iterrows():
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
                channel_list = ["ICP", "ABP", "PLETH","II"]
                sig_list = signal_record.sig_name

                # sig_list 是否包含了 channel_list 中所有的信号
                flag = set(channel_list).issubset(sig_list)
                print(flag)
                if flag:
                    sig, fields = wfdb.rdsamp(
                        record_name=signal, pn_dir=subset_file_path, channel_names=channel_list)
                    path = f"pXX/{subset_file}"
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

print("=================================  save finished ===============================")
'''


# # 测试上一步处理结果
'''
# file = file_signal_map['file'][0]
# # print(file)
# folder = file[:7]
# pXX = file[:3]
# subset_file = f"{pXX}/{folder}"
# root_path = 'mimic3wdb-matched/1.0/'
# subset_file_path = "".join([root_path, subset_file])
# file_record = wfdb.rdheader(file, subset_file_path)
# # display(file_record.__dict__)
# segName = file_record.seg_name
# fs = file_record.fs
# # print(fs)
#
# layout_pattern = r'^\d{7}_layout$'
# layout = list(filter(lambda file: re.match(layout_pattern, file), segName))
# # print(layout)
#
# # layout_record = wfdb.rdheader(layout[0], subset_file_path)
# # display(layout_record.__dict__)
# # 根据 layout 文件名 "XXXXXXX_layout", 在segName中找到对应的信号 "XXXXXXX_NNNN"
# segName_prefix = layout[0].replace("_layout", "")
# signals = [
#     seg for seg in segName
#     if seg.startswith(segName_prefix) and seg != f"{segName_prefix}_layout"
# ]
# # print(signals)
# # signal_path =  f"{subset_file_path}/{signals[0]}"
# channel_list = ["ICP", "ABP", "PLETH", "I", "II", "III", "RESP"]
# sig, fields = wfdb.rdsamp(record_name=signals[0], pn_dir=subset_file_path, channel_names=channel_list)
# print(sig)
# print(fields)
# # fields['sig_name'] 是否包含了 channel_list 中所有的信号
# sig_list = fields['sig_name']
# flag = set(channel_list).issubset(sig_list)
# print(flag)
# if flag:
#     path =  f"pXX/{subset_file}"
#     if not os.path.exists(path):
#         os.makedirs(path)
#     file_path = os.path.join(path, f"{signals[0]}.dat")
#     with open(file_path, 'wb') as f:
#         pickle.dump({'fields': fields, 'sig': sig}, f)
#     print(f"保存{file_path}")

'''


# 3. 第三步：波形读取与预处理
def plot_signals(sigs, chl, title=" "):
    """
    参数：
        signals (numpy.ndarray): 信号数组，形状为 (样本数, 通道数)。
        channel (list): 每个通道的名称列表。
        title (string): 标题
    """
    if sigs.shape[1] != len(chl):
        raise ValueError("signals 通道数和 channel 名称数量不匹配！")

    # 创建一个 Nx1 的子图布局
    fig, axes = plt.subplots(sigs.shape[1], 1, figsize=(8, 12))
    fig.suptitle(title, fontsize=16)

    # 遍历每个通道并绘制到对应的子图
    for n in range(sigs.shape[1]):  # 遍历通道
        ax = axes[n]  # 获取第 n 个子图
        ax.plot(sigs[:, n], label=f"{chl[n]}")
        ax.set_title(f"{chl[n]}")
        ax.set_xlabel("Sample")
        ax.set_ylabel("Amplitude")
        ax.legend()
        ax.grid(True)

    # 调整子图布局以避免重叠
    plt.tight_layout()
    # 显示图像
    plt.show()



def save_signals(sigs, chl, title=" ", save_path=None):
    """
    参数：
        signals (numpy.ndarray): 信号数组，形状为 (样本数, 通道数)。
        channel (list): 每个通道的名称列表。
        title (string): 标题
        save_path (string): 保存图像的路径
    """
    if sigs.shape[1] != len(chl):
        raise ValueError("signals 通道数和 channel 名称数量不匹配！")

    # 创建一个 Nx1 的子图布局
    fig, axes = plt.subplots(sigs.shape[1], 1, figsize=(8, 12))
    fig.suptitle(title, fontsize=16)

    # 遍历每个通道并绘制到对应的子图
    for n in range(sigs.shape[1]):  # 遍历通道
        ax = axes[n]  # 获取第 n 个子图
        ax.plot(sigs[:, n], label=f"{chl[n]}")
        ax.set_title(f"{chl[n]}")
        ax.set_xlabel("Sample")
        ax.set_ylabel("Amplitude")
        ax.legend()
        ax.grid(True)

    # 调整子图布局以避免重叠
    plt.tight_layout()

    # 保存图像到指定路径
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)  # 创建文件夹（如果不存在）
        fig_name = f"{title}.png"  # 图像文件名
        path_fig = os.path.join(save_path, "png")
        plt.savefig(os.path.join(path_fig, fig_name))  # 保存文件
        print(f"fig saved: {path_fig}/{fig_name}")

        # 保存信号数据
        data_name = f"{title}.npy"
        path_data = os.path.join(save_path, "npy")
        data_file = os.path.join(path_data, f"{title}.npy")  # 保存为 .npy 格式
        np.save(data_file, sigs)
        print(f"data saved: {path_data}/{data_name}")

    plt.close()  # 关闭图像，以避免显示时占用内存


# 3.1 统计时间大于 5min 的信号文件
'''
root_path = "pXX"
pattern = re.compile(r"^p0[0-9]$")
min_length = 125 * 60 * 5
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
csv_datLarge5min = os.path.join("result/pre", "datLarge5min.csv")
with open(csv_datLarge5min, mode="a", newline="", encoding="utf-8") as csv_file:
    csv_writer = csv.writer(csv_file)
    if csv_file.tell() == 0:
        csv_writer.writerow(["file_path", "sub_dirname", "file"])  # 写入表头
    csv_writer.writerows(res_datLarge5min)
print(
    f" ========================= save finished: {csv_datLarge5min} !!! ========================")
'''

# 3.2 no NaN values present in any signal
'''
# 读取 datLarge5min.csv 文件
datLarge5min = pd.read_csv("result/pre/datLarge5min.csv")
min_length = 125 * 60 * 5
# 根据 datLarge5min 中的文件路径读取数据
res_noNaN = []
for index, row in datLarge5min.iterrows():
    try:
        with open(row["file_path"], "rb") as f:
            data = pickle.load(f)
        signals = data.get("sig", None)

        if signals is not None:
            valid_indices = ~np.isnan(signals).any(axis=1)  # 一行中无 NaN 返回 True
            # valid_signals = np.where(valid_indices)[0]  # 保留有效行
            segments = []
            start_idx = None
            for i, valid in enumerate(valid_indices):
                if valid:
                    # 如果当前行有效，且上一行无效，则开始新的段
                    if start_idx is None:
                        start_idx = i
                else:  # 如果当前行无效
                    if start_idx is not None:  # 且上一行有效，则结束当前段
                        segments.append((start_idx, i))  # 记录当前段的开始和结束索引
                        start_idx = None

            # 检测最后一段
            if start_idx is not None:
                segments.append((start_idx, len(valid_indices) - 1))

            # 检查每个段的长度是否大于 5min，如果是则保存
            for start, end in segments:
                if end - start >= min_length:
                    print(
                        f"Valid segment: {row['file_path']} [{start}, {end}]")
                    res_noNaN.append(
                        [row['file_path'], row['sub_dirname'], row['file'], start, end])
    except Exception as e:
        print(f"Error loading file {row['file_path']}: {e}")

# 使用集合 set 去重 res_noNaN 中的 sub_dirname
noNaN_patient_set = set(
    sub_dirname for _, sub_dirname, _, _, _ in res_noNaN)
print(
    f"No NaN values present in any signal的病人总数: {len(noNaN_patient_set)}")

# 写入CSV文件
csv_noNaN = os.path.join("result/pre", "noNaN.csv")
with open(csv_noNaN, mode="a", newline="", encoding="utf-8") as csv_file:
    csv_writer = csv.writer(csv_file)
    if csv_file.tell() == 0:
        csv_writer.writerow(
            ["file_path", "sub_dirname", "file", "start", "end"])  # 写入表头
    csv_writer.writerows(res_noNaN)

print("============================================= NaN finished =====================================")
'''


# 将signal[start: end]划分为5min的片段，每个片段之间有30%的重叠
def splitSig(signal, sample_freq=125, segLenth=60*5, overlap=0.3):
    """
    将信号划分为带重叠的片段

    参数：
        signal (ndarray): 输入的信号数组。
        sample_freq (int): 采样率（Hz）。
        segment_length (int): 每个片段的长度（秒）。
        overlap (float): 重叠百分比（0 到 1）。

    返回：
        splitSeg (list of ndarray): 符合条件的信号片段。
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

    参数:
        signal: 输入的时间序列数据
        fs: 采样频率 (Hz)
        wavelet: 小波基类型 (默认为 'morl')
        scales_range: 小波尺度范围 (默认为从1到128)

    返回:
        dominant_period: 主导周期（秒）
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

        参数：
        icp: 输入的ICP信号 (numpy.ndarray)
        fs: 采样频率 (Hz)

        返回：
        period: 信号的主周期 (秒)
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


# 3.3 对信号进行分段、low-pass滤波等操作
# 结果暂存
res_normal = []
# 滤波器设计
order = 3
cutoff_freq = 5
sampling_freq = 125
normalized_cutoff_freq = cutoff_freq / (sampling_freq / 2)
(b, a) = butter(order, normalized_cutoff_freq, btype='low', analog=False)
print(a, b)
# 读取 noNaN.csv 文件
noNaN = pd.read_csv("result/pre/noNaN.csv")
noNaN_partial = noNaN.iloc[0:2]
num = 0  # 记录处理过的文件个数
# 根据 noNaN 中的文件路径读取数据
for index, row in noNaN.iterrows():
    try:
        with open(row["file_path"], "rb") as f:
            data = pickle.load(f)
        start = row['start']
        end = row['end']
        signals = data.get("sig", None)[start:end]
        fields = data.get("fields", None)
        # if signals is None or fields is None:
        #     print(f"Invalid data in file {row['file_path']}")
        #     continue
        fs = fields.get('fs', None)
        # if fs is None or fs != 125:
        #     print(f"Invalid fs in file {row['file_path']}")
        #     continue
        channel = fields.get('sig_name', [])

        # 1) low-pass 滤波
        filtered_icp = filtfilt(b, a, signals[:, 0])
        filtered_sig = signals.copy()
        filtered_sig[:, 0] = filtered_icp
        file = row["file_path"]
        # plot_signals(sigs=filtered_sig, chl=channel, title=f"filtered-{file}")
        # plot_signals(sigs=signals, chl=channel, title=f"raw-{file}")

        # 2）10s一段检测波形是否符合要求
        seg_len = 10 * fs  # 每段10秒
        # order = 0

        for i in range(0, len(filtered_icp), seg_len):
            # num >= 200 结束遍历
            if num >= 200:
                break
            # normal = False
            # order += 1
            icp = filtered_icp[i:i + seg_len]
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
                save_path = f"svm_miniset/0/{patient[:4]}/{patient}"
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
                save_path = f"svm_miniset/0/{patient[:4]}/{patient}"
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
                save_path = f"svm_miniset/0/{patient[:4]}/{patient}"
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
            save_path = f"svm_miniset/1/{patient[:4]}/{patient}"
            save_signals(
                sigs=filtered_sig[i:i + seg_len], chl=channel,
                title=f"{num}-{file.split('.')[0]}-{i}-{i + seg_len}",
                save_path=save_path)
            num += 1

            # 机器学习分类 ?

    except Exception as e:
        print(f"Error loading file {row['file_path']}: {e}")

print("save fig and data finished !!!")
