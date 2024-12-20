"""
@Project ：
@File    ：preprocess/select.py
@Author  ：Lei Xinyue
@Date    ：2024/12/20 20:32
@Description: 数据筛选
"""

import wfdb
import re
import os


def select():
    '''
     选出含有ICP、ABP信号，且fs=125Hz的记录
       - 读取pXXNNNN-YYYY-MM-DD-hh-mm.hea fs ，判断fs=125Hz?
       - 读取 seg_name 对应的xxxxxxx_layout.hea，根据 sig_name 判断是否含有ABP和ICP？
       - 记录下所有含有ABP和ICP且，fs=125Hz为的pXXNNNN-YYYY-MM-DD-hh-mm。
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

        sig_flag = False
        fs_flag = False

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
                    folder_name = "data/pXX"  # 创建对应的文件夹
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


if __name__ == '__main__':
    select()

