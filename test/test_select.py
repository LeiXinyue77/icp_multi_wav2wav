"""
@Project ：
@File    ：preprocess.py
@Author  ：Lei Xinyue
@Date    ：2024/12/20 20:32
@Description: 测试数据预处理
"""
import os
import pandas as pd
import wfdb
import re
from IPython.display import display


def test_select():
    """
      选出含有ICP、ABP信号，且fs=125Hz的记录
       - 读取pXXNNNN-YYYY-MM-DD-hh-mm.hea fs ，判断fs=125Hz?
       - 读取 seg_name 对应的xxxxxxx_layout.hea，根据 sig_name 判断是否含有ABP和ICP？
       - 记录下所有含有ABP和ICP且，fs=125Hz为的pXXNNNN-YYYY-MM-DD-hh-mm。
    """
    # 测试 preprocess/select.py 筛选结果, 读取pXX文件夹下的txt文件, 把文件内容存储在panda数组中
    file_contents = []
    for root, dirs, files in os.walk("data/pXX", topdown=False):
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
    df.to_csv("data/pXX/file_contents.dat", sep=' ', header=False, index=False, quoting=False)

    # 读取文件为 DataFrame，使用空格分隔
    df_loaded = pd.read_csv("data/pXX/file_contents.dat", sep=' ', header=None, engine='python')
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


if __name__ == '__main__':
    # 测试 preprocess/select.py 筛选结果
    test_select()