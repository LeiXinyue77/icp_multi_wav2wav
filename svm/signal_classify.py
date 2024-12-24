"""
@Project ：
@File    ：svm/signal_classify.py
@Author  ：Lei Xinyue
@Date    ：2024/12/23 21:45
@Description: 机器学习信号分类：正常信号、异常信号
"""
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import dump, load


# 数据加载函数
def load_data(base_path):
    """
       遍历指定路径下的第三级子文件夹中的 .npy 文件并加载数据
       :param base_path: 数据文件的根目录 (如 svm/data/v2/0 或 svm/data/v2/1)
       :return: x_data (特征数据), y_data (标签数据)
       """
    x_data = []
    y_data = []

    # 遍历 0 和 1 文件夹 (分别代表异常信号和正常信号)
    for label in ["0", "1"]:
        # label_path = os.path.join(base_path, label)
        label_path = f"{base_path}/{label}"
        if not os.path.isdir(label_path):
            continue

        # 遍历第一级子文件夹 (如 pXX)
        for first_level_folder in os.listdir(label_path):
            # first_level_path = os.path.join(label_path, first_level_folder)
            first_level_path = f"{label_path}/{first_level_folder}"
            if not os.path.isdir(first_level_path):
                continue

            # 遍历第二级子文件夹 (如 pNNNNNN)
            for second_level_folder in os.listdir(first_level_path):
                # second_level_path = os.path.join(first_level_path, second_level_folder)
                second_level_path = f"{first_level_path}/{second_level_folder}"
                if not os.path.isdir(second_level_path):
                    continue

                # 遍历第三级子文件夹 (如 npy)
                for third_level_folder in os.listdir(second_level_path):
                    # third_level_path = os.path.join(second_level_path, third_level_folder)
                    third_level_path = f"{second_level_path}/{third_level_folder}"
                    if not os.path.isdir(third_level_path):
                        continue

                    # 遍历 .npy 文件
                    for file_name in os.listdir(third_level_path):
                        if file_name.endswith(".npy"):
                            file_path = os.path.join(third_level_path, file_name)
                            try:
                                # 加载 .npy 文件数据
                                data = np.load(file_path)
                                # 检查数据形状是否正确
                                if data.shape != (1250,4):
                                    print(f"数据形状错误: {file_path}, 形状: {data.shape}")
                                    continue
                                x_data.append(data)
                                y_data.append(int(label))  # 0 表示异常信号，1 表示正常信号
                            except Exception as e:
                                print(f"加载文件失败: {file_path}, 错误: {e}")



    return np.array(x_data), np.array(y_data)


if __name__ == '__main__':
    # 加载数据
    x_data, y_data = load_data("data/v2")

    # 打印数据形状
    print("x_data.shape:", x_data.shape)
    print("y_data.shape:", y_data.shape)

    print("normal signal:", np.sum(y_data == 1))
    print("abnormal signal:", np.sum(y_data == 0))

    # 针对多维数组的归一化到 [0, 1]
    # x_data.shape: (样本数, 数据点数, 通道数)
    min_vals = x_data.min(axis=1, keepdims=True)  # 找到每个样本、每个通道的最小值
    max_vals = x_data.max(axis=1, keepdims=True)  # 找到每个样本、每个通道的最大值
    range_vals = max_vals - min_vals  # 最大值与最小值之差

    # 处理最大值和最小值相等的情况，避免除以0
    range_vals[range_vals == 0] = 1

    # 归一化
    x_data = (x_data - min_vals) / range_vals

    # 打印归一化后的数据形状确认
    print("Normalized x_data.shape:", x_data.shape)

    # x_data flatten
    x_data = x_data.reshape(x_data.shape[0], -1)
    print("x_data.shape after flatten:", x_data.shape)


    print("======================================= laod_data finished !!! ===========================================")

    # 将数据划分为训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

    # 初始化并训练 SVM 模型
    svm_model = SVC(kernel='linear', C=1.0, random_state=42)
    svm_model.fit(x_train, y_train)

    # 保存模型
    dump(svm_model, 'svm_model.joblib')
    print("SVM 模型已保存到 'svm_model.joblib'")

    # 加载模型并进行预测
    loaded_model = load('svm_model.joblib')
    y_pred = loaded_model.predict(x_test)

    # 输出分类报告
    print("分类报告:")
    print(classification_report(y_test, y_pred))

    print("======================================= signal_classify finished !!! ===========================================")


