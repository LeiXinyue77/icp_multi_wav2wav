import joblib
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import torch
from utils.plot import plot_signals

class IcpDataset(Dataset):
    """ICP Dataset with global normalization."""

    def __init__(self, folders, root_dir, device="cuda:0"):
        """
        Args:
            folders (List): List of folders containing the npy files
            root_dir (string): Directory of the dataset
        """
        self.data = []
        self.info = []
        self.device = device

        # 用于存储全局最小值和最大值
        all_data = []

        # 第一次遍历：收集所有数据以计算全局统计量
        for folder in folders:
            for root, dirs, files in os.walk(os.path.join(root_dir, folder)):
                for file in files:
                    if file.endswith('.npy'):
                        file_path = os.path.join(root, file)
                        try:
                            npy_data = np.load(file_path)
                            all_data.append(npy_data)  # 将数据加入全局数据列表
                        except Exception as e:
                            print(f"Error reading file {file_path}: {e}")

        # 合并所有数据计算全局最小值、最大值
        all_data = np.concatenate(all_data, axis=0)
        global_min = np.min(all_data)
        global_max = np.max(all_data)
        global_range = global_max - global_min if global_max != global_min else 1  # 防止除以0

        # 第二次遍历：对每个文件的数据进行标准化
        for folder in folders:
            for root, dirs, files in os.walk(os.path.join(root_dir, folder)):
                for file in files:
                    if file.endswith('.npy'):
                        file_path = os.path.join(root, file)
                        try:
                            npy_data = np.load(file_path)

                            # 使用全局最小值和最大值进行归一化
                            normalized_data = (npy_data - global_min) / global_range

                            self.data.append(normalized_data)
                            self.info.append(file_path[:-4])
                        except Exception as e:
                            print(f"Error reading file {file_path}: {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        info = self.info[idx]
        _input = self.data[idx][:, 1:4]  # 输入信号：ABP, PPG, ECG
        target = self.data[idx][:, 0].reshape(-1, 1)  # 目标信号：ICP

        # 转换为 PyTorch 张量并移动到设备
        _input = torch.tensor(_input).double().to(self.device)
        target = torch.tensor(target).double().to(self.device)

        return info, _input, target


# 设置 CUDA 设备
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == "__main__":
    folders = ['folder1']
    root_dir = 'data'
    train_dataset = IcpDataset(folders, root_dir, device="cuda:0")
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    for i, (info, _input, target) in enumerate(train_dataloader):
        plot_signals(_input[i].cpu().numpy(), ['abp', 'ppg', 'ecg'], title="Input")
        plot_signals(target[i].cpu().numpy(), ['icp'], title="Target")
        print(f"Info: {info}, Input shape: {_input.shape}, Target shape: {target.shape}")