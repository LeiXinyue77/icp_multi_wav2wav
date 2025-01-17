import joblib
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import torch
from utils.plot import plot_signals


class IcpDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, folders, root_dir, device="cuda:0"):
        """
        Args:
            folders (List): List of folders containing the npy files
            root_dir (string): Directory
        """
        # load data
        self.data = []
        self.info = []
        self.device = device
        for folder in folders:
            # print(folder)
            for root, dirs, files in os.walk(os.path.join(root_dir, folder)):
                for file in files:
                    if file.endswith('.npy'):
                        file_path = os.path.join(root, file)
                        try:
                            npy_data = np.load(file_path)
                            npy_data_min = np.min(npy_data, axis=0)  # 每列的最小值
                            npy_data_max = np.max(npy_data, axis=0)  # 每列的最大值
                            npy_data_range = np.where(npy_data_max - npy_data_min == 0, 1, npy_data_max - npy_data_min)
                            normalized_data = (npy_data - npy_data_min) / npy_data_range
                            self.data.append(normalized_data)
                            self.info.append(file_path[:-4])
                        except Exception as e:
                            print(f"Error reading file {file_path}: {e}")

    def __len__(self):
        # return number of frames
        return int(len(self.data))

    # Will pull an index between 0 and __len__.
    def __getitem__(self, idx):

        # input/target of shape: [n_samples, n_features].
        info = self.info[idx]  # 保持为字符串
        _input = self.data[idx][:, 1:4]
        target = self.data[idx][:, 0].reshape(-1, 1)

        # Convert to tensors and move to CUDA
        _input = torch.tensor(_input).double().to(self.device)
        target = torch.tensor(target).double().to(self.device)

        return info, _input, target


# 设置 CUDA 设备
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# test the dataloader
if __name__ == "__main__":
    folders = ['folder1']
    root_dir = 'data'
    train_dataset = IcpDataset(folders, root_dir, device="cuda:0")
    train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    for i, (info, _input, target) in enumerate(train_dataloader):
        plot_signals(_input[i].cpu().numpy(), ['abp','ppg','ecg'], title="Input")
        plot_signals(target[i].cpu().numpy(), ['icp'], title="Target")
        print(f"Info: {info}, Input shape: {_input.shape}, Target shape: {target.shape}")
