import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from sklearn.preprocessing import MinMaxScaler
import os
import torch

from utils.plot import plot_signals


class IcpDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, folders, root_dir):
        """
        Args:
            folders (List): List of folders containing the npy files
            root_dir (string): Directory
        """
        # load data
        self.data = []
        self.info = []
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for folder in folders:
            # print(folder)
            for root, dirs, files in os.walk(os.path.join(root_dir, folder)):
                for file in files:
                    if file.endswith('.npy'):
                        file_path = os.path.join(root, file)
                        try:
                            npy_data = np.load(file_path)
                            # print(npy_data.shape)
                            self.data.append(npy_data)
                            self.info.append(file_path[:-4])
                        except Exception as e:
                            print(f"Error reading file {file_path}: {e}")

        # Initialize scalers for normalization
        self.input_scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))

        # Fit scalers using all data
        all_inputs = np.concatenate([data[:, 1:4] for data in self.data], axis=0)
        all_targets = np.concatenate([data[:, 0] for data in self.data], axis=0).reshape(-1, 1)
        self.input_scaler.fit(all_inputs)
        self.target_scaler.fit(all_targets)

    def __len__(self):
        # return number of frames
        return int(len(self.data))

    # Will pull an index between 0 and __len__.
    def __getitem__(self, idx):

        # input/target of shape: [n_samples, n_features].
        info = self.info[idx]  # 保持为字符串
        _input = self.data[idx][:, 1:4]
        target = self.data[idx][:, 0].reshape(-1, 1)

        # Normalize input and target
        _input = self.input_scaler.transform(_input)
        target = self.target_scaler.transform(target)

        # Convert to tensors and move to CUDA
        _input = torch.tensor(_input).double().to(self.device)
        target = torch.tensor(target).double().to(self.device)

        return info, _input, target


# test the dataloader
if __name__ == "__main__":
    folders = ['folder1', 'folder2', 'folder3', 'folder4']
    root_dir = 'data'
    train_dataset = IcpDataset(folders, root_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    for i, (info, _input, target) in enumerate(train_dataloader):
        plot_signals(_input[0].cpu().numpy(), ['abp', 'ppg', 'ecg'], title="Input")
        plot_signals(target[0].cpu().numpy(), ['icp'], title="Target")
        print(f"Info: {info}, Input shape: {_input.shape}, Target shape: {target.shape}")
