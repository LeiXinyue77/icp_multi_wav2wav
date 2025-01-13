import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from dl.model.idv_net.IDV_NET import IVD_Net_asym
from dl.model.multi_wav_unet.MW_UNET import Multi_Wav_UNet
from dl.model.multi_wav_unet.mw_unet_separableConv import Multi_Wav_UNet_SeparableConv
from utils.plot import plot_signals

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def plot_pred_signals(signals, titles, suptitle):
    """
    绘制四行信号，每行一组信号
    """
    fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    for i in range(3):
        axs[i].plot(signals[i], label=titles[i])
        axs[i].set_ylabel('Amplitude')
        axs[i].legend()

    # 最后一行同时显示pred和target的信号
    axs[3].plot(signals[3][:, 0], label='Predicted ICP', linestyle='-', color='blue')
    axs[3].plot(signals[3][:, 1], label='Target ICP', linestyle='--', color='orange')
    axs[3].set_xlabel('Sample')
    axs[3].set_ylabel('ICP Amplitude')
    axs[3].legend()

    fig.suptitle(suptitle, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Adjust top to make space for suptitle
    plt.show()


if __name__ == "__main__":

    # load the model
    model = Multi_Wav_UNet_SeparableConv(input_nc=1, output_nc=1, ngf=8).double().to(device)
    criterion = torch.nn.L1Loss()
    path_to_save_model = "dl/result/save_model/fold1/ckpt_model_10.pth"
    if os.path.isdir(path_to_save_model):
        checkpoint = torch.load(
            path_to_save_model, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['net'])
    model.eval()

    # load the test data
    root_dir = 'data'
    folders = ['folder1']
    for folder in folders:
        for root, dirs, files in os.walk(os.path.join(root_dir, folder)):
            for file in files:
                if file.endswith('.npy'):
                    file_path = os.path.join(root, file)
                    try:
                        npy_data = np.load(file_path)

                        # 归一化
                        npy_data_min = np.min(npy_data, axis=0)  # 每列的最小值
                        npy_data_max = np.max(npy_data, axis=0)  # 每列的最大值
                        npy_data_range = np.where(npy_data_max - npy_data_min == 0, 1, npy_data_max - npy_data_min)
                        normalized_data = (npy_data - npy_data_min) / npy_data_range

                        _input = normalized_data[:, 1:4]  # 输入信号
                        target = normalized_data[:, 0].reshape(-1, 1)  # 目标信号
                        _input = torch.tensor(_input).double().to(device)
                        _input = _input.unsqueeze(0)
                        pred = model(_input.permute(0, 2, 1)).double()
                        pred = np.squeeze(pred.cpu().detach().numpy(), axis=0).T

                        # 1. 绘制反归一化前的信号
                        signals_before_normalization = [
                            normalized_data[:, 1],  # ABP
                            normalized_data[:, 2],  # PPG
                            normalized_data[:, 3],  # ECG
                            np.column_stack((pred, target))  # Pred ICP and Target ICP
                        ]
                        plot_pred_signals(signals_before_normalization, ['ABP', 'PPG', 'ECG', 'Pred ICP vs Target ICP'],
                                          suptitle=f"{file} Signals Before inverse normalization")
                        print("Before")


                        # 仅绘制pred和target的信号
                        signals = np.column_stack((pred, target))
                        plot_signals(signals, ['Predict ICP', 'Target ICP'], title=f"{file} Predict vs Target ICP")
                        print("Only Pred and Target")

                        # 2. 反归一化
                        pred = pred * npy_data_range[0] + npy_data_min[0]
                        target = target * npy_data_range[0] + npy_data_min[0]

                        # 3. 绘制反归一化后的信号
                        signals_after_normalization = [
                            npy_data[:,1],  # ABP
                            npy_data[:,2],  # PPG
                            npy_data[:,3],  # ECG
                            np.column_stack((pred, target))  # Pred ICP and Target ICP
                        ]
                        plot_pred_signals(signals_after_normalization, ['ABP', 'PPG', 'ECG', 'Pred ICP vs Target ICP'],
                                          suptitle=f"{file} Signals After inverse normalization")
                        print("After")


                        # 仅绘制pred和target的信号
                        signals = np.column_stack((pred, target))
                        plot_signals(signals, ['Predict ICP', 'Target ICP'], title=f"{file} Predict vs Target ICP")
                        print("Only Pred and Target")

                    except Exception as e:
                        print(f"Error reading file {file_path}: {e}")
