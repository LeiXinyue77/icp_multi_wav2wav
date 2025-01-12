import os
import joblib
import numpy as np
import torch
import matplotlib.pyplot as plt
from dl.model.multi_wav_unet.mw_unet_separableConv import Multi_Wav_UNet_SeparableConv

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    # load the model
    model = Multi_Wav_UNet_SeparableConv(input_nc=1, output_nc=1, ngf=8).double().to(device)
    criterion = torch.nn.MSELoss()
    path_to_save_model = "dl/result/save_model/fold1/ckpt_best.pth"
    if os.path.isdir(path_to_save_model):
        checkpoint = torch.load(
            path_to_save_model, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['net'])
    model.eval()

    # load the test data
    root_dir = 'data'
    folders =['folder1']
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

                        _input = normalized_data[:, 1:4]
                        target = normalized_data[:, 0].reshape(-1, 1)
                        _input = torch.tensor(_input).double().to(device)
                        # 增加一个维度
                        _input = _input.unsqueeze(0)
                        pred = model(_input.permute(0, 2, 1)).double()
                        pred = np.squeeze(pred.cpu().detach(), axis=0).T

                        plt.figure(figsize=(10, 6))  # 调整图像大小
                        plt.plot(pred, label='Predicted', color='blue', linestyle='-', linewidth=2)
                        plt.plot(target, label='Target', color='orange', linestyle='--', linewidth=2)
                        plt.title(f"{file}_normalized", fontsize=16)
                        plt.xlabel("Sample", fontsize=14)
                        plt.ylabel("Amplitude", fontsize=14)
                        plt.legend(fontsize=12)
                        plt.grid(alpha=0.5)
                        plt.show()


                        # 反归一化
                        pred = pred * npy_data_range[0] + npy_data_min[0]
                        target = target * npy_data_range[0] + npy_data_min[0]
                        plt.figure(figsize=(10, 6))  # 调整图像大小
                        plt.plot(pred, label='Predicted', color='blue', linestyle='-', linewidth=2)
                        plt.plot(target, label='Target', color='orange', linestyle='--', linewidth=2)
                        plt.title(f"{file}_origin", fontsize=16)
                        plt.xlabel("Sample", fontsize=14)
                        plt.ylabel("Amplitude", fontsize=14)
                        plt.legend(fontsize=12)
                        plt.grid(alpha=0.5)
                        plt.show()

                        print("Prediction and target comparison plot: ", file)

                    except Exception as e:
                        print(f"Error reading file {file_path}: {e}")