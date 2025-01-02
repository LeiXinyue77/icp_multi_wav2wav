import os
import joblib
import numpy as np
import torch
from dl.model.idv_net.IDV_NET import IVD_Net_asym
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    # load the model
    model = IVD_Net_asym(input_nc=1, output_nc=1, ngf=8).double().to(device)
    criterion = torch.nn.MSELoss()
    path_to_save_model = "result/save_model/IDV-NET-fold1"
    if os.path.isdir(path_to_save_model):
        checkpoint = torch.load(
            f'{path_to_save_model}/ckpt_model_9.pth', map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['net'])
    model.eval()

    # load the test data
    root_dir = 'data'
    folders =['folder3']
    for folder in folders:
        for root, dirs, files in os.walk(os.path.join(root_dir, folder)):
            for file in files:
                if file.endswith('.npy'):
                    file_path = os.path.join(root, file)
                    try:
                        npy_data = np.load(file_path)
                        # 归一化
                        _input = npy_data[:, 1:4]
                        target = npy_data[:, 0].reshape(-1, 1)
                        # laod the scaler
                        input_scaler = joblib.load(
                            "result/save_scaler/fold1/input_scaler.pkl")
                        target_scaler = joblib.load(
                            "result/save_scaler/fold1/target_scaler.pkl")
                        _input = input_scaler.transform(_input)
                        _input = np.expand_dims(_input, axis=0)
                        _input = torch.tensor(_input).double().to(device)
                        # predict
                        pred = model(_input.permute(0, 2, 1)).double()
                        pred = np.squeeze(pred.cpu().detach(), axis=0).T
                        target = target_scaler.transform(target)

                        plt.figure(figsize=(10, 6))  # 调整图像大小
                        plt.plot(pred, label='Predicted', color='blue', linestyle='-', linewidth=2)
                        plt.plot(target, label='Target', color='orange', linestyle='--', linewidth=2)
                        plt.title(f"{file}", fontsize=16)
                        plt.xlabel("Sample", fontsize=14)
                        plt.ylabel("Amplitude", fontsize=14)
                        plt.legend(fontsize=12)
                        plt.grid(alpha=0.5)
                        plt.show()

                        pred = target_scaler.inverse_transform(pred)
                        target = target_scaler.inverse_transform(target)
                        plt.figure(figsize=(10, 6))  # 调整图像大小
                        plt.plot(pred, label='Predicted', color='blue', linestyle='-', linewidth=2)
                        plt.plot(target, label='Target', color='orange', linestyle='--', linewidth=2)
                        plt.title(f"{file}", fontsize=16)
                        plt.xlabel("Sample", fontsize=14)
                        plt.ylabel("Amplitude", fontsize=14)
                        plt.legend(fontsize=12)
                        plt.grid(alpha=0.5)
                        plt.show()

                        print("Prediction and target comparison plot generated successfully")

                    except Exception as e:
                        print(f"Error reading file {file_path}: {e}")