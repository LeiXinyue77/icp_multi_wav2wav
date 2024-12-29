import argparse
from torch.utils.data import DataLoader
from train import Train
from helpers import *
import torch
from dl.dataloader import IcpDataset

# 设置 CUDA 设备
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def main(
        epoch: int = 200,
        batch_size: int = 32,
        path_to_save_model="result/save_model",
        path_to_save_loss="result/save_loss",
        device="cuda:0"
):
    # 设置随机数种子
    setup_seed(20)

    # 初始化设备
    device = torch.device(device if torch.cuda.is_available() and "cuda" in device else "cpu")

    all_folders = ["folder1", "folder2", "folder3", "folder4", "folder5"]

    fold = 1
    for val_folder in all_folders:
        # For each fold, the validation folder is different; the rest are training folders
        print(f"=================================== Start Training Fold {fold} ====================================")

        # Prepare training and validation datasets
        train_folder = [folder for folder in all_folders if folder != val_folder]
        print(f"Training folders: {train_folder}, Validation folder: {val_folder}")

        train_dataset = IcpDataset(folders=train_folder, root_dir='data')
        val_dataset = IcpDataset(folders=[val_folder], root_dir='data')
        print(f"Training dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")

        # DataLoader for the fold
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
        print("================================= DataLoader finished successfully =================================")

        # Generate paths for saving models and loss for each fold
        fold_model_path = os.path.join(path_to_save_model, f"fold{fold}")
        fold_loss_path = os.path.join(path_to_save_loss, f"fold{fold}")
        os.makedirs(fold_model_path, exist_ok=True)
        os.makedirs(fold_loss_path, exist_ok=True)

        # Start training for this fold
        Train(train_dl=train_dataloader, val_dl=val_dataloader, EPOCH=epoch,
              path_to_save_model=fold_model_path, path_to_save_loss=fold_loss_path,
              device=device, RESUME=False)

        # Increment fold count
        fold += 1



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=200, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for DataLoader.")
    parser.add_argument("--path_to_save_model", type=str, default="result/save_model",
                        help="Path to save trained model.")
    parser.add_argument("--path_to_save_loss", type=str, default="result/save_loss", help="Path to save loss "
                                                                                            "metrics.")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to run the training on, e.g., 'cuda:0' or 'cpu'.")

    # 解析命令行参数
    args = parser.parse_args()

    # 启动主函数
    main(
        epoch=args.epoch,
        batch_size=args.batch_size,
        path_to_save_model=args.path_to_save_model,
        path_to_save_loss=args.path_to_save_loss,
        device=args.device
    )

    print("=============================== Training completed successfully !!! ===================================")