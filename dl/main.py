import argparse
from torch.utils.data import DataLoader, random_split
from train import Train
from helpers import *
import torch
from dl.dataloader import IcpDataset

# 设置 CUDA 设备
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.backends.cudnn.deterministic = True  # 确保每次运行产生相同的结果
torch.backends.cudnn.benchmark = False    # 禁用 cuDNN 自动优化，确保确定性


def main(
        epoch: int = 200,
        batch_size: int = 32,
        path_to_save_model="result/save_model",
        path_to_save_loss="result/save_loss",
        device="cuda:0",
        start_fold=1,
):
    # 设置随机数种子
    setup_seed(22)
    # 初始化设备
    device = torch.device(device if torch.cuda.is_available() and "cuda" in device else "cpu")
    print(f"=================================== Device: {device} ================================================")
    all_folders = ["folder1", "folder2", "folder3", "folder4", "folder5"]
    fold = start_fold
    for test_folder in all_folders:
        # For each fold, the validation folder is different; the rest are training folders
        print(f"=================================== Start Training Fold {fold} "
              f"=========================================")

        # Prepare training and validation datasets
        train_folder = [folder for folder in all_folders if folder != test_folder]
        print(f"Training folders: {train_folder}, test folder: {test_folder}")

        dataset = IcpDataset(folders=train_folder, root_dir='data',device=device)
        print(f"Dataset size: {len(dataset)}")

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        print(f"Training dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")

        # DataLoader for the fold
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
        print("================================= DataLoader finished successfully =================================")

        # Generate paths for saving models and loss for each fold
        fold_model_path = os.path.join(path_to_save_model, f"fold{fold}")
        fold_loss_path = os.path.join(path_to_save_loss, f"fold{fold}")

        # Start training for this fold
        # resume = fold == 1    # Resume training for the first fold
        Train(train_dl=train_dataloader, val_dl=val_dataloader, train_epoch=epoch,
              path_to_save_model=fold_model_path, path_to_save_loss=fold_loss_path,
              device=device, resume=True)

        # Print fold completion
        print(f"======================= Training Fold {fold} completed successfully !!! ===========================")

        # Increment fold count
        fold += 1

        if fold > 1:
            break


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
    parser.add_argument("--start_fold", type=int, default=1, help="Start fold number for training.")

    # 解析命令行参数
    args = parser.parse_args()

    # 启动主函数
    main(
        epoch=args.epoch,
        batch_size=args.batch_size,
        path_to_save_model=args.path_to_save_model,
        path_to_save_loss=args.path_to_save_loss,
        device=args.device,
        start_fold=args.start_fold
    )

    print("=============================== Training completed successfully !!! ===================================")
