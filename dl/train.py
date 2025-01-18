import logging
import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from dl.loss import CombinedWaveLoss
from dl.model.multi_wav_unet.MW_UNET import Multi_Wav_UNet
from dl.model.multi_wav_unet.mw_unet_separableConv import Multi_Wav_UNet_SeparableConv
from dl.model.wav_unet.W2W_UNET import UNet
from helpers import *


def Train(train_dl, val_dl, train_epoch, path_to_save_model, path_to_save_loss, device, resume):
    # 配置日志记录
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
                        datefmt="[%Y-%m-%d %H:%M:%S]",
                        filename=f"result/train_log/training_log.txt",
                        filemode="a")
    logger = logging.getLogger(__name__)

    # 初始化模型和优化器
    start_epoch = 1
    device = torch.device(device)
    model = Multi_Wav_UNet(input_nc=1, output_nc=1, ngf=8).double().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    criterion = CombinedWaveLoss(alpha=0.5)  # alpha=0.5 表示值和形状各占 50%
    mse_criterion = torch.nn.MSELoss()  # MSE 损失c
    min_val_loss = float('inf')

    # 断点续训，加载模型
    if resume:
        path_checkpoint = f"{path_to_save_model}/ckpt_best.pth"
        checkpoint = torch.load(path_checkpoint)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        min_val_loss = checkpoint['min_val_loss']

    # 训练循环
    for epoch in range(start_epoch, train_epoch + 1):
        train_loss, train_mse, train_cos_sim = 0, 0, 0
        model.train()
        # 训练进度条
        with tqdm(total=len(train_dl), desc=f"Epoch {epoch}/{train_epoch} [Training]", leave=True, unit="it",
                  unit_scale=False, ncols=200) as train_bar:
            for batch_idx, (info, _input, target) in enumerate(train_dl):  # for each data set
                optimizer.zero_grad()
                src = _input.permute(0, 2, 1)  # torch.Size([batch_size, n_features, n_samples])
                tgt = target.permute(0, 2, 1)  # torch.Size([batch_size, n_features, n_samples])
                pred = model(src).double()

                loss = criterion(pred, tgt)
                mse = mse_criterion(pred, tgt)
                cos_sim = torch.mean(F.cosine_similarity(pred, tgt, dim=-1))

                loss.backward()
                optimizer.step()

                # 累加损失
                train_loss += loss.detach().item()
                train_mse += mse.detach().item()
                train_cos_sim += cos_sim.detach().item()

                avg_loss = train_loss / (batch_idx + 1)  # 当前平均损失
                avg_mse = train_mse / (batch_idx + 1)
                avg_cos_sim = train_cos_sim / (batch_idx + 1)

                train_bar.set_postfix(loss=f"{avg_loss:.4f}", mse=f"{avg_mse:.4f}", cos_sim=f"{avg_cos_sim:.4f}")
                train_bar.update(1)

        # 计算平均训练损失
        train_loss /= len(train_dl)
        train_mse /= len(train_dl)
        train_cos_sim /= len(train_dl)

        # 验证阶段
        val_loss, val_mse, val_cos_sim = 0, 0, 0
        model.eval()
        with tqdm(total=len(val_dl), desc=f"Epoch {epoch}/{train_epoch} [Validation]", leave=True, unit="it",
                  unit_scale=False, ncols=200) as val_bar:
            with torch.no_grad():
                for batch_idx, (info, _input, target) in enumerate(val_dl):
                    src = _input.permute(0, 2, 1)  # torch.Size([batch_size, n_features, n_samples])
                    tgt = target.permute(0, 2, 1)  # torch.Size([batch_size, n_features, n_samples])
                    pred = model(src).double()

                    loss = criterion(pred, tgt)
                    mse = mse_criterion(pred, tgt)
                    cos_sim = torch.mean(F.cosine_similarity(pred, tgt, dim=-1))

                    # 累加损失
                    val_loss += loss.detach().item()
                    val_mse += mse.detach().item()
                    val_cos_sim += cos_sim.detach().item()

                    avg_loss = val_loss / (batch_idx + 1)  # 当前平均损失
                    avg_mse = val_mse / (batch_idx + 1)
                    avg_cos_sim = val_cos_sim / (batch_idx + 1)

                    val_bar.set_postfix(loss=f"{avg_loss:.4f}", mse=f"{avg_mse:.4f}", cos_sim=f"{avg_cos_sim:.4f}")
                    val_bar.update(1)

            val_loss /= len(val_dl)
            val_mse /= len(val_dl)
            val_cos_sim /= len(val_dl)

        # 训练日志
        logger.info(
            f"Epoch: {epoch}, Train_loss: {train_loss:.4f}, Train_MSE: {train_mse:.4f}, Train_CosSim: {train_cos_sim:.4f}, "
            f"Val_loss: {val_loss:.4f}, Val_MSE: {val_mse:.4f}, Val_CosSim: {val_cos_sim:.4f}, "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # 如果当前验证损失较小，则保存最优模型
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            checkpoint = {
                "net": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch,
                "min_val_loss": min_val_loss
            }
            if not os.path.isdir(path_to_save_model):
                os.mkdir(path_to_save_model)
            logger.info(f"Saving best model at epoch {epoch} with validation loss {val_loss:.4f}")
            torch.save(checkpoint, f'{path_to_save_model}/ckpt_best.pth')

        # 每10个epoch保存一次模型
        if epoch % 10 == 0:
            checkpoint = {
                "net": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss
            }
            if not os.path.isdir(path_to_save_model):
                os.mkdir(path_to_save_model)
            logger.info(f"Saving model at epoch {epoch}")
            torch.save(checkpoint, f'{path_to_save_model}/ckpt_model_{epoch}.pth')
