import logging
import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from dl.model.multi_wav_unet.mw_unet_separableConv import Multi_Wav_UNet_SeparableConv
from dl.model.wav_unet.W2W_UNET import UNet
from helpers import *



class LogCosineLoss(torch.nn.Module):
    def __init__(self):
        super(LogCosineLoss, self).__init__()

    def forward(self, pred, target):
        # 计算预测和目标之间的余弦相似度
        cos_sim = F.cosine_similarity(pred, target, dim=-1)  # 计算余弦相似度
        # 防止 log(0) 出现负无穷的问题
        cos_sim = torch.clamp(cos_sim, min=1e-6)
        # 对余弦相似度取对数
        log_cos_sim = torch.log(cos_sim)
        # 返回对数余弦相似度的负值（因为通常优化目标是最小化损失）
        return -torch.mean(log_cos_sim)


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
    model = UNet(ngf=8).double().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # CosineAnnealingWarmRestarts 学习率调度器
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    # 使用Log-Cosine Loss
    # criterion = LogCosineLoss()
    # 使用 MAE Loss
    criterion = torch.nn.L1Loss()
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
        train_loss = 0
        model.train()
        # 训练进度条
        with tqdm(total=len(train_dl), desc=f"Epoch {epoch}/{train_epoch} [Training]", leave=True, unit="it", unit_scale=False) as train_bar:
            for batch_idx, (info, _input, target) in enumerate(train_dl):  # for each data set
                optimizer.zero_grad()
                src = _input.permute(0, 2, 1)  # torch.Size([batch_size, n_features, n_samples])
                tgt = target.permute(0, 2, 1)  # torch.Size([batch_size, n_features, n_samples])
                pred = model(src).double()
                loss = criterion(pred, tgt)
                loss.backward()
                optimizer.step()

                # 累加损失
                train_loss += loss.detach().item()
                avg_loss = train_loss / (batch_idx + 1)  # 当前平均损失
                train_bar.set_postfix(avg_loss=f"{avg_loss:.4f}")
                train_bar.update(1)

        # 计算平均训练损失
        train_loss /= len(train_dl)

        # 验证阶段
        val_loss = 0
        model.eval()
        with tqdm(total=len(val_dl), desc=f"Epoch {epoch}/{train_epoch} [Validation]", leave=True, unit="it", unit_scale=False) as val_bar:
            with torch.no_grad():
                for batch_idx, (info, _input, target) in enumerate(val_dl):
                    src = _input.permute(0, 2, 1)  # torch.Size([batch_size, n_features, n_samples])
                    tgt = target.permute(0, 2, 1)  # torch.Size([batch_size, n_features, n_samples])
                    pred = model(src).double()
                    loss = criterion(pred, tgt)

                    # 累加损失
                    val_loss += loss.detach().item()

                    avg_loss = val_loss / (batch_idx + 1)  # 当前平均损失
                    val_bar.set_postfix(avg_loss=f"{avg_loss:.4f}")
                    val_bar.update(1)

            val_loss /= len(val_dl)

        # 更新学习率（CosineAnnealingWarmRestarts 会自动调整）
        # scheduler.step(epoch)

        # 记录训练和验证损失
        logger.info(f"Epoch: {epoch}, Training loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        log_loss(epoch, train_loss, val_loss, path_to_save_loss)

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
