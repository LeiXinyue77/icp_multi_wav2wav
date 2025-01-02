import logging
from tqdm import tqdm
import torch
from dl.model.attention_u_net.AM_UNET import Attention_Multi_UNet
from helpers import *  # 假设此处已经包含 log_loss 函数


def Train(train_dl, val_dl, EPOCH, path_to_save_model, path_to_save_loss, val_folder, device, RESUME):
    # 配置日志记录
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
                        datefmt="[%Y-%m-%d %H:%M:%S]",
                        filename=f"result/train_log/training_log_{val_folder}.txt",
                        filemode="a")
    logger = logging.getLogger(__name__)

    # 初始化模型和优化器
    start_epoch = -1
    device = torch.device(device)
    model = Attention_Multi_UNet(input_nc=1, output_nc=1, ngf=8).double().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()
    min_val_loss = float('inf')

    # 断点续训，加载模型
    if RESUME:
        path_checkpoint = f"{path_to_save_model}/ckpt_best.pth"
        checkpoint = torch.load(path_checkpoint)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        min_val_loss = checkpoint['min_val_loss']

    # 训练循环
    for epoch in range(start_epoch + 1, EPOCH):
        train_loss = 0
        model.train()

        # 训练进度条
        train_bar = tqdm(enumerate(train_dl), total=len(train_dl), desc=f"Epoch {epoch + 1}/{EPOCH} [Training]", dynamic_ncols=True)

        for info, _input, target in train_dl:  # for each data set
            optimizer.zero_grad()
            src = _input.permute(0, 2, 1)  # torch.Size([batch_size, n_features, n_samples])
            tgt = target.permute(0, 2, 1)  # torch.Size([batch_size,n_features,n_samples])
            pred = model(src).double()
            loss = criterion(pred, tgt)
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item()

            # 更新训练进度条
            train_bar.set_postfix(loss=f"{loss.detach().item():.4f}", refresh=True)
            train_bar.update(1)  # 显式更新进度条

        # 计算平均训练损失
        train_loss /= len(train_dl)

        # 验证阶段
        val_loss = 0
        model.eval()
        val_bar = tqdm(enumerate(val_dl), total=len(val_dl), desc=f"Epoch {epoch + 1}/{EPOCH} [Validation]", dynamic_ncols=True)

        with torch.no_grad():
            for info, _input, target in val_dl:
                src = _input.permute(0, 2, 1)  # torch.Size([batch_size, n_features, n_samples])
                tgt = target.permute(0, 2, 1)  # torch.Size([batch_size, n_features, n_samples])
                pred = model(src).double()
                loss = criterion(pred, tgt)
                val_loss += loss.detach().item()

                # 更新验证进度条
                val_bar.set_postfix(loss=f"{loss.detach().item():.4f}", refresh=True)
                val_bar.update(1)  # 显式更新进度条

            # 计算平均验证损失
            val_loss /= len(val_dl)

        # 记录训练和验证损失
        logger.info(f"Epoch: {epoch}, Training loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}")
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
        if (epoch + 1) % 10 == 0:
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
