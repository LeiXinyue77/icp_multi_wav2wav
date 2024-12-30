import logging

from dl.model.idv_net.IDV_NET import IVD_Net_asym
from helpers import *

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
                    datefmt="[%Y-%m-%d %H:%M:%S]")
logger = logging.getLogger(__name__)


def Train(train_dl, val_dl, EPOCH, path_to_save_model, path_to_save_loss, device, RESUME):

    start_epoch = -1
    device = torch.device(device)
    model = IVD_Net_asym(input_nc=1, output_nc=1, ngf=8).double().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()
    min_val_loss = float('inf')

    # 断点续训 加载模型
    if RESUME:
        path_checkpoint = f"{path_to_save_model}/ckpt_best.pth"
        checkpoint = torch.load(path_checkpoint)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        min_val_loss = checkpoint['min_val_loss']

    for epoch in range(start_epoch+1, EPOCH):
        train_loss = 0
        model.train()
        for info, _input, target in train_dl:  # for each data set

            # Shape of _input : [batch_size, n_samples, n_features]
            # Desired input for model: [batch_size, n_features, n_samples]
            optimizer.zero_grad()
            src = _input.permute(0, 2, 1)  # torch.Size([batch_size, n_features, n_samples])
            tgt = target.permute(0, 2, 1)  # torch.Size([batch_size,n_features,n_samples])
            # torch.Size([batch_size,n_features,n_samples])
            pred = model(src).double()
            loss = criterion(pred, tgt)
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item()
        train_loss /= len(train_dl)

        val_loss = 0
        with torch.no_grad():
            model.eval()
            for info, _input, target in val_dl:
                src = _input.permute(0, 2, 1) # torch.Size([batch_size, n_features, n_samples])
                tgt = target.permute(0, 2, 1) # torch.Size([batch_size, n_features, n_samples])
                # torch.Size([batch_size, n_features, n_samples])
                pred = model(src).double()
                loss = criterion(pred, tgt)
                val_loss += loss.detach().item()
            val_loss /= len(val_dl)

        logger.info(
            f"Epoch: {epoch}, Training loss: {train_loss}, Validation loss: {val_loss}")
        log_loss(epoch, train_loss, val_loss, path_to_save_loss)

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
            print("save best model")
            torch.save(checkpoint, f'{path_to_save_model}/ckpt_best.pth')

        if (epoch+1) % 10 == 0:
            checkpoint = {
                "net": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss
            }
            if not os.path.isdir(path_to_save_model):
                os.mkdir(path_to_save_model)
            print("save model per 10 epochs")
            torch.save(
                checkpoint, f'{path_to_save_model}/ckpt_model_{epoch}.pth')
