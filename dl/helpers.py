import os
import random
import shutil
import torch
import numpy as np
import pandas as pd


def log_loss(epoch, train_epoch_loss, val_epoch_loss, path_to_save_loss):

    loss = [epoch, train_epoch_loss, val_epoch_loss]
    data = pd.DataFrame([loss])
    data.to_csv(f'{path_to_save_loss}/loss.csv', mode='a',
                header=False, index=False)


# Remove all files from previous executions and re-run the model.
def clean_directory():
    if os.path.exists('result/save_loss'):
        shutil.rmtree('result/save_loss')
    if os.path.exists('result/save_model'):
        shutil.rmtree('result/save_model')
    # if os.path.exists('save_predictions'):
    #     shutil.rmtree('save_predictions')
    os.mkdir("result/save_loss")
    os.mkdir("result/save_model")
    # os.mkdir("save_predictions")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


