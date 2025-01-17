import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedWaveLoss(nn.Module):
    def __init__(self, alpha=0.5):
        """
        复合波形损失函数
        Args:
            alpha (float): 权重参数，0 <= alpha <= 1。控制值损失和形状相似度的占比。
        """
        super(CombinedWaveLoss, self).__init__()
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()  # 均方误差用于值差异的衡量

    def forward(self, pred, target):
        """
        Args:
            pred (Tensor): 模型预测的波形，形状为 (batch_size, n_features, seq_len)
            target (Tensor): 目标波形，形状为 (batch_size, n_features, seq_len)
        Returns:
            loss (Tensor): 复合损失值
        """
        # 1. 波形值的损失（MSE）
        value_loss = self.mse_loss(pred, target)

        # 2. 波形形状的相似度（归一化互相关，或余弦相似度）
        # 归一化预测和目标
        pred_norm = F.normalize(pred, p=2, dim=-1)  # L2 归一化
        target_norm = F.normalize(target, p=2, dim=-1)  # L2 归一化

        # 计算余弦相似度
        cosine_similarity = F.cosine_similarity(pred_norm, target_norm, dim=-1)  # (batch_size, n_features)

        # 取相似度的均值作为形状相似度
        shape_similarity = cosine_similarity.mean()

        # 3. 复合损失
        loss = self.alpha * value_loss + (1 - self.alpha) * (1 - shape_similarity)
        return loss
