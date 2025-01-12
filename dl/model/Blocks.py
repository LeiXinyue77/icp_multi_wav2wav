"""
@Project ：
@File    ：dl/model/Blocks.py
@Author  ：Lei Xinyue
@Date    ：2025/01/11 21:14
@Description: 基础模块
"""
import torch.nn as nn


def depthwise_separable_conv(in_dim, out_dim, act_fn, kernel_size=3, stride=1, padding=0, dilation=1):
    return nn.Sequential(
        nn.Conv1d(in_dim, in_dim, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_dim),
        nn.Conv1d(in_dim, out_dim, kernel_size=1),
        nn.BatchNorm1d(out_dim),
        act_fn
    )


def depthwise_separable_deconv(in_dim, out_dim, act_fn, kernel_size=2, stride=2, padding=0, output_padding=0):
    return nn.Sequential(
        nn.ConvTranspose1d(in_dim, in_dim, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=in_dim),
        nn.Conv1d(in_dim, out_dim, kernel_size=1),
        nn.BatchNorm1d(out_dim),
        act_fn,
    )


def conv_block(in_dim, out_dim, act_fn, kernel_size=3, stride=1, padding=0, dilation=1):
    model = nn.Sequential(
        nn.Conv1d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
        nn.BatchNorm1d(out_dim),
        act_fn,
    )
    return model


def conv_decod_block(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.ConvTranspose1d(in_dim, out_dim, kernel_size=2, stride=2, padding=0, output_padding=0),
        nn.BatchNorm1d(out_dim),
        act_fn,
    )
    return model


def maxpool():
    pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
    return pool



def conv_block_Asym_Inception(in_dim, out_dim, act_fn, kernel_size=3, stride=1, padding=1, dilation=1):
    model = nn.Sequential(
        nn.Conv1d(in_dim, out_dim, kernel_size=kernel_size, padding=padding, dilation=dilation),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(),
        nn.Conv1d(out_dim, out_dim, kernel_size=kernel_size, padding=padding, dilation=dilation),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(),
    )
    return model