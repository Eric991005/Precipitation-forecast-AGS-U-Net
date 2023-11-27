import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np


__all__ = ["GaussianSmoothing"]


class GaussianSmoothing(nn.Module):
    def __init__(self, channels, kernel_size, sigma):
        super(GaussianSmoothing, self).__init__()
        self.channels = channels
        # 创建高斯核
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)
        mean = (kernel_size - 1)/2.
        variance = sigma**2.
        gaussian_kernel = (1./(2.*np.pi*variance)) * \
                          torch.exp(
                              -torch.sum((xy_grid - mean)**2., dim=-1) / \
                              (2*variance)
                          )
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        
        # 确保高斯核是一个权重矩阵
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
        
        self.register_buffer('weight', gaussian_kernel)
        self.groups = channels
        
    def forward(self, x):
        """
        对输入图像应用高斯平滑。
        """
        padding = (self.weight.shape[2] - 1) // 2  # 计算所需的填充量
        return F.conv2d(x, weight=self.weight, padding=padding, groups=self.channels)  # 在conv2d中添加padding参数

# 定义高斯平滑层
smoothing_layer = GaussianSmoothing(channels=3, kernel_size=5, sigma=1)
