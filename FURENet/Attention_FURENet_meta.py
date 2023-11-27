import torch
from torch import nn, Tensor
import torch.nn.functional as F

from FURENet.ResidualConv2d import ResidualConv2d  # 导入ResidualConv2d模块
from FURENet.SEBlock import SEBlock  # 导入SEBlock模块
from FURENet.SelfAttention import SelfAttention, Attention_block
from scipy.stats import truncnorm

# from ResidualConv2d import ResidualConv2d  # 导入ResidualConv2d模块
# from SEBlock import SEBlock  # 导入SEBlock模块
# from SelfAttention import SelfAttention, Attention_block

__all__ = ["FURENet"]

class FeedForward(nn.Module):
    def __init__(self, fea, res_ln=False):
        super(FeedForward, self).__init__()
        self.res_ln = res_ln
        self.L = len(fea) - 1
        self.linear = nn.ModuleList([nn.Linear(fea[i], fea[i+1]) for i in range(self.L)])
        self.ln = nn.LayerNorm(fea[self.L], elementwise_affine=False)
        # torch.nn.init.uniform_(self.ln.weight)

    def forward(self, inputs):
        x = inputs
        for i in range(self.L):
            x = self.linear[i](x)
            if i != self.L-1:
                x = F.relu(x)
        if self.res_ln:
            x += inputs
            x = self.ln(x)
        return x

def truncated_normal_(tensor, mean=0.0, std=1.0):
    # PT doesn't have truncated normal.
    # https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/18
    values = truncnorm.rvs(-2, 2, size=tensor.shape)
    values = mean + std * values
    tensor.copy_(torch.from_numpy(values))
    return tensor

def fc_init_(module):
    if hasattr(module, 'weight') and module.weight is not None:
        truncated_normal_(module.weight.data, mean=0.0, std=0.01)
    if hasattr(module, 'bias') and module.bias is not None:
        torch.nn.init.constant_(module.bias.data, 0.0)
    return module

def maml_init_(module):
    if isinstance(module, ResidualConv2d):
        # Initialize the weights of the convolutional layers in self.operate1
        for sub_module in module.operate1:
            if isinstance(sub_module, (nn.Conv2d, nn.ConvTranspose2d)):
                torch.nn.init.xavier_uniform_(sub_module.weight.data, gain=1.0)
        
        # Initialize the weights of the convolutional layers in self.operate2
        for sub_module in module.operate2:
            if isinstance(sub_module, nn.Conv2d):
                torch.nn.init.xavier_uniform_(sub_module.weight.data, gain=1.0)



class FURENet(nn.Module):
    def __init__(self, in_channels: int):
        super(FURENet, self).__init__()
        # Z变量的下采样操作
        self.z_downSample_operate_1_to_2 = ResidualConv2d(in_channels=in_channels, out_channels=64)
        self.z_downSample_operate_2_to_3 = ResidualConv2d(in_channels=64, out_channels=128)
        self.z_downSample_operate_3_to_4 = ResidualConv2d(in_channels=128, out_channels=192)
        self.z_downSample_operate_4_to_5 = ResidualConv2d(in_channels=192, out_channels=256)
        self.z_downSample_operate_5_to_6 = ResidualConv2d(in_channels=256, out_channels=384)
        self.z_downSample_operate_6_to_7 = ResidualConv2d(in_channels=384, out_channels=512)

        # Zdr变量的下采样操作
        self.zdr_downSample_operate_1_to_2 = ResidualConv2d(in_channels=in_channels, out_channels=64)
        self.zdr_downSample_operate_2_to_3 = ResidualConv2d(in_channels=64, out_channels=128)
        self.zdr_downSample_operate_3_to_4 = ResidualConv2d(in_channels=128, out_channels=192)
        self.zdr_downSample_operate_4_to_5 = ResidualConv2d(in_channels=192, out_channels=256)
        self.zdr_downSample_operate_5_to_6 = ResidualConv2d(in_channels=256, out_channels=384)
        self.zdr_downSample_operate_6_to_7 = ResidualConv2d(in_channels=384, out_channels=512)

        # Kdp变量的下采样操作
        self.kdp_downSample_operate_1_to_2 = ResidualConv2d(in_channels=in_channels, out_channels=64)
        self.kdp_downSample_operate_2_to_3 = ResidualConv2d(in_channels=64, out_channels=128)
        self.kdp_downSample_operate_3_to_4 = ResidualConv2d(in_channels=128, out_channels=192)
        self.kdp_downSample_operate_4_to_5 = ResidualConv2d(in_channels=192, out_channels=256)
        self.kdp_downSample_operate_5_to_6 = ResidualConv2d(in_channels=256, out_channels=384)
        self.kdp_downSample_operate_6_to_7 = ResidualConv2d(in_channels=384, out_channels=512)

        # self.se_block = SEBlock(channels=512 * 3)  # 定义SEBlock
        self.self_attention = SelfAttention(in_channels=512 * 3)  # 定义SelfAttention

        # 上采样操作
        self.upSample_7_to_6 = ResidualConv2d(in_channels=512 * 3, out_channels=384)
        self.upSample_6_to_5 = ResidualConv2d(in_channels=384 * 4, out_channels=256)
        self.upSample_5_to_4 = ResidualConv2d(in_channels=256 * 4, out_channels=192)
        self.upSample_4_to_3 = ResidualConv2d(in_channels=192 * 4, out_channels=128)
        self.upSample_3_to_2 = ResidualConv2d(in_channels=128 * 4, out_channels=64)
        self.upSample_2_to_1 = ResidualConv2d(in_channels=64 * 4, out_channels=in_channels)

        self.attention_blocks = nn.ModuleList([
            Attention_block(384, 384, 192),  # 为上采样层7_to_6定义注意力模块
            Attention_block(256, 256, 128),  # 为上采样层6_to_5定义注意力模块
            Attention_block(192, 192, 96),   # 为上采样层5_to_4定义注意力模块
            Attention_block(128, 128, 64),   # 为上采样层4_to_3定义注意力模块
            Attention_block(64, 64, 32)      # 为上采样层3_to_2定义注意力模块
        ])


    def forward(self, z: Tensor, zdr: Tensor, kdp: Tensor, out_len: int = 10):
        # Z变量的下采样过程
        z2 = self.z_downSample_operate_1_to_2(z)
        z3 = self.z_downSample_operate_2_to_3(z2)
        z4 = self.z_downSample_operate_3_to_4(z3)
        z5 = self.z_downSample_operate_4_to_5(z4)
        z6 = self.z_downSample_operate_5_to_6(z5)
        z7 = self.z_downSample_operate_6_to_7(z6)

        # Zdr变量的下采样过程
        zdr2 = self.zdr_downSample_operate_1_to_2(zdr)
        zdr3 = self.zdr_downSample_operate_2_to_3(zdr2)
        zdr4 = self.zdr_downSample_operate_3_to_4(zdr3)
        zdr5 = self.zdr_downSample_operate_4_to_5(zdr4)
        zdr6 = self.zdr_downSample_operate_5_to_6(zdr5)
        zdr7 = self.zdr_downSample_operate_6_to_7(zdr6)

        # Kdp变量的下采样过程
        kdp2 = self.kdp_downSample_operate_1_to_2(kdp)
        kdp3 = self.kdp_downSample_operate_2_to_3(kdp2)
        kdp4 = self.kdp_downSample_operate_3_to_4(kdp3)
        kdp5 = self.kdp_downSample_operate_4_to_5(kdp4)
        kdp6 = self.kdp_downSample_operate_5_to_6(kdp5)
        kdp7 = self.kdp_downSample_operate_6_to_7(kdp6)

        # 在通道维度上连接三个雷达变量的特征图
        concat_7 = torch.cat([zdr7, kdp7, z7], dim=1)
        # upSampleInput = self.se_block(concat_7)  # 应用SEBlock
        upSampleInput = self.self_attention(concat_7)  # 应用SelfAttention

        # 上采样过程
        upSample6 = self.upSample_7_to_6(upSampleInput)
        upSample6 = self.attention_blocks[0](upSample6, upSample6)  # 应用注意力模块
        concat_6 = torch.cat([zdr6, kdp6, z6, upSample6], dim=1)

        upSample5 = self.upSample_6_to_5(concat_6)
        upSample5 = self.attention_blocks[1](upSample5, upSample5)  # 应用注意力模块
        concat_5 = torch.cat([zdr5, kdp5, z5, upSample5], dim=1)

        upSample4 = self.upSample_5_to_4(concat_5)
        upSample4 = self.attention_blocks[2](upSample4, upSample4)  # 应用注意力模块
        concat_4 = torch.cat([zdr4, kdp4, z4, upSample4], dim=1)

        upSample3 = self.upSample_4_to_3(concat_4)
        upSample3 = self.attention_blocks[3](upSample3, upSample3)  # 应用注意力模块
        concat_3 = torch.cat([zdr3, kdp3, z3, upSample3], dim=1)

        upSample2 = self.upSample_3_to_2(concat_3)
        upSample2 = self.attention_blocks[4](upSample2, upSample2)  # 应用注意力模块
        concat_2 = torch.cat([zdr2, kdp2, z2, upSample2], dim=1)

        out = self.upSample_2_to_1(concat_2)  # 最终输出

        return out
    

class FURENetBackbone(nn.Module):
    def __init__(self, in_channels: int):
        super(FURENetBackbone, self).__init__()
        # Z变量的下采样操作
        self.z_downSample_operate_1_to_2 = ResidualConv2d(in_channels=in_channels, out_channels=64)
        self.z_downSample_operate_2_to_3 = ResidualConv2d(in_channels=64, out_channels=128)
        self.z_downSample_operate_3_to_4 = ResidualConv2d(in_channels=128, out_channels=192)
        self.z_downSample_operate_4_to_5 = ResidualConv2d(in_channels=192, out_channels=256)
        self.z_downSample_operate_5_to_6 = ResidualConv2d(in_channels=256, out_channels=384)
        self.z_downSample_operate_6_to_7 = ResidualConv2d(in_channels=384, out_channels=512)

        # Zdr变量的下采样操作
        self.zdr_downSample_operate_1_to_2 = ResidualConv2d(in_channels=in_channels, out_channels=64)
        self.zdr_downSample_operate_2_to_3 = ResidualConv2d(in_channels=64, out_channels=128)
        self.zdr_downSample_operate_3_to_4 = ResidualConv2d(in_channels=128, out_channels=192)
        self.zdr_downSample_operate_4_to_5 = ResidualConv2d(in_channels=192, out_channels=256)
        self.zdr_downSample_operate_5_to_6 = ResidualConv2d(in_channels=256, out_channels=384)
        self.zdr_downSample_operate_6_to_7 = ResidualConv2d(in_channels=384, out_channels=512)

        # Kdp变量的下采样操作
        self.kdp_downSample_operate_1_to_2 = ResidualConv2d(in_channels=in_channels, out_channels=64)
        self.kdp_downSample_operate_2_to_3 = ResidualConv2d(in_channels=64, out_channels=128)
        self.kdp_downSample_operate_3_to_4 = ResidualConv2d(in_channels=128, out_channels=192)
        self.kdp_downSample_operate_4_to_5 = ResidualConv2d(in_channels=192, out_channels=256)
        self.kdp_downSample_operate_5_to_6 = ResidualConv2d(in_channels=256, out_channels=384)
        self.kdp_downSample_operate_6_to_7 = ResidualConv2d(in_channels=384, out_channels=512)

        # self.se_block = SEBlock(channels=512 * 3)  # 定义SEBlock
        self.self_attention = SelfAttention(in_channels=512 * 3)  # 定义SelfAttention

        # 上采样操作
        self.upSample_7_to_6 = ResidualConv2d(in_channels=512 * 3, out_channels=384)
        self.upSample_6_to_5 = ResidualConv2d(in_channels=384 * 4, out_channels=256)
        self.upSample_5_to_4 = ResidualConv2d(in_channels=256 * 4, out_channels=192)
        self.upSample_4_to_3 = ResidualConv2d(in_channels=192 * 4, out_channels=128)
        self.upSample_3_to_2 = ResidualConv2d(in_channels=128 * 4, out_channels=64)
        # self.upSample_2_to_1 = ResidualConv2d(in_channels=64 * 4, out_channels=in_channels)

        self.attention_blocks = nn.ModuleList([
            Attention_block(384, 384, 192),  # 为上采样层7_to_6定义注意力模块
            Attention_block(256, 256, 128),  # 为上采样层6_to_5定义注意力模块
            Attention_block(192, 192, 96),   # 为上采样层5_to_4定义注意力模块
            Attention_block(128, 128, 64),   # 为上采样层4_to_3定义注意力模块
            Attention_block(64, 64, 32)      # 为上采样层3_to_2定义注意力模块
        ])


    def forward(self, z: Tensor, zdr: Tensor, kdp: Tensor, out_len: int = 10):
        # Z变量的下采样过程
        z2 = self.z_downSample_operate_1_to_2(z)
        z3 = self.z_downSample_operate_2_to_3(z2)
        z4 = self.z_downSample_operate_3_to_4(z3)
        z5 = self.z_downSample_operate_4_to_5(z4)
        z6 = self.z_downSample_operate_5_to_6(z5)
        z7 = self.z_downSample_operate_6_to_7(z6)

        # Zdr变量的下采样过程
        zdr2 = self.zdr_downSample_operate_1_to_2(zdr)
        zdr3 = self.zdr_downSample_operate_2_to_3(zdr2)
        zdr4 = self.zdr_downSample_operate_3_to_4(zdr3)
        zdr5 = self.zdr_downSample_operate_4_to_5(zdr4)
        zdr6 = self.zdr_downSample_operate_5_to_6(zdr5)
        zdr7 = self.zdr_downSample_operate_6_to_7(zdr6)

        # Kdp变量的下采样过程
        kdp2 = self.kdp_downSample_operate_1_to_2(kdp)
        kdp3 = self.kdp_downSample_operate_2_to_3(kdp2)
        kdp4 = self.kdp_downSample_operate_3_to_4(kdp3)
        kdp5 = self.kdp_downSample_operate_4_to_5(kdp4)
        kdp6 = self.kdp_downSample_operate_5_to_6(kdp5)
        kdp7 = self.kdp_downSample_operate_6_to_7(kdp6)

        # 在通道维度上连接三个雷达变量的特征图
        concat_7 = torch.cat([zdr7, kdp7, z7], dim=1)
        # upSampleInput = self.se_block(concat_7)  # 应用SEBlock
        upSampleInput = self.self_attention(concat_7)  # 应用SelfAttention

        # 上采样过程
        upSample6 = self.upSample_7_to_6(upSampleInput)
        upSample6 = self.attention_blocks[0](upSample6, upSample6)  # 应用注意力模块
        concat_6 = torch.cat([zdr6, kdp6, z6, upSample6], dim=1)

        upSample5 = self.upSample_6_to_5(concat_6)
        upSample5 = self.attention_blocks[1](upSample5, upSample5)  # 应用注意力模块
        concat_5 = torch.cat([zdr5, kdp5, z5, upSample5], dim=1)

        upSample4 = self.upSample_5_to_4(concat_5)
        upSample4 = self.attention_blocks[2](upSample4, upSample4)  # 应用注意力模块
        concat_4 = torch.cat([zdr4, kdp4, z4, upSample4], dim=1)

        upSample3 = self.upSample_4_to_3(concat_4)
        upSample3 = self.attention_blocks[3](upSample3, upSample3)  # 应用注意力模块
        concat_3 = torch.cat([zdr3, kdp3, z3, upSample3], dim=1)

        upSample2 = self.upSample_3_to_2(concat_3)
        upSample2 = self.attention_blocks[4](upSample2, upSample2)  # 应用注意力模块
        concat_2 = torch.cat([zdr2, kdp2, z2, upSample2], dim=1)

        # out = self.upSample_2_to_1(concat_2)  # 最终输出

        return concat_2
    
class FURENetOutput(nn.Module):
    def __init__(self, in_channels: int):
        super(FURENetOutput, self).__init__()
        self.upSample_2_to_1 = ResidualConv2d(in_channels=64 * 4, out_channels=in_channels)
    def forward(self, concat_2):   
        out = self.upSample_2_to_1(concat_2)  # 最终输出
        return out

class FURENetMeta(torch.nn.Module):
    def __init__(self, in_channels: int):
        super(FURENetMeta, self).__init__()
        self.features = FURENetBackbone(in_channels)
        self.classifier = FURENetOutput(in_channels)
        maml_init_(self.classifier)

    def forward(self, z: Tensor, zdr: Tensor, kdp: Tensor, out_len: int = 10):
        concat_2 = self.features(z, zdr, kdp, out_len)
        out = self.classifier(concat_2)
        return out




if __name__ == '__main__':
    z = torch.ones(3, 10, 256, 256)  # 创建模拟输入数据
    zdr = torch.ones(3, 10, 256, 256)
    kdp = torch.ones(3, 10, 256, 256)
    net = FURENetMeta(10)  # 实例化网络
    r = net(z, zdr, kdp)  # 前向传播
    print(r.shape)  # 打印输出形状
