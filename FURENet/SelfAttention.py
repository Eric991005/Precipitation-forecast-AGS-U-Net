import torch
from torch import nn, Tensor

__all__ = ["SelfAttention"]

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        query = self.query_conv(x)
        key = self.key_conv(x)
        value = self.value_conv(x)
        
        query = query.view(query.size(0), query.size(1), -1).permute(0, 2, 1)
        key = key.view(key.size(0), key.size(1), -1).permute(0, 2, 1)
        value = value.view(value.size(0), value.size(1), -1)

        energy = torch.bmm(query, key.permute(0, 2, 1))
        attention = self.softmax(energy)
        out = torch.bmm(value, attention.permute(0, 2, 1)).view(value.size(0), value.size(1), x.size(2), x.size(3))
        
        out = out + x
        return out

class AttentionDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionDecoder, self).__init__()
        self.self_attention = SelfAttention(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        attention_out = self.self_attention(x)
        conv_out = self.conv(attention_out)
        return conv_out
    
class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g) #1x512x64x64->conv(512，256)/B.N.->1x256x64x64
        x1 = self.W_x(x) #1x512x64x64->conv(512，256)/B.N.->1x256x64x64
        psi = self.relu(g1+x1)#1x256x64x64di
        psi = self.psi(psi)#得到权重矩阵  1x256x64x64 -> 1x1x64x64 ->sigmoid 结果到（0，1）

        return x*psi #与low-level feature相乘，将权重矩阵赋值进去