#PredRNN_Seq2Seq
from PredRNN.PredRNN_Model import PredRNN
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader,random_split
import os
import time
import torch

#
#
# input1= np.load('/Users/gaoyihan/Desktop/F题/dataset/all_sample/zdr_sample.npz')
# input2= np.load('/Users/gaoyihan/Desktop/F题/dataset/all_sample/dbz_sample.npz')

# 假设 tensor1 和 tensor2 是你的两个四维张量
# tensor1 = torch.tensor(input1['data'])  # 示例张量
# tensor2 = torch(input2['data'])  # 示例张量

# 在第二个和第三个维度上增加一个维度
# tensor1_expanded = tensor1.unsqueeze(2).unsqueeze(3)
# tensor2_expanded = tensor2.unsqueeze(2).unsqueeze(3)
# # 现在 tensor1_expanded 和 tensor2_expanded 的维度是 [1000,10,2,256,256]
# # 在第二个和第三个维度上拼接张量
# input = torch.cat((tensor1_expanded, tensor2_expanded), dim=2)

# 现在 concatenated_tensor 的维度是 [1000,10,2,256,256]





# Batch_size , time_step, channels, hight/width, width/hight
# Batch_size , time_step, channels, hight/width, width/hight



class PredRNN_enc(nn.Module):
    def __init__(self):
        super(PredRNN_enc, self).__init__()
        self.pred1_enc=PredRNN(input_size=(256,256),
                input_dim=2,  #changed
                hidden_dim=[7, 1],
                hidden_dim_m=[7, 7],
                kernel_size=(7, 7),
                num_layers=2,
                batch_first=True,
                bias=True)
    def forward(self,enc_input, out_len=None):
        _, layer_h_c, all_time_h_m, _ = self.pred1_enc(enc_input)
        return layer_h_c, all_time_h_m

class PredRNN_dec(nn.Module):
    def __init__(self):
        super(PredRNN_dec, self).__init__()
        self.pred1_dec=PredRNN(input_size=(256,256),
                input_dim=2, #changed
                hidden_dim=[7, 1],
                hidden_dim_m=[7, 7],
                kernel_size=(7, 7),
                num_layers=2,
                batch_first=True,
                bias=True)
        self.relu = nn.ReLU()
    def forward(self,dec_input,enc_hidden,enc_h_m, out_len=None):
        out, layer_h_c, last_h_m, _ = self.pred1_dec(dec_input,enc_hidden,enc_h_m)
        out = self.relu(out)
        return out, layer_h_c, last_h_m

###############################################
