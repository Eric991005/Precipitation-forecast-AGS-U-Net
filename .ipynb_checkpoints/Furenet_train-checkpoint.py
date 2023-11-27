import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, TensorDataset,random_split
import numpy as np
from util.TrainingTemplate import TrainingTemplate
from util.TestingTemplate import TestingTemplate
# from FURENet.Attention_FURENet_meta import FURENetMeta
# from FURENet.FURENet_raw import FURENet
from FURENet.Attention_FURENet_Gaussian import FURENet
import random

# 定义一个函数来计算每个像素的权重
def compute_weights(x):
    weights = torch.ones_like(x)
    weights[x >= 2] = 2
    weights[x >= 5] = 5
    weights[x >= 10] = 10
    weights[x >= 30] = 50
    return weights

# 定义一个自定义的B-MSE损失函数类
class BMSELoss(nn.Module):
    def __init__(self):
        super(BMSELoss, self).__init__()

    def forward(self, x, target):
        weights = compute_weights(target)  # 计算权重
        loss = torch.mean(weights * (x - target) ** 2)  # 计算B-MSE损失
        return loss

class FURENetTraining(TrainingTemplate):
    def check_data(self, data):
        # 重写此方法以处理三个输入张量
        z, zdr, kdp, labels = data
        z = z.to(self.device)
        zdr = zdr.to(self.device)
        kdp = kdp.to(self.device)
        labels = labels.to(self.device)
        return (z, zdr, kdp), labels

    def forward(self, inputs):
        # 重写此方法以处理三个输入张量
        z, zdr, kdp = inputs
        return self.model(z, zdr, kdp)

class FURENetTesting(TestingTemplate):
    def check_data(self, data):
        # 重写此方法以处理三个输入张量
        z, zdr, kdp, labels = data
        z = z.to(self.device)
        zdr = zdr.to(self.device)
        kdp = kdp.to(self.device)
        labels = labels.to(self.device)
        return (z, zdr, kdp), labels

    def forward(self, inputs, out_len: int = 10):
        # 重写此方法以处理三个输入张量
        z, zdr, kdp = inputs
        return self.model(z, zdr, kdp, out_len=out_len)

class MyDataset(Dataset):
    def __init__(self, dbz, kdp, zdr, y, norm_param):
        self.dbz = self.normalize_data(dbz, norm_param['dBZ'])
        self.kdp = self.normalize_data(kdp, norm_param['KDP'])
        self.zdr = self.normalize_data(zdr, norm_param['ZDR'])
        self.y = self.normalize_data(y, norm_param['dBZ'])  # 标签的归一化范围和dBZ相同

    def normalize_data(self, data, range_param):
        mmin, mmax = range_param
        normalized_data = (data - mmin) / (mmax - mmin)
        return torch.tensor(normalized_data, dtype=torch.float32)
    
    def __len__(self):
        return len(self.dbz)

    def __getitem__(self, idx):
        return self.dbz[idx], self.kdp[idx], self.zdr[idx], self.y[idx]

# 归一化参数
norm_param = {
    'dBZ': [0, 65],
    'ZDR': [-1, 5],
    'KDP': [-1, 6],
    # 'KDP': [0, 350], ####降雨
    'y': [0, 500]
}

# 加载你的数据
dbz = np.load('/root/autodl-tmp/competition/data/all_sample/dbz_sample.npz')['data']
# dbz = np.clip(dbz, 0, None)
kdp = np.load('/root/autodl-tmp/competition/data/all_sample/kdp_sample.npz')['data']
# kdp = np.load('/root/autodl-tmp/competition/data/all_sample/now_rain_sample.npz')['data']
# kdp = dbz**1.4
zdr = np.load('/root/autodl-tmp/competition/data/all_sample/zdr_sample.npz')['data']
y = np.load('/root/autodl-tmp/competition/data/all_sample/y_sample.npz')['data']
# y = np.load('/root/autodl-tmp/competition/data/all_sample/rain_y_sample.npz')['data']

# 创建数据集实例
dataset = MyDataset(dbz, kdp, zdr, y, norm_param)

# 确定训练集和测试集的大小
train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size

# 创建一个随机数生成器并设置随机种子
generator = torch.Generator().manual_seed(42)

# 使用随机数生成器拆分数据集
train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)

# 创建数据加载器实例
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = FURENetMeta(10).to(device) ### Meta

model = FURENet(10).to(device) ### Gaussian

# criterion = torch.nn.CrossEntropyLoss().to(device)
# criterion = torch.nn.MSELoss().to(device)
criterion = BMSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



# 创建训练模板实例
training_template = FURENetTraining(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    criterion=criterion,
    optimizer=optimizer,
    max_epochs=50,
    to_save='/root/autodl-tmp/competition/cpt/FURENet_Gaussian'
)

# 运行训练
training_template.run()

# 创建测试模板实例
testing_template = FURENetTesting(
    model=model,
    test_loader=test_loader,
    to_save='/root/autodl-tmp/competition/cpt/FURENet_Gaussian'
)

# 运行测试
ground_truth, prediction = testing_template.run()
