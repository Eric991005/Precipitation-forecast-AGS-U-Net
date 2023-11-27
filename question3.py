import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, TensorDataset,random_split
import numpy as np
from util.TrainingTemplate import TrainingTemplate
from util.TestingTemplate import TestingTemplate
from FURENet.Attention_FURENet import FURENet
import random

class MyDataset(Dataset):
    def __init__(self, dbz, y, norm_param):
        self.dbz = self.normalize_data(dbz, norm_param['dBZ'])
        self.y = y  # 标签的归一化范围和dBZ相同

    def normalize_data(self, data, range_param):
        mmin, mmax = range_param
        normalized_data = (data - mmin) / (mmax - mmin)
        return torch.tensor(normalized_data, dtype=torch.float32)
    
    def __len__(self):
        return len(self.dbz)

    def __getitem__(self, idx):
        return self.dbz[idx], self.y[idx]

# 归一化参数
norm_param = {
    'dBZ': [0, 65],
    'ZDR': [-1, 5],
    'KDP': [-1, 6]
}

# 加载你的数据
dbz = np.load('/root/autodl-tmp/competition/data/all_sample/dbz_sample.npz')['data']
y = np.load('/root/autodl-tmp/competition/data/all_sample/rain_y_sample.npz')['data']

# 创建数据集实例
dataset = MyDataset(dbz, y, norm_param)

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

# class RainfallEstimationModel(nn.Module):
#     def __init__(self):
#         super(RainfallEstimationModel, self).__init__()
#         self.a = nn.Parameter(torch.tensor(0.1))  # 初始化 a 为 1.0
#         self.b = nn.Parameter(torch.tensor(0.1))  # 初始化 b 为 1.0

#     def forward(self, Z_H, out_len: int = 10):
#         R = self.a * (Z_H ** self.b)
#         return R

# class RainfallEstimationModelTraining(TrainingTemplate):
#     def check_data(self, data):
#         # 重写此方法以处理三个输入张量
#         z, labels = data
#         z = z.to(self.device)
#         labels = labels.to(self.device)
#         return z, labels

#     def forward(self, inputs):
#         # 重写此方法以处理三个输入张量
#         z = inputs
#         return self.model(z)

# class RainfallEstimationModelTesting(TestingTemplate):
#     def check_data(self, data):
#         # 重写此方法以处理三个输入张量
#         z, labels = data
#         z = z.to(self.device)
#         labels = labels.to(self.device)
#         return z, labels

#     def forward(self, inputs, out_len: int = 10):
#         # 重写此方法以处理三个输入张量
#         z = inputs
#         return self.model(z, out_len=out_len)
    

# model = RainfallEstimationModel().to(device)

# # criterion = torch.nn.CrossEntropyLoss().to(device)
# criterion = torch.nn.MSELoss().to(device)
# # criterion = BMSELoss().to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)



# # 创建训练模板实例
# training_template = RainfallEstimationModelTraining(
#     model=model,
#     train_loader=train_loader,
#     test_loader=test_loader,
#     criterion=criterion,
#     optimizer=optimizer,
#     max_epochs=20,
#     to_save='/root/autodl-tmp/competition/cpt/FURENet_Forecasting_rain'
# )

# # 运行训练
# training_template.run()

# # 创建测试模板实例
# testing_template = RainfallEstimationModelTesting(
#     model=model,
#     test_loader=test_loader,
#     to_save='/root/autodl-tmp/competition/cpt/FURENet_Forecasting_rain'
# )

# # 运行测试
# ground_truth, prediction = testing_template.run()

class LogRainfallEstimationModel(nn.Module):
    def __init__(self):
        super(LogRainfallEstimationModel, self).__init__()
        self.log_a = nn.Parameter(torch.tensor(0.0))  # 初始化 log(a) 为 0.0
        self.b = nn.Parameter(torch.tensor(1.0))  # 初始化 b 为 1.0

    def forward(self, Z_H):
        # 确保 Z_H 是正的，以避免取对数时出现问题
        log_Z_H = torch.log(Z_H + 1e-10)
        log_R = self.log_a + self.b * log_Z_H
        return log_R

def train(model, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for dbz, y in train_loader:
        dbz, y = dbz.to(device), y.to(device)
        # 确保 R 是正的，以避免取对数时出现问题
        log_R = torch.log(y + 1e-10)
        optimizer.zero_grad()
        # print(torch.isnan(dbz).any(), torch.isinf(dbz).any())  # 检查输入值
        outputs = model(dbz)
        # print(torch.isnan(outputs).any(), torch.isinf(outputs).any())  # 检查输出值
        loss = criterion(outputs, log_R)
        loss.backward()

        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(name, torch.isnan(param.grad).any(), torch.isinf(param.grad).any())  # 检查梯度值
       
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def validate(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for dbz, y in test_loader:
            dbz, y = dbz.to(device), y.to(device)
            outputs = model(dbz)
            loss = criterion(outputs, y)
            running_loss += loss.item()
    return running_loss / len(test_loader)

def main():
    model = LogRainfallEstimationModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.MSELoss()

    num_epochs = 20
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion)
        val_loss = validate(model, test_loader, criterion)
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

    print(f'Estimated a: {model.a.item()}, Estimated b: {model.b.item()}')

if __name__ == '__main__':
    main()