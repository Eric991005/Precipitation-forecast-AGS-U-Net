import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from util.TrainingTemplate import TrainingTemplate
from util.TestingTemplate import TestingTemplate
from FURENet.FURENet import FURENet
import random

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

class RandomDataset(Dataset):
    def __init__(self, num_samples, input_shape):
        self.num_samples = num_samples
        self.input_shape = input_shape  # Assuming input_shape is (10, 256, 256)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        z = torch.randn(self.input_shape)  # Shape: (10, 256, 256)
        zdr = torch.randn(self.input_shape)  # Shape: (10, 256, 256)
        kdp = torch.randn(self.input_shape)  # Shape: (10, 256, 256)
        label = torch.randn(self.input_shape)  # Shape: (10, 256, 256), now label has the same shape as input frames
        return z, zdr, kdp, label


# 创建数据集和数据加载器
train_dataset = RandomDataset(num_samples=1000, input_shape=(10, 256, 256))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = RandomDataset(num_samples=200, input_shape=(10, 256, 256))
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FURENet(10).to(device)
# criterion = torch.nn.CrossEntropyLoss().to(device)
criterion = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



# 创建训练模板实例
training_template = FURENetTraining(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    criterion=criterion,
    optimizer=optimizer,
    max_epochs=20,
    to_save=''
)

# 运行训练
training_template.run()

# 创建测试模板实例
testing_template = FURENetTesting(
    model=model,
    test_loader=test_loader,
    to_save=''
)

# 运行测试
ground_truth, prediction = testing_template.run()
