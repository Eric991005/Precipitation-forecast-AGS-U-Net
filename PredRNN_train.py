import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, TensorDataset,random_split
import numpy as np
from util.TrainingTemplate import TrainingTemplate
from util.TestingTemplate import TestingTemplate
# from FURENet.FURENet import FURENet
from PredRNN.PredRNN_Seq2Seq import PredRNN_enc, PredRNN_dec
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

class MyDataset(Dataset):
    def __init__(self, dbz, zdr, y, norm_param):
        self.dbz = self.normalize_data(dbz, norm_param['dBZ'])
        self.zdr = self.normalize_data(zdr, norm_param['ZDR'])
        self.y = self.normalize_data(y, norm_param['y'])  # 标签的归一化范围和dBZ相同

        # 在第二维和第三维之间增加一个维度
        self.dbz_expanded = self.dbz.unsqueeze(2)
        self.zdr_expanded = self.zdr.unsqueeze(2)
        self.y_expanded = self.y.unsqueeze(2)  # 修复了这里

        # 在新的第三维上拼接dbz和zdr
        self.concatenated_tensor = torch.cat((self.dbz_expanded, self.zdr_expanded), dim=2)

    def normalize_data(self, data, range_param):
        mmin, mmax = range_param
        normalized_data = (data - mmin) / (mmax - mmin)
        return normalized_data.clone().detach().to(dtype=torch.float32)

    def __len__(self):
        return len(self.y)  # 假设y的第一维度是样本数量

    def __getitem__(self, idx):
        return self.concatenated_tensor[idx], self.y_expanded[idx]



# 示例用法
norm_param = {
    'dBZ': [0,65],
    'ZDR': [-1,5],
    'y': [0,1015]
}


dbz = torch.tensor(np.load('/root/autodl-tmp/competition/data/all_sample/dbz_sample.npz')['data'])
zdr = torch.tensor(np.load('/root/autodl-tmp/competition/data/all_sample/zdr_sample.npz')['data'])
y = torch.tensor(np.load('/root/autodl-tmp/competition/data/all_sample/PredRNN_y_sample.npz')['data'])




dataset = MyDataset(dbz, zdr, y, norm_param)

class PredRNNTraining(TrainingTemplate):
    def __init__(self, *args, **kwargs):
        super(PredRNNTraining, self).__init__(*args, **kwargs)

    def check_data(self, data):
        # 如果需要，可以覆盖此方法以自定义数据处理
        inputs, labels = data
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        return inputs, labels

    def forward(self, inputs) -> torch.Tensor:
        # 如果需要，可以覆盖此方法以自定义前向传播过程
        return super(PredRNNTraining, self).forward(inputs)

    def set_states(self, epoch, loss):
        # 如果需要，可以覆盖此方法以自定义保存状态的内容
        return super(PredRNNTraining, self).set_states(epoch, loss)

    def is_best(self, loss):
        # 如果需要，可以覆盖此方法以自定义最优模型的判断条件
        return super(PredRNNTraining, self).is_best(loss)

    def run(self):
        # 如果需要，可以覆盖此方法以自定义整个训练过程
        super(PredRNNTraining, self).run()


class PredRNNTesting(TestingTemplate):
    def __init__(self, *args, **kwargs):
        super(PredRNNTesting, self).__init__(*args, **kwargs)

    def check_data(self, data):
        inputs, labels = data
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        return inputs, labels

    def check_data_back(self, labels, outputs):
        # 如果需要，可以覆盖此方法以自定义输出和标签的处理
        return super(PredRNNTesting, self).check_data_back(labels, outputs)

    def forward(self, inputs, out_len: int = 10) -> torch.Tensor:
        # 如果需要，可以覆盖此方法以自定义前向传播过程
        return super(PredRNNTesting, self).forward(inputs, out_len=out_len)

    def load(self):
        # 如果需要，可以覆盖此方法以自定义模型加载过程
        super(PredRNNTesting, self).load()

    def run(self, out_len: int = 10):
        # 如果需要，可以覆盖此方法以自定义测试运行过程
        return super(PredRNNTesting, self).run(out_len=out_len)

    def test(self, out_len: int = 10):
        # 如果需要，可以覆盖此方法以自定义测试过程
        return super(PredRNNTesting, self).test(out_len=out_len)





###############################################
# 分割数据集

train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# 定义模型
# 假设你的模型是 PredRNN_enc 和 PredRNN_dec 的组合
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.enc = PredRNN_enc().to(device)  # 假设你已经定义了 PredRNN_enc
        self.dec = PredRNN_dec().to(device)  # 假设你已经定义了 PredRNN_dec

    def forward(self, x, out_len=None):
        enc_hidden, enc_h_m = self.enc(x)
        out, _, _ = self.dec(x, enc_hidden, enc_h_m[-1])
        return out



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModel().to(device)
# criterion = torch.nn.CrossEntropyLoss().to(device)
criterion = torch.nn.MSELoss().to(device)
# criterion = BMSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



# 创建PredRNNTraining实例
pred_rnn_training = PredRNNTraining(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    criterion=criterion,
    optimizer=optimizer,
    max_epochs=10,
    to_save="cpt/PredRNN",
    # test_frequency=1,
    # start_save=0,
    # visualize=False
)

# 运行训练
pred_rnn_training.run()

# 创建PredRNNTesting实例
pred_rnn_testing = PredRNNTesting(
    model=model,
    test_loader=test_loader,
    device="cuda",
    to_save="cpt/PredRNN"
)

# 运行测试
ground_truth, prediction = pred_rnn_testing.run(out_len=10)
