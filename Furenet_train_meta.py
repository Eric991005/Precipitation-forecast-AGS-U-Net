import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, TensorDataset,random_split
import numpy as np
from util.TrainingTemplate import TrainingTemplate
from util.TestingTemplate import TestingTemplate
from FURENet.Attention_FURENet_meta import FURENetMeta
import random
import pickle
from collections import defaultdict
import os
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error
import csv
from learn2learn.data.transforms import TaskTransform
from learn2learn.data.task_dataset import DataDescription
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels, ConsecutiveLabels, FusedNWaysKShots
import learn2learn as l2l
import time
import datetime

def map_multidim_to_scalar(label_tensor):
    # 这是一个示例函数，你需要根据你的需求来实现它
    return hash(str(label_tensor.tolist()))

class MyFusedNWaysKShots(TaskTransform):

    def __init__(self, dataset, meta_bsz):
        super(MyFusedNWaysKShots, self).__init__(dataset)
        self.meta_bsz = meta_bsz

    def new_task(self):
        task_size = len(self.dataset) // self.meta_bsz
        task_description = []
        for i in range(0, len(self.dataset), task_size):
            if i + task_size > len(self.dataset):  # If the remaining tasks are less than task_size
                remaining = len(self.dataset) - i
                for j in range(i, i + remaining):
                    dd = DataDescription(j)
                    dd.transforms.append(lambda x: self.dataset[x])  # Add a transform to load the data
                    task_description.append(dd)
                # Randomly sample tasks to fill the last batch
                for j in random.sample(range(len(self.dataset)), task_size - remaining):
                    dd = DataDescription(j)
                    dd.transforms.append(lambda x: self.dataset[x])  # Add a transform to load the data
                    task_description.append(dd)
                break
            for j in range(i, i + task_size):
                dd = DataDescription(j)
                dd.transforms.append(lambda x: self.dataset[x])  # Add a transform to load the data
                task_description.append(dd)
        return task_description

    def __call__(self, task_description):
        if task_description is None:
            task_description = self.new_task()
        task_description = [DataDescription(dd.index) for dd in task_description]
        return task_description

# 定义一个函数来计算每个像素的权重
def compute_weights(x):
    weights = torch.ones_like(x)
    weights[x >= 2] = 2
    weights[x >= 5] = 5
    weights[x >= 10] = 10
    weights[x >= 35] = 50
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
        x = (self.dbz[idx], self.kdp[idx], self.zdr[idx])
        y = map_multidim_to_scalar(self.y[idx])
        return x, y
    
class MetaDataset(Dataset):
    """

    **Description**

    Wraps a classification dataset to enable fast indexing of samples within classes.

    This class exposes two attributes specific to the wrapped dataset:

    * `labels_to_indices`: maps a class label to a list of sample indices with that label.
    * `indices_to_labels`: maps a sample index to its corresponding class label.

    Those dictionary attributes are often used to quickly create few-shot classification tasks.
    They can be passed as arguments upon instantiation, or automatically built on-the-fly.
    If the wrapped dataset has an attribute `_bookkeeping_path`, then the built attributes will be cached on disk and reloaded upon the next instantiation.
    This caching strategy is useful for large datasets (e.g. ImageNet-1k) where the first instantiation can take several hours.

    Note that if only one of `labels_to_indices` or `indices_to_labels` is provided, this class builds the other one from it.

    **Arguments**

    * **dataset** (Dataset) -  A torch Dataset.
    * **labels_to_indices** (dict, **optional**, default=None) -  A dictionary mapping labels to the indices of their samples.
    * **indices_to_labels** (dict, **optional**, default=None) -  A dictionary mapping sample indices to their corresponding label.

    **Example**
    ~~~python
    mnist = torchvision.datasets.MNIST(root="/tmp/mnist", train=True)
    mnist = l2l.data.MetaDataset(mnist)
    ~~~
    """

    def __init__(self, dataset, labels_to_indices=None, indices_to_labels=None):

        if not isinstance(dataset, Dataset):
            raise TypeError(
                "MetaDataset only accepts a torch dataset as input")

        self.dataset = dataset

        if hasattr(dataset, '_bookkeeping_path'):
            self.load_bookkeeping(dataset._bookkeeping_path)
        else:
            self.create_bookkeeping(
                labels_to_indices=labels_to_indices,
                indices_to_labels=indices_to_labels,
            )

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)

    def create_bookkeeping(self, labels_to_indices=None, indices_to_labels=None):
        """
        Iterates over the entire dataset and creates a map of target to indices.

        Returns: A dict with key as the label and value as list of indices.
        """

        assert hasattr(self.dataset, '__getitem__'), \
            'Requires iterable-style dataset.'

        # Bootstrap from arguments
        if labels_to_indices is not None:
            indices_to_labels = {
                idx: label
                for label, indices in labels_to_indices.items()
                for idx in indices
            }
        elif indices_to_labels is not None:
            labels_to_indices = defaultdict(list)
            for idx, label in indices_to_labels.items():
                labels_to_indices[label].append(idx)
        else:  # Create from scratch
            labels_to_indices = defaultdict(list)
            indices_to_labels = defaultdict(int)
            for i in range(len(self.dataset)):
                try:
                    label = self.dataset[i][1]
                    # if label is a Tensor, then take get the scalar value
                    if hasattr(label, 'item'):
                        label = self.dataset[i][1].item()
                except ValueError as e:
                    raise ValueError(
                        'Requires scalar labels. \n' + str(e))

                labels_to_indices[label].append(i)
                indices_to_labels[i] = label

        self.labels_to_indices = labels_to_indices
        self.indices_to_labels = indices_to_labels
        self.labels = list(self.labels_to_indices.keys())

        self._bookkeeping = {
            'labels_to_indices': self.labels_to_indices,
            'indices_to_labels': self.indices_to_labels,
            'labels': self.labels
        }

    def load_bookkeeping(self, path):
        if not os.path.exists(path):
            self.create_bookkeeping()
            self.serialize_bookkeeping(path)
        else:
            with open(path, 'rb') as f:
                self._bookkeeping = pickle.load(f)
            self.labels_to_indices = self._bookkeeping['labels_to_indices']
            self.indices_to_labels = self._bookkeeping['indices_to_labels']
            self.labels = self._bookkeeping['labels']

    def serialize_bookkeeping(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self._bookkeeping, f, protocol=-1)

class MetaFURENetDataset(MyDataset, MetaDataset):
    def __init__(self, *args, **kwargs):
        MyDataset.__init__(self, *args, **kwargs)
        MetaDataset.__init__(self, self)

    def __getitem__(self, index):
        x, y = MyDataset.__getitem__(self, index)
        return x, y

    def __len__(self):
        return MyDataset.__len__(self)
    
def fast_adapt(batch, learner, features, adapt_steps, device):
    bmseloss = BMSELoss()
    data, labels = batch
    data = [x.float() for x in data]
    labels = [x.float() for x in labels]
    dbz, kdp, zdr = data
    dbz, kdp, zdr = dbz.to(device), kdp.to(device), zdr.to(device)
    y = labels
    y = y.to(device)

    y_hat = features( dbz, kdp, zdr )

    # 确定适应集和评估集的大小
    n_total = len(dbz)
    n_adaptation = n_total // 2  # 假设我们将一半的数据用于适应集，一半用于评估集

    # 使用不放回抽样来随机选择适应集的索引
    adaptation_indices = np.random.choice(n_total, size=n_adaptation, replace=False)

    # 创建一个布尔数组来标记哪些样本属于适应集
    is_adaptation = np.zeros(n_total, dtype=bool)
    is_adaptation[adaptation_indices] = True

    # 创建一个布尔数组来标记哪些样本属于评估集
    is_evaluation = ~is_adaptation

    adaptation_y_hat= y_hat[adaptation_indices]
    adaptation_y = y[adaptation_indices]
    evaluation_y_hat = y_hat[is_evaluation]
    evaluation_y = y[is_evaluation]

    # Adapt the model
    for step in range(adapt_steps):
        y_hat = learner(adaptation_y_hat)
        adaptation_error = bmseloss(adaptation_y, y_hat)

        # Check if loss is too large
        if torch.isnan(adaptation_error):
            print("Error is NaN, stopping training")
            return 

        if torch.isinf(adaptation_error):
            print("Error is Infinity, stopping training")
            return 

        learner.adapt(adaptation_error)

    # Evaluate the adapted model
    y_hat = learner(evaluation_y_hat)
    evaluation_error = bmseloss(evaluation_y, y_hat)
    return evaluation_error, evaluation_y, y_hat

def fast_adapt_test(batch, learner, features, mean, std, device):
    bmseloss = BMSELoss()
    data, labels = batch
    data = [x.float() for x in data]
    labels = [x.float() for x in labels]
    dbz, kdp, zdr = data
    dbz, kdp, zdr = dbz.to(device), kdp.to(device), zdr.to(device)
    y = labels
    y = y.to(device)

    y_hat = features( dbz, kdp, zdr )

    # Evaluate the model
    y_hat = learner(y_hat)
    evaluation_error = bmseloss(y, y_hat)
    return evaluation_error, y, y_hat

# 归一化参数
norm_param = {
    'dBZ': [0, 65],
    'ZDR': [-1, 5],
    'KDP': [-1, 6]
}

# model = FURENetMeta(10).to(device)
# # criterion = torch.nn.CrossEntropyLoss().to(device)
# # criterion = torch.nn.MSELoss().to(device)
# criterion = BMSELoss().to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



# # 创建训练模板实例
# training_template = FURENetTraining(
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
# testing_template = FURENetTesting(
#     model=model,
#     test_loader=test_loader,
#     to_save='/root/autodl-tmp/competition/cpt/FURENet_Forecasting_rain'
# )

# # 运行测试
# ground_truth, prediction = testing_template.run()

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_elements = y_true != 0  # 避免除以零
    return np.mean(np.abs((y_true[non_zero_elements] - y_pred[non_zero_elements]) / y_true[non_zero_elements])) * 100

def save_to_csv(data, file_path):
    # 写入 CSV 文件
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in data:
            writer.writerow(row)

    print(f"已保存数据到 {file_path}")

def train_meta(features, head, train_tasks, val_tasks, meta_lr, fast_lr, adapt_steps, meta_bsz, max_epoch, device):
    all_parameters = list(features.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adamax(all_parameters, lr=meta_lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=300,    
                                verbose=False, threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=2e-6, eps=1e-08)

    maes = []
    rmses = []
    mapes = []
    avg_maes = []
    avg_rmses = []
    avg_mapes = []

    for epoch in tqdm(range(1, max_epoch+1)):
        optimizer.zero_grad()
        meta_train_error = 0.0
        meta_valid_error = 0.0
        for task in range(meta_bsz):
            # Compute meta-training loss
            learner = head.clone()
            # Sample a task from the TaskDataset
            batch = train_tasks.sample()
            evaluation_error, evaluation_Y, y_hat = fast_adapt(batch,
                                           learner,
                                           features,
                                           adapt_steps,
                                           device)
            evaluation_error.backward()
            meta_train_error += evaluation_error.item()

            # Compute meta-validation loss
            learner = head.clone()

            # Sample a task from the TaskDataset
            batch = val_tasks.sample()

            evaluation_error, evaluation_Y, y_hat = fast_adapt(batch,
                                        learner,
                                        features,
                                        adapt_steps,
                                        device)
            meta_valid_error += evaluation_error.item()

            label = evaluation_Y.detach().cpu().numpy()
            pred = y_hat.detach().cpu().numpy()
            mae = mean_absolute_error(label, pred)
            rmse = np.sqrt(mean_squared_error(label, pred))
            mape = mean_absolute_percentage_error(label, pred)

            maes.append(mae)
            rmses.append(rmse)
            mapes.append(mape)

        avg_mae = np.mean(maes)
        avg_rmse = np.mean(rmses)
        avg_mape = np.mean(mapes)

        avg_maes.append(avg_mae)
        avg_rmses.append(avg_rmse)
        avg_mapes.append(avg_mape)

        print('iters {}, lr {:.6f}'.format(epoch, optimizer.param_groups[0]['lr']))
        print('Meta Train Error: {:.4f}'.format(meta_train_error / meta_bsz))
        print('Meta Valid Error: {:.4f}'.format(meta_valid_error / meta_bsz))
        print('Valid step %d, mae: %.4f, rmse: %.4f, mape: %.4f' % (epoch, avg_mae, avg_rmse, avg_mape))


        # Average the accumulated gradients and optimize
        for p in all_parameters:
            if p.grad is not None:
                p.grad.data.mul_(1.0 / meta_bsz)

        optimizer.step()

        # lr_scheduler.step(meta_valid_error / meta_bsz)
        lr_scheduler.step(avg_maes[-1])

    torch.save({
        'features': features.state_dict(),
        'head': head.state_dict()
    }, '/root/autodl-tmp/competition/cpt/FURENet_Meta')    

    return avg_maes, avg_rmses, avg_mapes

def test_meta(model_file, test_tasks, adapt_steps, mean, std, device):
    state = torch.load(model_file)
    features.load_state_dict(state['features'])
    head.load_state_dict(state['head'])
    print('Successfully loaded model')

    meta_test_error = 0.0
    meta_bsz = len(test_tasks)

    maes = []
    rmses = []
    mapes = []

    for task in range(meta_bsz):
        # Compute meta-test loss
        learner = head.clone()

        # Sample a task from the TaskDataset
        batch = test_tasks.sample()

        evaluation_error, evaluation_Y, y_hat = fast_adapt_test(batch,
                                      learner,
                                      features,
                                      device)
        meta_test_error += evaluation_error.item()

        label = evaluation_Y.detach().cpu().numpy()
        pred = y_hat.detach().cpu().numpy()*std+mean
        mae = mean_absolute_error(label, pred)
        rmse = np.sqrt(mean_squared_error(label, pred))
        mape = mean_absolute_percentage_error(label, pred)

        # 定义数据和文件路径的列表
        data_list = [pred,label]
        file_paths = [
            '/root/autodl-tmp/MYSTWave/output/import_meta_pred_out/first_pred.csv',
            '/root/autodl-tmp/MYSTWave/output/import_meta_pred_out/second_pred.csv',
        ]
        # 遍历列表并保存数据到 CSV 文件
        for data, file_path in zip(data_list, file_paths):
            save_to_csv(data, file_path)        

    avg_mae = np.mean(maes)
    avg_rmse = np.mean(rmses)
    avg_mape = np.mean(mapes)

    print('Meta Test Error: {:.4f}'.format(meta_test_error / meta_bsz))
    print('Test, mae: %.4f, rmse: %.4f, mape: %.4f' % (avg_mae, avg_rmse, avg_mape))

if __name__ == '__main__':
    meta_lr=1e-3
    fast_lr=1e-10
    adapt_steps=5
    meta_bsz=3
    iters=30000
    cuda=1
    seed=42
    picture_dir = "/root/autodl-tmp/MYSTWave/picture"
    print( "loading data....")
    # 加载你的数据
    dbz = np.load('/root/autodl-tmp/competition/data/all_sample/dbz_sample.npz')['data']
    kdp = np.load('/root/autodl-tmp/competition/data/all_sample/kdp_sample.npz')['data']
    zdr = np.load('/root/autodl-tmp/competition/data/all_sample/zdr_sample.npz')['data']
    y = np.load('/root/autodl-tmp/competition/data/all_sample/y_sample.npz')['data']
    seed = 42
    generator = torch.Generator().manual_seed(seed)
    model_file = '/root/autodl-tmp/competition/cpt/FURENet_Meta/saved_model'

    # 创建数据集实例
    dataset = MetaFURENetDataset(dbz, kdp, zdr, y, norm_param)
    # Determine the size of each split
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size  # To ensure the sum of sizes equals the total size

    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=generator)



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transforms = [
        MyFusedNWaysKShots(train_dataset, meta_bsz),  # Replace 32 with your desired batch size
        LoadData(train_dataset),
    ]

    train_tasks = l2l.data.TaskDataset(train_dataset,
                                       task_transforms=train_transforms,
                                       num_tasks=20000) 
    valid_transforms = [
        MyFusedNWaysKShots(val_dataset, meta_bsz),
        LoadData(val_dataset),
    ]
    valid_tasks = l2l.data.TaskDataset(val_dataset,
                                       task_transforms=valid_transforms,
                                       num_tasks=600)

    test_transforms = [
        MyFusedNWaysKShots(test_dataset, meta_bsz),
        LoadData(test_dataset),
    ]
    test_tasks = l2l.data.TaskDataset(test_dataset,
                                      task_transforms=test_transforms,
                                      num_tasks=1)
    print("loading end....")

    print("constructing model begin....")
    # Create model
    model = FURENetMeta(10).to(device)
    features = model.features
    head = model.classifier
    features.to(device)
    head = l2l.algorithms.MAML(head, lr=fast_lr)
    head.to(device)
    print("constructing model end....")


    print("training begin....")
    start_time = time.time()  # 获取训练开始时间
    start_datetime = datetime.datetime.now()  # Get the current date and time
    print("Training started on: {}".format(start_datetime.strftime("%Y-%m-%d %H:%M:%S")))
    # avg_val_maes, avg_val_rmses, avg_val_mapes, avg_test_maes, avg_test_rmses, avg_test_mapes = train_meta(features, head, train_tasks, valid_tasks, test_tasks, meta_lr, fast_lr, adapt_steps, meta_bsz, iters, train_dataset.mean, train_dataset.std, device)
    avg_maes, avg_rmses, avg_mapes = train_meta(features, head, train_tasks, valid_tasks, meta_lr, fast_lr, adapt_steps, meta_bsz, iters, device)
    os.makedirs(picture_dir, exist_ok=True)
    # plot_metrics(avg_maes, avg_rmses, avg_mapes, iters, picture_dir)
    # plot_metrics(avg_val_maes, avg_val_rmses, avg_val_mapes, avg_test_maes, avg_test_rmses, avg_test_mapes, iters, picture_dir)
    end_time = time.time()  # 获取训练结束时间
    training_time = end_time - start_time  # 计算训练时长，单位为秒
    print( "training end....")
    print("Training duration: {:.2f} seconds".format(training_time))

    print("testing begin....")
    # test_meta(args.model_file, test_tasks, adapt_steps, train_dataset.mean, train_dataset.std, device)
    test_meta(model_file,test_tasks, adapt_steps, device)
    print("testing end....")