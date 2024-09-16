# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import os
from torchsummary import summary
from train import train
from model import MLPNet, ConvNet
#transforms是对PIL.image.image或ndarray对象处理的函数，datasets用于数据集的下载 
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
#先转换为Tensor对象，在进行均值为0.5，方差为0.5的变化

# 超参数
num_epochs = 2
batch_size = 64
learning_rate = 1e-4
device = torch.device('mps')
#device = 'cuda'

data_path = './data'
saved_model_path = './saved_model'
if not os.path.exists(saved_model_path):
    os.mkdir(saved_model_path)

train_datasets = datasets.MNIST(
    root=data_path, #下载地址为当前目录下的data文件夹，../表示当前目录的上层文件夹
    train=True,
    transform=transform,
    download=True,
)
test_datasets = datasets.MNIST(
    root=data_path,
    train=False,
    transform=transform,
    download=True,
)
# 加载数据集
train_loader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True) 
test_loader = DataLoader(test_datasets, batch_size=batch_size)

# 初始化网络实例
net = MLPNet()
#net = ConvNet()
summary(net, input_size=(28, 28), batch_size=batch_size)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# 训练网络
net = net.to(device)
train(net, train_loader, num_epochs, optimizer, criterion, device, test_loader, saved_model_path)
