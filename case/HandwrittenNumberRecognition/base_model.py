# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import os
#transforms是对PIL.image.image或ndarray对象处理的函数，datasets用于数据集的下载 
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
#先转换为Tensor对象，在进行均值为0.5，方差为0.5的变化

# 超参数
num_epochs = 3
batch_size = 64
learning_rate = 1e-4
device = torch.device('mps')
saved_model_path = "./saved_model"
if not os.path.exists(saved_model_path):
    os.mkdir(saved_model_path)

train_datasets = datasets.MNIST(
    root='./data', #下载地址为当前目录下的data文件夹，../表示当前目录的上层文件夹
    train=True,
    transform=transform,
    download=True,
)
test_datasets = datasets.MNIST(
    root='./data',
    train=False,
    transform=transform,
    download=True,
)
# 加载数据集
train_loader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True) 
test_loader = DataLoader(test_datasets, batch_size=batch_size)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)  
        self.fc2 = nn.Linear(256, 128)   
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 初始化网络实例
net = Net()
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# 训练网络
net = net.to(device)
for epoch in range(num_epochs):
    train_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        # 将图像数据展平为一维向量
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        
        # 前向传播
        outputs = net(images)
        loss = criterion(outputs, labels)
        train_loss += loss.item()

        # 反向传播并更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 打印训练信息
        if (i + 1) % 100 == 0:
            print('Epoch:{} | Step:{}/{} | Loss: {:.6f}'.format(epoch + 1, i + 1, \
                                    len(train_loader), train_loss / 100))
            train_loss = 0

    # test
    with torch.no_grad():
        true_num = 0
        for test_data, test_label in test_loader:
            test_data = test_data.reshape(-1, 28*28).to(device)
            test_label = test_label.to(device)
            test_out = net(test_data)
            # print(test_out.shape,test_label.shape) [64,10],[64]
            num = (torch.argmax(test_out, dim=1) == test_label).sum().item()
            true_num += num
            
        print('Accuracy: {:.3f}%'.format(100 * true_num / len(test_loader.dataset)))
    
    saved_epoch_model_path = os.path.join(saved_model_path, f"model{epoch + 1}.pth")
    torch.save(net, saved_epoch_model_path)