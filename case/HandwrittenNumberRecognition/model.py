import torch.nn as nn

class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_block = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, input):
        x = self.flatten(input)
        logits = self.linear_relu_block(x)
        return logits

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # 特征提取网络设置
        self.conv1 = self.conv_block(1, 32, 3, 2)
        self.conv2 = self.conv_block(32, 64, 3, 2)
       
        # 分类网络设置
        self.fc = nn.Sequential(
            nn.Linear(1600, 64), # 降维，1600-->64维
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def conv_block(self, input_channel, output_channel, kernel_size, pool_size):
        return nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(pool_size)
        )

    # 前向传播
    def forward(self, input):
        if input.dim() == 3:
            input = input.unsqueeze(1)
        x = self.conv1(input)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x