# Pytorch学习笔记

## Pytorch安装

pytorch官网地址: https://pytorch.org/get-started/locally/
mac安装命令: `pip3 install torch torchvision torchaudio`

## 测试安装是否成功

```python
import torch
import torchvision
import torchaudio
print("python version: ", torch.__version__)
print("if MacOs GPU is available: ", torch.device('mps'))
```
## 神经网络可视化工具-Netron
安装地址: https://github.com/lutzroeder/netron/releases
```python
import torch

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Linear(28*28, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10)
        )

    def forward(self, input):
        logits = self.block(input)
        return logits

model = Net()
torch.save(model, './model.pth')
```
保存模型为pth格式, 在Netron中打开该模型文件，得到可视化的模型结构
